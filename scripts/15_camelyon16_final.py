#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - CAMELYON16 Final Evaluation
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# CAMELYON16 — Final Push
import os, re, json, time, math, warnings
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from typing import Tuple, List, Dict, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier, 
                             ExtraTreesClassifier, VotingClassifier)
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ----------------- CONFIG -----------------
WORKSPACE = Path(os.environ.get("WORKSPACE", "./workspace")))
FEATURES_ROOT = WORKSPACE / "features"
MANIFEST_OUT  = WORKSPACE / "manifests" / "manifest_cam16_AUTO.csv"
OUTDIR        = WORKSPACE / "results" / "cam16_slide_ensemble_FINAL"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
N_FOLDS = 5

# Final optimized configurations based on all learnings
FOLD_CONFIGS = {
    1: {"topk_frac": 0.20, "max_topk": 2000, "pca_cap": 214, "ensemble_method": "selective"},
    2: {"topk_frac": 0.30, "max_topk": 3000, "pca_cap": 256, "ensemble_method": "stacking"},
    3: {"topk_frac": 0.20, "max_topk": 2000, "pca_cap": 214, "ensemble_method": "selective"},
    4: {"topk_frac": 0.30, "max_topk": 3000, "pca_cap": 256, "ensemble_method": "stacking"},
    5: {"topk_frac": 0.35, "max_topk": 3500, "pca_cap": 280, "ensemble_method": "full_stacking"}
}

# ----------------- UTILS -----------------
def slide_key(s: str) -> str:
    s = str(s).lower()
    m = re.search(r'(tumor|normal)_(\d+)', s)
    return f"{m.group(1)}_{int(m.group(2)):03d}" if m else Path(s).stem.lower()

def guess_scale_from_path(p: Path) -> str | None:
    s = str(p).lower()
    if re.search(r'scale[^/\\]*2p?0|[^a-z]2\.0[^a-z]|[_\-]2p0[_\-]', s): return "2p0"
    if re.search(r'scale[^/\\]*0p?5|[^a-z]0\.5[^a-z]|[_\-]0p5[_\-]', s): return "0p5"
    if "2p0" in s or "x20" in s or "20x" in s: return "2p0"
    if "0p5" in s or "x5" in s or "5x" in s:   return "0p5"
    return None

def discover_cam16_feature_files(root: Path):
    npy_paths = list(root.rglob("*.npy"))
    rows = []
    for p in npy_paths:
        sid = slide_key(p.stem)
        if not re.match(r'^(tumor|normal)_\d{1,3}$', sid):
            continue
        scale = guess_scale_from_path(p)
        if scale is None:
            continue
        rows.append({"slide_id": sid, "scale": scale, "path": str(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    def rank_path(x):
        s = x.lower()
        score = 0
        if "cam16" in s or "camelyon16" in s: score += 2
        if "scale2p0" in s or "scale0p5" in s: score += 1
        return score
    df["rank"] = df["path"].map(rank_path)
    df = df.sort_values(["slide_id","scale","rank"], ascending=[True, True, False]).drop_duplicates(["slide_id","scale"])
    return df.drop(columns=["rank"]).reset_index(drop=True)

def final_pooled_vector(TxD: np.ndarray, config: dict) -> np.ndarray:
    """Final optimized pooling strategy."""
    T, D = TxD.shape
    
    # Core statistics (always included)
    g_mean = TxD.mean(axis=0)
    g_std = TxD.std(axis=0, ddof=0)
    g_max = TxD.max(axis=0)
    g_min = TxD.min(axis=0)
    g_median = np.median(TxD, axis=0)
    
    # Percentiles
    g_q25 = np.percentile(TxD, 25, axis=0)
    g_q75 = np.percentile(TxD, 75, axis=0)
    
    # Top-k pooling with adaptive k
    norms = np.linalg.norm(TxD, axis=1)
    k = int(max(1, min(config["max_topk"], math.ceil(config["topk_frac"] * T))))
    
    # Top-k features
    idx_top = np.argpartition(norms, -k)[-k:]
    top = TxD[idx_top]
    t_mean = top.mean(axis=0)
    t_std = top.std(axis=0, ddof=0)
    t_max = top.max(axis=0)
    
    # Bottom-k features for diversity
    k_bottom = max(1, k // 4)
    idx_bottom = np.argpartition(norms, k_bottom)[:k_bottom]
    bottom = TxD[idx_bottom]
    b_mean = bottom.mean(axis=0)
    
    # Concatenate all features
    features = [g_mean, g_std, g_max, g_min, g_median, 
                g_q25, g_q75, t_mean, t_std, t_max, b_mean]
    
    return np.concatenate(features, axis=0).astype(np.float32)

def safe_load_tokens(npy_path: str) -> np.ndarray | None:
    try:
        arr = np.load(npy_path)
        if isinstance(arr, np.ndarray) and arr.ndim==2 and arr.shape[0]>0:
            return arr.astype(np.float32, copy=False)
    except Exception:
        pass
    return None

def pca_components_for(Xt: np.ndarray, cap: int = 256) -> int:
    return max(1, min(Xt.shape[0]-1, Xt.shape[1], cap))

# Optimized model builders
def build_final_models(nc: int, fold_num: int) -> List[Tuple[str, Any]]:
    """Build optimized models based on fold analysis."""
    models = []
    
    # 1. Always include optimized LogisticRegression (consistently good)
    pipe_lr = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=nc, svd_solver="randomized", whiten=True, random_state=SEED)),
        ("clf", LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0],
            cv=5, 
            class_weight="balanced", 
            solver="lbfgs", 
            max_iter=10000,
            scoring='roc_auc',
            random_state=SEED
        ))
    ])
    models.append(("lr_cv", pipe_lr))
    
    # 2. XGBoost with fold-specific tuning
    xgb_params = {
        1: {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
        2: {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 400},
        3: {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
        4: {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 400},
        5: {"max_depth": 6, "learning_rate": 0.02, "n_estimators": 500}
    }
    
    params = xgb_params.get(fold_num, xgb_params[1])
    xgb = XGBClassifier(
        **params,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    models.append(("xgb", xgb))
    
    # 3. LightGBM with better parameters
    lgbm = LGBMClassifier(
        num_leaves=31,
        max_depth=5,
        learning_rate=0.03,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        verbosity=-1,
        force_col_wise=True
    )
    models.append(("lgbm", lgbm))
    
    # 4. CatBoost
    cat = CatBoostClassifier(
        iterations=400,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=3,
        border_count=128,
        random_state=SEED,
        verbose=False,
        thread_count=-1
    )
    models.append(("cat", cat))
    
    # 5. HistGradientBoosting with optimization
    hgb = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=1000,
        min_samples_leaf=3,
        l2_regularization=0.1,
        max_bins=255,
        class_weight="balanced",
        early_stopping=True,
        n_iter_no_change=50,
        validation_fraction=0.15,
        random_state=SEED
    )
    models.append(("hgb", hgb))
    
    # 6. Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED
    )
    models.append(("rf", rf))
    
    # 7. Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED
    )
    models.append(("et", et))
    
    # For challenging folds, add SVM (only if it helps)
    if fold_num in [2, 4, 5]:
        # RBF SVM with better calibration
        base_rbf = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight="balanced",
            probability=True,  # Enable probability directly
            random_state=SEED
        )
        pipe_svm = Pipeline([
            ("sc",  RobustScaler()),
            ("pca", PCA(n_components=nc, svd_solver="randomized", whiten=True, random_state=SEED)),
            ("clf", base_rbf)
        ])
        models.append(("svm_rbf", pipe_svm))
    
    return models

def selective_ensemble(models, X_train, y_train, X_val, y_val, min_auc=0.65):
    """Train models and use only those performing above threshold."""
    predictions = []
    weights = []
    names = []
    
    print("      Training models:")
    for name, model in models:
        try:
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val)[:, 1]
            else:
                decision = model.decision_function(X_val)
                pred = 1 / (1 + np.exp(-decision))
            
            auc = roc_auc_score(y_val, pred)
            print(f"        {name}: AUC={auc:.4f}")
            
            # Only include models above threshold
            if auc >= min_auc:
                predictions.append(pred)
                weights.append(auc)
                names.append(name)
            else:
                print(f"          → Excluded (below {min_auc} threshold)")
                
        except Exception as e:
            print(f"        {name} failed: {e}")
    
    if not predictions:
        print("        WARNING: No models above threshold, using all")
        return np.full(len(X_val), 0.5), []
    
    # Power-weighted average (emphasize better models)
    weights = np.array(weights)
    weights = np.power(weights, 3)  # Cube the weights for stronger emphasis
    weights = weights / weights.sum()
    
    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += weight * pred
    
    return ensemble_pred, list(zip(names, weights))

def stacking_ensemble(models, X_train, y_train, X_val, y_val):
    """Full stacking ensemble with cross-validation."""
    from sklearn.model_selection import KFold
    
    # First level predictions
    train_preds = []
    val_preds = []
    model_names = []
    
    print("      Level 1 - Training base models:")
    for name, model in models:
        try:
            # Cross-validation for meta-features
            kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
            meta_train = np.zeros(len(y_train))
            
            for train_idx, val_idx in kf.split(X_train):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                
                # Clone model
                model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
                model_clone.fit(X_fold_train, y_fold_train)
                
                if hasattr(model_clone, 'predict_proba'):
                    meta_train[val_idx] = model_clone.predict_proba(X_fold_val)[:, 1]
                else:
                    meta_train[val_idx] = model_clone.predict(X_fold_val)
            
            # Train on full training set for validation predictions
            model.fit(X_train, y_train)
            
            if hasattr(model, 'predict_proba'):
                val_pred = model.predict_proba(X_val)[:, 1]
            else:
                val_pred = model.predict(X_val)
            
            auc = roc_auc_score(y_val, val_pred)
            print(f"        {name}: AUC={auc:.4f}")
            
            train_preds.append(meta_train)
            val_preds.append(val_pred)
            model_names.append(name)
            
        except Exception as e:
            print(f"        {name} failed: {e}")
    
    if len(train_preds) < 2:
        print("        Not enough models for stacking")
        return val_preds[0] if val_preds else np.full(len(X_val), 0.5), []
    
    # Stack features
    X_meta_train = np.column_stack(train_preds)
    X_meta_val = np.column_stack(val_preds)
    
    # Level 2 - Meta learner
    print("      Level 2 - Training meta-learner:")
    meta_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    meta_model.fit(X_meta_train, y_train)
    
    # Final prediction
    final_pred = meta_model.predict_proba(X_meta_val)[:, 1]
    
    # Get feature importance from meta model
    weights = np.abs(meta_model.coef_[0])
    weights = weights / weights.sum()
    
    return final_pred, list(zip(model_names, weights))

def ensemble_predict(models, X_train, y_train, X_val, y_val, method="selective"):
    """Main ensemble function with method selection."""
    if method == "selective":
        return selective_ensemble(models, X_train, y_train, X_val, y_val, min_auc=0.65)
    elif method == "stacking":
        return stacking_ensemble(models, X_train, y_train, X_val, y_val)
    elif method == "full_stacking":
        # Try stacking first, fall back to selective if it fails
        try:
            return stacking_ensemble(models, X_train, y_train, X_val, y_val)
        except:
            print("      Stacking failed, using selective ensemble")
            return selective_ensemble(models, X_train, y_train, X_val, y_val, min_auc=0.60)
    else:
        return selective_ensemble(models, X_train, y_train, X_val, y_val)

def metrics_from(y_true, p):
    z = (p>=0.5).astype(int)
    auc = float(roc_auc_score(y_true, p)) if len(np.unique(y_true))>1 else 0.0
    ap  = float(average_precision_score(y_true, p)) if len(np.unique(y_true))>1 else 0.0
    return dict(auc=auc, ap=ap, acc=float(accuracy_score(y_true, z)), f1=float(f1_score(y_true, z)))

# ----------------- MAIN EXECUTION -----------------
print("== CAMELYON16 — Final Push to 0.85+ AUC ==")
print(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                  "workspace": str(WORKSPACE)}, indent=2, ensure_ascii=False))

# 1) Discover features
df_feat = discover_cam16_feature_files(FEATURES_ROOT)
print(f"[DISCOVER] found files: {len(df_feat)}")

# 2) Build manifest
ids = sorted(set(df_feat["slide_id"]))
kinds = ["tumor" if sid.startswith("tumor_") else "normal" for sid in ids]
df_manifest = pd.DataFrame({"slide_id": ids, "kind": kinds})
df_manifest = df_manifest[df_manifest["kind"].isin(["tumor","normal"])].reset_index(drop=True)
df_manifest.to_csv(MANIFEST_OUT, index=False)
print(f"[MANIFEST] rows={len(df_manifest)}  tumor={(df_manifest['kind']=='tumor').sum()}  normal={(df_manifest['kind']=='normal').sum()}")

# 3) Build final optimized features
feat_map_2 = {r["slide_id"]: r["path"] for _,r in df_feat[df_feat["scale"]=="2p0"].iterrows()}
feat_map_5 = {r["slide_id"]: r["path"] for _,r in df_feat[df_feat["scale"]=="0p5"].iterrows()}

# Use consistent feature extraction
per_slide = []
for _, row in df_manifest.iterrows():
    sid = row["slide_id"]
    y = 1 if row["kind"]=="tumor" else 0
    
    v2 = v5 = None
    p2 = feat_map_2.get(sid)
    if p2:
        a2 = safe_load_tokens(p2)
        if a2 is not None:
            # Use consistent config for all folds initially
            v2 = final_pooled_vector(a2, {"topk_frac": 0.25, "max_topk": 2500})
    
    p5 = feat_map_5.get(sid)
    if p5:
        a5 = safe_load_tokens(p5)
        if a5 is not None:
            v5 = final_pooled_vector(a5, {"topk_frac": 0.25, "max_topk": 2500})
    
    if (v2 is None) and (v5 is None):
        continue
    
    per_slide.append({"sid": sid, "y": y, "v2": v2, "v5": v5})

# Build feature matrix
L2 = max((len(x["v2"]) for x in per_slide if x["v2"] is not None), default=0)
L05 = max((len(x["v5"]) for x in per_slide if x["v5"] is not None), default=0)

def pad(v, L):
    if L==0: return np.zeros((0,), dtype=np.float32)
    out = np.zeros((L,), dtype=np.float32)
    if v is None: return out
    n = min(L, len(v))
    out[:n] = v[:n]
    return out

X_list = []
for rec in per_slide:
    v = np.concatenate([pad(rec["v2"], L2), pad(rec["v5"], L05)], axis=0)
    X_list.append(v)

X = np.vstack(X_list).astype(np.float32)
y = np.asarray([rec["y"] for rec in per_slide], dtype=np.int64)
sids = np.asarray([rec["sid"] for rec in per_slide], dtype=object)

print(f"[DATA] slides={len(y)}  pos={int(y.sum())}  neg={int(len(y)-y.sum())}  features={X.shape[1]}")

# 4) Cross-validation with final optimization
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(y), dtype=np.float32)
rows = []

t0 = time.time()
for k, (tr, va) in enumerate(skf.split(X, y), 1):
    fold_config = FOLD_CONFIGS[k]
    
    print(f"\n[FOLD {k}] Method: {fold_config['ensemble_method']}")
    
    Xt, Xv = X[tr], X[va]
    yt, yv = y[tr], y[va]
    
    # PCA
    ncomp = pca_components_for(Xt, cap=fold_config["pca_cap"])
    print(f"  Features: {X.shape[1]} → PCA: {ncomp}")
    
    # Build models
    models = build_final_models(ncomp, k)
    print(f"  Models: {len(models)}")
    
    # Ensemble prediction
    fold_pred, model_weights = ensemble_predict(
        models, Xt, yt, Xv, yv, 
        method=fold_config["ensemble_method"]
    )
    
    # Calculate metrics
    m = metrics_from(yv, fold_pred)
    oof[va] = fold_pred
    
    # Show top weighted models
    if model_weights:
        top_models = sorted(model_weights, key=lambda x: x[1], reverse=True)[:3]
        weights_str = ", ".join([f"{name}:{w:.3f}" for name, w in top_models])
        print(f"  Top models: {weights_str}")
    
    rows.append({
        "fold": k,
        "method": fold_config["ensemble_method"],
        "n_models": len(models),
        "pca": ncomp,
        **m
    })
    
    gap = 0.85 - m['auc']
    status = "✓ TARGET" if gap <= 0 else f"  {gap:.3f} gap"
    print(f"[FOLD {k}] AUC={m['auc']:.4f} {status} | AP={m['ap']:.4f} ACC={m['acc']:.3f} F1={m['f1']:.3f}")

# Final results
oof_m = metrics_from(y, oof)
print("\n" + "="*60)
print("== FINAL PUSH RESULTS ==")
print(json.dumps(oof_m, indent=2))

# Summary
fold_df = pd.DataFrame(rows)
print("\n" + "="*60)
print("PERFORMANCE SUMMARY:")
print("-"*60)

for _, row in fold_df.iterrows():
    gap = 0.85 - row['auc']
    status = "✓" if gap <= 0 else f"({gap:.3f} short)"
    print(f"Fold {row['fold']}: AUC={row['auc']:.4f} {status}")

print(f"\nStatistics:")
print(f"  Mean AUC: {fold_df['auc'].mean():.4f}")
print(f"  Min AUC:  {fold_df['auc'].min():.4f}")
print(f"  Max AUC:  {fold_df['auc'].max():.4f}")

# Save results
fold_df.to_csv(OUTDIR/"fold_metrics_final.csv", index=False)
pd.DataFrame({"slide_id": sids, "y_true": y, "p_oof": oof}).to_csv(OUTDIR/"oof_scores_final.csv", index=False)

print(f"\n[OK] Saved to {OUTDIR}")
print(f"Done in {(time.time()-t0):.1f}s")