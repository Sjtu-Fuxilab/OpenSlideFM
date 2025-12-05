#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - UNI Benchmarking & Comparison
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

#Complete UNI2-h Benchmarking & Fair Comparison

import os
import sys
import json
import math
import time
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import h5py

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_curve, auc, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold

import matplotlib
import matplotlib.pyplot as plt

# Repro / quiet
warnings.filterwarnings("ignore")


# ======================
# CONFIG
# ======================
@dataclass
class Config:
    BASE_DIR: Path = Path(os.environ.get("UNI_FEATURES", "./features/uni")))
    OUTPUT_DIR: Path = Path(os.environ.get("UNI_RESULTS", "./results/uni")))
    TCGA_TYPES: Tuple[str, ...] = (
        'TCGA-ACC', 'TCGA-BRCA_IDC', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-GBM',
        'TCGA-HNSC', 'TCGA-KIRC', 'TCGA-LUAD', 'TCGA-SKCM', 'TCGA-UCEC'
    )
    AGGREGATION: str = "mean"     # 'mean' | 'max' | 'attention'
    N_FOLDS: int = 5
    RANDOM_STATE: int = 42
    GROUP_BY_SITE: bool = True     # use TSS grouping if available for SGKF
    MODEL_TYPE: str = "logistic"   # 'logistic' | 'rf'
    N_BOOTSTRAP: int = 2000        # for 95% CI on macro metrics


CFG = Config()
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# Utils
# ======================
def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def save_json(obj: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ======================
# H5 Loading & Aggregation
# ======================
def _try_get(f, keys: Tuple[str, ...]) -> np.ndarray:
    for k in keys:
        if k in f:
            return f[k][:]
    raise KeyError(f"None of keys {keys} found in file")

def load_slide_features(h5_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flexible UNI2-h H5 loader. Returns (features, coords or None)
    features: (num_patches, D)
    """
    with h5py.File(h5_path, 'r') as f:
        features = _try_get(f, ('features', 'feats', 'patch_features', 'x'))
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        elif features.ndim == 1:
            features = features.reshape(1, -1)
        coords = None
        for ck in ('coords', 'xy', 'positions'):
            if ck in f:
                coords = f[ck][:]
                if coords.ndim == 3 and coords.shape[0] == 1:
                    coords = coords.squeeze(0)
                break
    if features.size == 0:
        raise ValueError(f"Empty features in {h5_path.name}")
    return features, coords

def aggregate_slide_features(features: np.ndarray, method: str = 'mean') -> np.ndarray:
    if method == 'mean':
        return features.mean(axis=0)
    elif method == 'max':
        return features.max(axis=0)
    elif method == 'attention':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        weights = norms / (norms.sum() + 1e-8)
        return (features * weights).sum(axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {method}")

def load_cancer_type_features(cancer_dir: Path, aggregation: str = 'mean') -> pd.DataFrame:
    h5_files = sorted(list(cancer_dir.glob("*.h5")))
    print(f"  Loading {len(h5_files)} slides from: {cancer_dir.name}")
    rows = []
    for fp in tqdm(h5_files, desc="   reading h5", leave=False):
        try:
            feats, _ = load_slide_features(fp)
            slide_vec = aggregate_slide_features(feats, aggregation)
            rows.append({
                "slide_id": fp.stem,
                "num_patches": feats.shape[0],
                "features": slide_vec
            })
        except Exception as e:
            print(f"   ⚠ {fp.name}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    feats_arr = np.vstack(df["features"].values)
    feat_cols = [f"f{i:04d}" for i in range(feats_arr.shape[1])]
    feats_df = pd.DataFrame(feats_arr, columns=feat_cols)
    out = pd.concat([df[["slide_id", "num_patches"]].reset_index(drop=True), feats_df], axis=1)
    return out

def load_all_tcga_features(base_dir: Path, tcga_types: Tuple[str, ...], aggregation: str) -> pd.DataFrame:
    print("\n" + "="*70)
    print("LOADING UNI2-h FEATURES FOR TCGA")
    print("="*70)
    all_dfs, loaded = [], []
    for ct in tcga_types:
        p = base_dir / ct
        if not p.exists():
            print(f"❌ {ct}: missing → skip")
            continue
        print(f"\n📊 {ct}")
        df = load_cancer_type_features(p, aggregation)
        if df.empty:
            print("  ⚠ No slides → skip")
            continue
        df["cancer_type"] = ct
        loaded.append(ct)
        all_dfs.append(df)
        print(f"  ✓ {len(df)} slides")
    if not all_dfs:
        raise RuntimeError("No TCGA cohorts loaded")
    all_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(all_df)} slides from {len(loaded)} cohorts")
    print(f"Loaded: {', '.join(loaded)}")
    print(f"{'='*70}\n")
    return all_df

def load_panda_features(base_dir: Path, aggregation: str) -> pd.DataFrame:
    p = base_dir / "panda"
    if not p.exists():
        print("❌ PANDA dir not found; skipping PANDA")
        return pd.DataFrame()
    print("\n" + "="*70)
    print("LOADING PANDA FEATURES")
    print("="*70)
    df = load_cancer_type_features(p, aggregation)
    if df.empty:
        print("⚠ PANDA empty")
        return df
    df["dataset"] = "PANDA"
    print(f"✓ {len(df)} PANDA slides\n")
    return df


# ======================
# Grouping (TCGA site)
# ======================
def extract_tcga_tss(slide_id: str) -> str:
    # TCGA-XX-YYYY-... → 'XX' is tissue source site
    try:
        parts = slide_id.split('-')
        return parts[1] if len(parts) > 1 else 'NA'
    except Exception:
        return 'NA'


# ======================
# Modeling
# ======================
def get_model(model_type: str = "logistic"):
    if model_type == "logistic":
        return LogisticRegression(
            max_iter=2000, random_state=CFG.RANDOM_STATE,
            class_weight='balanced', multi_class='multinomial', solver='lbfgs'
        )
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=400, random_state=CFG.RANDOM_STATE,
            class_weight='balanced', n_jobs=-1
        )
    else:
        raise ValueError("model_type must be 'logistic' or 'rf'")


@dataclass
class CVArtifacts:
    fold_metrics: pd.DataFrame
    per_class_metrics: pd.DataFrame
    y_true_all: np.ndarray
    y_prob_all: np.ndarray
    y_pred_all: np.ndarray
    labels: List[str]
    conf_mat: np.ndarray


def stratified_group_kfold(n_splits=5, shuffle=True, random_state=42):
    """
    Returns a splitter; prefers StratifiedGroupKFold if available,
    else falls back to GroupKFold or StratifiedKFold (no groups).
    """
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), "SGKF"
    except Exception:
        try:
            from sklearn.model_selection import GroupKFold
            return GroupKFold(n_splits=n_splits), "GK"
        except Exception:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), "SKF"


def cross_validate_multiclass(df: pd.DataFrame, label_col="cancer_type") -> CVArtifacts:
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values
    labels = df[label_col].values
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = list(le.classes_)

    # groups (by TSS) if enabled
    groups = np.array([extract_tcga_tss(s) for s in df["slide_id"].values]) if CFG.GROUP_BY_SITE else None

    splitter, mode = stratified_group_kfold(CFG.N_FOLDS, True, CFG.RANDOM_STATE)
    print(f"CV splitter: {mode} | folds={CFG.N_FOLDS} | group_by_site={CFG.GROUP_BY_SITE}")

    scaler = StandardScaler()
    fold_rows = []
    per_class_rows = []

    # Store OOF predictions
    y_true_all = np.zeros(X.shape[0], dtype=int) - 1
    y_pred_all = np.zeros(X.shape[0], dtype=int) - 1
    y_prob_all = np.zeros((X.shape[0], len(classes)), dtype=float)

    split_iter = splitter.split(X, y, groups=groups) if groups is not None and mode in ("SGKF", "GK") else splitter.split(X, y)
    for fold, (tr, te) in enumerate(split_iter, 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = get_model(CFG.MODEL_TYPE)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)

        # fold metrics
        fold_metrics = {
            "fold": fold,
            "accuracy": accuracy_score(y_te, y_pred),
            "f1_macro": f1_score(y_te, y_pred, average="macro"),
            "f1_weighted": f1_score(y_te, y_pred, average="weighted"),
            "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
        }
        try:
            auc_macro = roc_auc_score(y_te, y_proba, multi_class="ovr", average="macro")
            auc_weighted = roc_auc_score(y_te, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            auc_macro, auc_weighted = np.nan, np.nan
        fold_metrics["auc_macro"] = auc_macro
        fold_metrics["auc_weighted"] = auc_weighted
        fold_rows.append(fold_metrics)

        # per-class AUC OvR + F1
        y_te_bin = label_binarize(y_te, classes=np.arange(len(classes)))
        # per-class AUC
        per_class_auc = []
        for k in range(len(classes)):
            try:
                auc_k = roc_auc_score(y_te_bin[:, k], y_proba[:, k])
            except Exception:
                auc_k = np.nan
            per_class_auc.append(auc_k)
            f1_k = f1_score((y_te == k).astype(int), (y_pred == k).astype(int), zero_division=0)
            per_class_rows.append({"fold": fold, "class": classes[k], "auc_ovr": auc_k, "f1": f1_k})

        # store OOF
        y_true_all[te] = y_te
        y_pred_all[te] = y_pred
        y_prob_all[te, :] = y_proba

        print(f"  Fold {fold}/{CFG.N_FOLDS} | acc={fold_metrics['accuracy']:.4f} | f1m={fold_metrics['f1_macro']:.4f} | aucm={fold_metrics['auc_macro']:.4f}")

    fold_df = pd.DataFrame(fold_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    # sanity
    assert (y_true_all >= 0).all(), "OOF y_true not fully assigned."
    assert (y_pred_all >= 0).all(), "OOF y_pred not fully assigned."

    # confusion matrix on OOF
    cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(classes)))

    return CVArtifacts(
        fold_metrics=fold_df,
        per_class_metrics=per_class_df,
        y_true_all=y_true_all,
        y_prob_all=y_prob_all,
        y_pred_all=y_pred_all,
        labels=classes,
        conf_mat=cm
    )


# ======================
# Statistics: Bootstrap CIs & ECE
# ======================
def bootstrap_ci_macro(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap 95% CI for macro metrics from OOF predictions.
    """
    rng = np.random.RandomState(seed)
    n = y_true.shape[0]
    classes = np.unique(y_true)
    metrics = {"accuracy": [], "f1_macro": [], "auc_macro": []}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)  # sample with replacement
        yt = y_true[idx]
        yp = y_pred[idx]
        ypb = y_prob[idx]

        metrics["accuracy"].append(accuracy_score(yt, yp))
        metrics["f1_macro"].append(f1_score(yt, yp, average="macro"))
        try:
            metrics["auc_macro"].append(roc_auc_score(yt, ypb, multi_class="ovr", average="macro"))
        except Exception:
            metrics["auc_macro"].append(np.nan)

    out = {}
    for k, vals in metrics.items():
        arr = np.array(vals, dtype=float)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[k] = (float(lo), float(hi))
    return out

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, pd.DataFrame]:
    """
    Multiclass ECE using max-probability approach.
    """
    max_conf = y_prob.max(axis=1)
    y_pred = y_prob.argmax(axis=1)
    correct = (y_pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = (max_conf >= lo) & (max_conf < hi) if i < n_bins - 1 else (max_conf >= lo) & (max_conf <= hi)
        if sel.sum() == 0:
            rows.append({"bin": i+1, "conf": 0.0, "acc": 0.0, "count": 0})
            continue
        bin_conf = max_conf[sel].mean()
        bin_acc = correct[sel].mean()
        rows.append({"bin": i+1, "conf": float(bin_conf), "acc": float(bin_acc), "count": int(sel.sum())})
        ece += (sel.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece), pd.DataFrame(rows)


# ======================
# Plots (matplotlib only)
# ======================
def save_boxplots(cv_df: pd.DataFrame, out: Path, title_prefix="UNI"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("accuracy", "Accuracy"), ("f1_macro", "F1 (Macro)"), ("auc_macro", "AUC (Macro)")]
    for ax, (key, name) in zip(axes, metrics):
        ax.boxplot(cv_df[key].values, showmeans=True)
        ax.set_title(f"{title_prefix}: {name}")
        ax.set_xticks([1]); ax.set_xticklabels([name])
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(name)
    plt.tight_layout()
    png = out / f"{title_prefix.lower()}_cv_boxplots.png"
    pdf = out / f"{title_prefix.lower()}_cv_boxplots.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {png}")

def save_per_class_auc(per_class_df: pd.DataFrame, out: Path, title_prefix="UNI"):
    # mean across folds
    g = per_class_df.groupby("class", as_index=False)["auc_ovr"].mean().sort_values("auc_ovr", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(g["class"], g["auc_ovr"])
    ax.set_xlabel("AUC (OvR, mean across folds)")
    ax.set_title(f"{title_prefix}: Per-class AUC (OvR)")
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    png = out / f"{title_prefix.lower()}_per_class_auc.png"
    pdf = out / f"{title_prefix.lower()}_per_class_auc.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {png}")

def save_confusion(cm: np.ndarray, labels: List[str], out: Path, title_prefix="UNI"):
    # normalize by true class
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cmn, aspect='auto', interpolation='nearest')
    ax.set_title(f"{title_prefix}: Normalized Confusion Matrix (OOF)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    png = out / f"{title_prefix.lower()}_confusion_matrix.png"
    pdf = out / f"{title_prefix.lower()}_confusion_matrix.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {png}")

def save_reliability_plot(y_true: np.ndarray, y_prob: np.ndarray, out: Path, title_prefix="UNI"):
    ece, bins_df = expected_calibration_error(y_true, y_prob, n_bins=15)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0,1],[0,1], linestyle='--')
    ax.plot(bins_df["conf"], bins_df["acc"], marker='o')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{title_prefix}: Reliability Diagram (ECE={ece:.3f})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png = out / f"{title_prefix.lower()}_reliability.png"
    pdf = out / f"{title_prefix.lower()}_reliability.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {png}")

def save_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], out: Path, title_prefix="UNI"):
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    cols = 2
    rows = math.ceil(len(class_names)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4.5*rows))
    axes = np.array(axes).reshape(rows, cols)
    for k, cname in enumerate(class_names):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        pr, rc, _ = precision_recall_curve(y_bin[:, k], y_prob[:, k])
        aupr = auc(rc, pr)
        ax.plot(rc, pr)
        ax.set_title(f"{cname} (AUPR={aupr:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.grid(True, alpha=0.3)
    # hide empty
    for idx in range(len(class_names), rows*cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    plt.tight_layout()
    png = out / f"{title_prefix.lower()}_pr_curves.png"
    pdf = out / f"{title_prefix.lower()}_pr_curves.pdf"
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(pdf, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {png}")


# ======================
# FAIR COMPARISON UNI vs YOUR MODEL
# ======================
def build_common_matrices(uni_df: pd.DataFrame, your_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        X_uni, X_your, y (encoded), groups, class_names
        (slides restricted to intersection by slide_id and cancer_type)
    """
    fcols_u = [c for c in uni_df.columns if c.startswith("f")]
    fcols_y = [c for c in your_df.columns if c.startswith("f")]
    assert fcols_u, "UNI df missing feature columns f*"
    assert fcols_y, "YOUR df missing feature columns f*"

    u = uni_df[["slide_id", "cancer_type"] + fcols_u].copy()
    y = your_df[["slide_id", "cancer_type"] + fcols_y].copy()
    merged = u.merge(y, on=["slide_id", "cancer_type"], suffixes=("_uni", "_your"))
    if merged.empty:
        raise RuntimeError("No overlapping slides with matching cancer_type between UNI and YOUR model.")

    X_uni = merged[[c for c in merged.columns if c.startswith("f") and c.endswith("_uni")]].values
    X_your = merged[[c for c in merged.columns if c.startswith("f") and c.endswith("_your")]].values
    y_lbl = merged["cancer_type"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_lbl)
    class_names = list(le.classes_)

    # groups by TSS
    groups = np.array([extract_tcga_tss(s) for s in merged["slide_id"].values])
    return X_uni, X_your, y_enc, groups, class_names

def fair_compare_same_folds(X_uni: np.ndarray, X_your: np.ndarray, y: np.ndarray, groups: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    splitter, mode = stratified_group_kfold(CFG.N_FOLDS, True, CFG.RANDOM_STATE)
    print(f"Fair compare splitter: {mode}")

    split_iter = splitter.split(X_uni, y, groups=groups) if groups is not None and mode in ("SGKF", "GK") else splitter.split(X_uni, y)

    rows_uni, rows_your = [], []
    for fold, (tr, te) in enumerate(split_iter, 1):
        # UNI
        sc_u = StandardScaler()
        Xtr_u, Xte_u = sc_u.fit_transform(X_uni[tr]), sc_u.transform(X_uni[te])
        clf_u = get_model(CFG.MODEL_TYPE)
        clf_u.fit(Xtr_u, y[tr])
        ypr_u = clf_u.predict(Xte_u); ypb_u = clf_u.predict_proba(Xte_u)

        # YOUR
        sc_y = StandardScaler()
        Xtr_y, Xte_y = sc_y.fit_transform(X_your[tr]), sc_y.transform(X_your[te])
        clf_y = get_model(CFG.MODEL_TYPE)
        clf_y.fit(Xtr_y, y[tr])
        ypr_y = clf_y.predict(Xte_y); ypb_y = clf_y.predict_proba(Xte_y)

        def fold_stats(ytrue, ypred, yprob):
            d = {
                "accuracy": accuracy_score(ytrue, ypred),
                "f1_macro": f1_score(ytrue, ypred, average="macro"),
            }
            try:
                d["auc_macro"] = roc_auc_score(ytrue, yprob, multi_class="ovr", average="macro")
            except Exception:
                d["auc_macro"] = np.nan
            return d

        ru = fold_stats(y[te], ypr_u, ypb_u); ru["fold"] = fold
        ry = fold_stats(y[te], ypr_y, ypb_y); ry["fold"] = fold
        rows_uni.append(ru); rows_your.append(ry)

        print(f"  Fold {fold}: UNI acc={ru['accuracy']:.4f} | YOUR acc={ry['accuracy']:.4f}")

    uni_cv = pd.DataFrame(rows_uni); your_cv = pd.DataFrame(rows_your)
    cmp = pd.DataFrame({
        "Metric": ["Accuracy", "F1 (macro)", "AUC (macro)"],
        "UNI (mean±std)": [
            f"{uni_cv['accuracy'].mean():.4f}±{uni_cv['accuracy'].std():.4f}",
            f"{uni_cv['f1_macro'].mean():.4f}±{uni_cv['f1_macro'].std():.4f}",
            f"{uni_cv['auc_macro'].mean():.4f}±{uni_cv['auc_macro'].std():.4f}"
        ],
        "Yours (mean±std)": [
            f"{your_cv['accuracy'].mean():.4f}±{your_cv['accuracy'].std():.4f}",
            f"{your_cv['f1_macro'].mean():.4f}±{your_cv['f1_macro'].std():.4f}",
            f"{your_cv['auc_macro'].mean():.4f}±{your_cv['auc_macro'].std():.4f}"
        ],
        "Δ (UNI − Yours)": [
            f"{(uni_cv['accuracy'] - your_cv['accuracy']).mean():+.4f}",
            f"{(uni_cv['f1_macro'] - your_cv['f1_macro']).mean():+.4f}",
            f"{(uni_cv['auc_macro'] - your_cv['auc_macro']).mean():+.4f}",
        ]
    })
    return cmp, uni_cv, your_cv


# ======================
# IO helpers for your feature table
# ======================
def save_df(df: pd.DataFrame, path: Path):
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)

def load_your_features_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    need = {"slide_id", "cancer_type"}
    assert need.issubset(set(df.columns)), f"your features must include {need}"
    fcols = [c for c in df.columns if c.startswith("f")]
    assert fcols, "No feature columns starting with 'f'"
    return df


# ======================
# MAIN
# ======================
def main():
    set_global_seed(CFG.RANDOM_STATE)
    start = time.time()
    print(f"[{now()}] START")

    # --- Load UNI features
    uni_df = load_all_tcga_features(CFG.BASE_DIR, CFG.TCGA_TYPES, CFG.AGGREGATION)
    uni_parquet = CFG.OUTPUT_DIR / "uni_features_all_tcga.parquet"
    uni_df.to_parquet(uni_parquet, index=False)
    print(f"✓ Saved {uni_parquet}")

    # --- Cross-validate UNI
    print("\n" + "="*70)
    print("CROSS-VALIDATION: UNI")
    print("="*70)
    uni_art = cross_validate_multiclass(uni_df, label_col="cancer_type")

    # Save artifacts
    save_df(uni_art.fold_metrics, CFG.OUTPUT_DIR / "uni_cv_folds.csv")
    save_df(uni_art.per_class_metrics, CFG.OUTPUT_DIR / "uni_cv_per_class.csv")

    # Bootstrap CIs on OOF
    uni_ci = bootstrap_ci_macro(
        uni_art.y_true_all, uni_art.y_prob_all, uni_art.y_pred_all,
        n_boot=CFG.N_BOOTSTRAP, seed=CFG.RANDOM_STATE
    )

    # Summary
    uni_summary = {
        "n_slides": int(len(uni_df)),
        "n_classes": int(len(uni_art.labels)),
        "feature_dim": int(sum(c.startswith("f") for c in uni_df.columns)),
        "aggregation": CFG.AGGREGATION,
        "model": CFG.MODEL_TYPE,
        "cv_folds": CFG.N_FOLDS,
        "group_by_site": CFG.GROUP_BY_SITE,
        "metrics_mean": {
            "accuracy": float(uni_art.fold_metrics["accuracy"].mean()),
            "f1_macro": float(uni_art.fold_metrics["f1_macro"].mean()),
            "auc_macro": float(uni_art.fold_metrics["auc_macro"].mean())
        },
        "metrics_std": {
            "accuracy": float(uni_art.fold_metrics["accuracy"].std()),
            "f1_macro": float(uni_art.fold_metrics["f1_macro"].std()),
            "auc_macro": float(uni_art.fold_metrics["auc_macro"].std())
        },
        "metrics_ci95": uni_ci,
        "ece": float(expected_calibration_error(uni_art.y_true_all, uni_art.y_prob_all)[0]),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": __import__("sklearn").__version__,
            "matplotlib": matplotlib.__version__
        },
        "seed": CFG.RANDOM_STATE
    }
    save_json(uni_summary, CFG.OUTPUT_DIR / "uni_summary.json")
    print("✓ Saved uni_summary.json")

    # --- Plots for UNI
    save_boxplots(uni_art.fold_metrics, CFG.OUTPUT_DIR, "UNI")
    save_per_class_auc(uni_art.per_class_metrics, CFG.OUTPUT_DIR, "UNI")
    save_confusion(uni_art.conf_mat, uni_art.labels, CFG.OUTPUT_DIR, "UNI")
    save_reliability_plot(uni_art.y_true_all, uni_art.y_prob_all, CFG.OUTPUT_DIR, "UNI")
    save_pr_curves(uni_art.y_true_all, uni_art.y_prob_all, uni_art.labels, CFG.OUTPUT_DIR, "UNI")

    # --- FAIR COMPARISON (if your features available)
    # Put your features parquet/csv path here (must include slide_id, cancer_type, and f0000... columns)
    YOUR_FEATURES_PATH = CFG.OUTPUT_DIR / "your_features_all_tcga.parquet"  # <-- set this to your actual file
    if YOUR_FEATURES_PATH.exists():
        print("\n" + "="*70)
        print("FAIR COMPARISON: UNI vs YOUR MODEL (same slides, same folds)")
        print("="*70)
        your_df = load_your_features_table(YOUR_FEATURES_PATH)

        X_u, X_y, y_enc, groups, class_names = build_common_matrices(uni_df, your_df)
        cmp_table, uni_cv_fair, your_cv_fair = fair_compare_same_folds(X_u, X_y, y_enc, groups, class_names)

        save_df(uni_cv_fair, CFG.OUTPUT_DIR / "uni_cv_fair.csv")
        save_df(your_cv_fair, CFG.OUTPUT_DIR / "your_cv_fair.csv")
        save_df(cmp_table, CFG.OUTPUT_DIR / "comparison_fair.csv")
        print("\n" + cmp_table.to_string(index=False))

        # Simple fairness plot (boxplot pairs)
        def plot_pair_box(uni_series, your_series, metric_name, outstem):
            fig, ax = plt.subplots(figsize=(5,5))
            ax.boxplot([uni_series.values, your_series.values], showmeans=True)
            ax.set_xticks([1,2]); ax.set_xticklabels(["UNI","Yours"])
            ax.set_ylabel(metric_name); ax.grid(True, alpha=0.3)
            ax.set_title(f"Fair CV: {metric_name}")
            png = CFG.OUTPUT_DIR / f"{outstem}.png"
            pdf = CFG.OUTPUT_DIR / f"{outstem}.pdf"
            plt.tight_layout()
            plt.savefig(png, dpi=300, bbox_inches='tight')
            plt.savefig(pdf, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved {png}")

        plot_pair_box(uni_cv_fair["accuracy"], your_cv_fair["accuracy"], "Accuracy", "fair_accuracy")
        plot_pair_box(uni_cv_fair["f1_macro"], your_cv_fair["f1_macro"], "F1 (Macro)", "fair_f1_macro")
        plot_pair_box(uni_cv_fair["auc_macro"], your_cv_fair["auc_macro"], "AUC (Macro)", "fair_auc_macro")
    else:
        print(f"\n⚠ Skipping fair comparison: file not found → {YOUR_FEATURES_PATH}")

    elapsed = time.time() - start
    print(f"\n[{now()}] DONE in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
