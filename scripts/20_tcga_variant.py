#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - TCGA Variant Analysis
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# ===== cap BLAS threads before imports =====
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

import sys, json, time, threading, pickle, re, hashlib, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# ========================= CONFIG =========================
@dataclass
class CFG:
    WORKSPACE: Path = Path(os.environ.get("WORKSPACE", "./workspace")))
    F05: Path = WORKSPACE / "features" / "scale0p5"
    F20: Path = WORKSPACE / "features" / "scale2p0"
    OUT: Path = WORKSPACE / "results" / "ablations_complete"
    DATASET: str = "tcga"
    SEED: int = 42
    CV_JOBS: int = max(2, (os.cpu_count() or 8) - 1)
    HEARTBEAT_SEC: int = 45
    VEC_RECIPE: str = "cls_both_concat_mean"  # reuse from classifier run

cfg = CFG()
LOG = cfg.OUT / "logs" / f"check_scale_vs_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg, end="\n"):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, end=end, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(line+"\n")

# ========================= UTILS ==========================
def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def normalize_proba(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P)
    if P.ndim == 1: P = np.stack([1.0-P, P], axis=1)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    s = P.sum(axis=1, keepdims=True)
    bad = (s.reshape(-1) <= 0)
    if np.any(bad):
        P[bad,:] = 1.0 / max(1, P.shape[1]); s = P.sum(axis=1, keepdims=True)
    return P / np.clip(s, 1e-12, None)

def read_json(p: Path): 
    with open(p, "r", encoding="utf-8") as f: 
        return json.load(f)

# ===================== MANIFEST/SPLITS ====================
def infer_patient(sid: str) -> str:
    s = str(sid)
    if s.startswith("TCGA-"):
        m = re.match(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s)
        if m: return m.group(1)
    t = re.split(r"[-_\.]", s)
    if len(t)>=3: return "-".join(t[:3])
    if len(t)>=2: return "-".join(t[:2])
    return s

def load_manifest_base() -> pd.DataFrame:
    mp = cfg.WORKSPACE / "manifests" / "manifest_tcga.csv"
    if not mp.exists(): raise FileNotFoundError(mp)
    df = pd.read_csv(mp)
    if "slide_id" not in df or "cancer_code" not in df:
        raise ValueError("Manifest needs slide_id & cancer_code.")
    df["slide_id"] = df["slide_id"].astype(str)
    df["cancer_code"] = df["cancer_code"].astype(str)
    if "case_id" in df: df["group_id"] = df["case_id"].astype(str)
    elif "patient_id" in df: df["group_id"] = df["patient_id"].astype(str)
    else:
        log("[WARN] 'case_id/patient_id' missing — inferring patient from slide_id.")
        df["group_id"] = df["slide_id"].map(infer_patient)
    df["has_both"] = df["slide_id"].map(lambda s: (cfg.F05 / f"{s}.npy").exists() and (cfg.F20 / f"{s}.npy").exists())
    base = df[df["has_both"]].reset_index(drop=True)
    log(f"Manifest loaded: {len(df)} rows | base (both scales): {len(base)}")
    return base

def load_splits(mb: pd.DataFrame):
    sp = cfg.OUT / cfg.DATASET / "splits.json"
    if not sp.exists(): raise FileNotFoundError(f"Missing splits.json → {sp}")
    s = read_json(sp)
    # integrity (patient-level disjoint)
    gids = mb["group_id"].astype(str).values
    for r, rep in enumerate(s, 1):
        for f, (tr,va) in enumerate(rep, 1):
            assert set(gids[tr]).isdisjoint(set(gids[va])), f"Patient overlap r{r} f{f}"
    log(f"Using existing splits: {sp}")
    return s

# ====================== VECTOR CACHE (reuse) ======================
class VectorCache:
    def __init__(self):
        d = cfg.OUT / cfg.DATASET / "vec_cache"
        d.mkdir(parents=True, exist_ok=True)
        self.dir = d
    def _safe(self, s): return re.sub(r"[^A-Za-z0-9_.-]","_", s)
    def path(self, sid, recipe):
        h = hashlib.md5(recipe.encode()).hexdigest()[:8]
        return self.dir / f"{self._safe(sid)}__{self._safe(recipe)}__{h}.npy"
    def get(self, sid, recipe):
        p = self.path(sid, recipe)
        if not p.exists(): return None
        return sanitize(np.load(p))

def load_vectors_from_cache(mb: pd.DataFrame, recipe: str):
    vcache = VectorCache()
    X=[]; y=[]
    missing=0
    for row in mb.itertuples(index=False):
        sid=row.slide_id; v = vcache.get(sid, recipe)
        if v is None:
            missing+=1
        else:
            X.append(v); y.append(row.cancer_code)
    if missing>0:
        raise RuntimeError(f"{missing} vectors missing for recipe {recipe}. "
                           f"Run the classifier script first to populate cache.")
    X = np.vstack(X); y = np.array(y, dtype=str)
    log(f"Loaded vectors from cache: {X.shape[0]} samples, dim={X.shape[1]}")
    return X, y

# ====================== EVAL (same as classifier) ======================
def run_fold(X, y_enc, n_cls, tr, va):
    sc = StandardScaler()
    Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
    clf = LogisticRegression(solver="sag", max_iter=500, tol=1e-3,
                             class_weight="balanced", multi_class="auto",
                             random_state=cfg.SEED, n_jobs=1)
    clf.fit(Xtr, y_enc[tr])
    y_pred = clf.predict(Xva)
    proba = normalize_proba(getattr(clf, "predict_proba")(Xva))
    acc  = accuracy_score(y_enc[va], y_pred)
    bacc = balanced_accuracy_score(y_enc[va], y_pred)
    f1m  = f1_score(y_enc[va], y_pred, average="macro")
    auc  = roc_auc_score(y_enc[va], proba, multi_class="ovr", average="macro") if n_cls>2 else roc_auc_score(y_enc[va], proba[:,1])
    return dict(acc=float(acc), bacc=float(bacc), f1m=float(f1m), auc=float(auc))

def eval_with_splits(X, y, splits):
    le = LabelEncoder(); y_enc = le.fit_transform(y); n_cls = len(np.unique(y_enc))
    allm = defaultdict(list)
    idx=0
    for r, rep in enumerate(splits, 1):
        for f, (tr,va) in enumerate(rep, 1):
            idx+=1
            m = run_fold(X, y_enc, n_cls, np.array(tr), np.array(va))
            allm["accuracy"].append(m["acc"])
            allm["balanced_accuracy"].append(m["bacc"])
            allm["f1_macro"].append(m["f1m"])
            allm["auc"].append(m["auc"])
            log(f"  → Fold r{r}/{len(splits)} f{f}/{len(rep)} [{idx}/{len(splits)*len(rep)}] "
                f"| acc={m['acc']:.4f}, bacc={m['bacc']:.4f}, f1M={m['f1m']:.4f}, auc={m['auc']:.4f}")
    out={}
    for k,v in allm.items():
        arr=np.asarray(v, dtype=float)
        out[k]=dict(mean=float(arr.mean()), std=float(arr.std()),
                    ci_lower=float(np.percentile(arr,2.5)),
                    ci_upper=float(np.percentile(arr,97.5)))
    return out

def write_csv(metrics):
    rows=[]
    for metric, mv in metrics.items():
        rows.append(dict(dataset=cfg.DATASET, ablation="scale_check", variant="both_scales",
                         metric=metric, mean=mv["mean"], std=mv["std"],
                         ci_lower=mv["ci_lower"], ci_upper=mv["ci_upper"]))
    df = pd.DataFrame(rows)
    out = cfg.OUT / cfg.DATASET / "scale_ablation_check.csv"
    df.to_csv(out, index=False)
    log(f"Saved → {out}")
    return out

# ============================== MAIN ==============================
def main():
    log("="*86)
    log("OPENSLIDEFM — SCALE/CLASSIFIER CONSISTENCY CHECK (both_scales → mean → concat → logistic)")
    log(f"Workspace: {cfg.WORKSPACE}")
    log(f"Results:   {cfg.OUT}")
    log("="*86)

    # 1) manifest & splits
    mb = load_manifest_base()
    splits = load_splits(mb)

    # 2) load the VECTORS that classifier script just built
    X, y = load_vectors_from_cache(mb, cfg.VEC_RECIPE)

    # 3) evaluate identically to classifier
    metrics = eval_with_splits(X, y, splits)
    csv_new = write_csv(metrics)

    # 4) side-by-side vs classifier CSV
    clf_csv = cfg.OUT / cfg.DATASET / "classifier_ablation.csv"
    if clf_csv.exists():
        clf = pd.read_csv(clf_csv)
        def row(metric): 
            s = clf[(clf["ablation"]=="classifier") & (clf["variant"]=="logistic") & (clf["metric"]==metric)]
            return None if s.empty else (float(s["mean"]), float(s["std"]))
        log("\n====== SIDE-BY-SIDE (mean ± sd) ======")
        for m in ["accuracy","auc","balanced_accuracy","f1_macro"]:
            a = metrics[m]["mean"]; b = metrics[m]["std"]
            c = row(m)
            if c is None:
                log(f"{m:<18}: scale_check {a:.4f} ± {b:.4f} | classifier MISSING")
            else:
                log(f"{m:<18}: scale_check {a:.4f} ± {b:.4f} | classifier {c[0]:.4f} ± {c[1]:.4f}")
        log("======================================\n")
    else:
        log("Classifier CSV not found for side-by-side; only scale_check printed.")

    log("DONE ✓")

if __name__ == "__main__":
    main()
