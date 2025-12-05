#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - TCGA Classification
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# ================== cap BLAS threads BEFORE any scientific imports ==================
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("JOBLIB_START_METHOD", "spawn")
# ====================================================================================

import sys, json, time, threading, pickle, hashlib, warnings, re, glob, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch

from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

# ================================== CONFIG ==================================
@dataclass
class CFG:
    WORKSPACE: Path = Path(r"D:\个人文件夹\Sanwal\OpenSlide")
    F05: Path = WORKSPACE / "features" / "scale0p5"
    F20: Path = WORKSPACE / "features" / "scale2p0"
    OUT: Path = WORKSPACE / "results" / "ablations_complete"
    DATASET: str = "tcga"

    N_FOLDS: int = 5
    N_REPEATS: int = 3
    SEED: int = 42

    # speed knobs (keep 0 to match prior results; set to 256 for faster runs)
    PCA_DIM: int = 0
    CV_JOBS: int = max(2, (os.cpu_count() or 8) - 1)
    HEARTBEAT_SEC: int = 45

    # behavior
    USE_EXISTING_SPLITS_IF_FOUND: bool = True   # match prior ablations
    REBUILD_IF_MISSING: bool = True
    CLEAR_OLD_CLASSIFIER_ARTIFACTS: bool = True # nuke stale classifier CSV/ckpts
    OVERWRITE_CSV: bool = True                  # always write fresh classifier CSV

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = CFG()
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.SEED)

# ================================== LOGGING =================================
LOG_DIR = cfg.OUT / "logs"; LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"run_classifier_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log(msg: str, end: str="\n"):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(line + "\n")

class Heartbeat:
    def __init__(self, label="RUN", sec=60):
        self.label=label; self.sec=sec; self._stop=threading.Event()
        self._t=threading.Thread(target=self._loop, daemon=True)
    def _loop(self):
        t=0
        while not self._stop.is_set():
            time.sleep(self.sec); t+=self.sec
            log(f"♥ HEARTBEAT[{self.label}] alive ~{t//60} min …")
    def __enter__(self): self._t.start(); return self
    def __exit__(self, *a): self._stop.set(); self._t.join(timeout=2)

# ================================ UTILITIES ================================
def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def normalize_proba(P: np.ndarray) -> np.ndarray:
    """Robust n_samples×n_classes probabilities; fixes NaN/row-sum issues."""
    P = np.asarray(P)
    if P.ndim == 1:
        P = np.stack([1.0 - P, P], axis=1)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    s = P.sum(axis=1, keepdims=True)
    bad = (s.reshape(-1) <= 0)
    if np.any(bad):
        P[bad, :] = 1.0 / max(1, P.shape[1])
        s = P.sum(axis=1, keepdims=True)
    s = np.clip(s, 1e-12, None)
    return P / s

def save_json(o, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: json.dump(o, f, indent=2)

def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

# ============================ FEATURE/VECTOR CACHE =========================
class FeatureCache:
    def __init__(self, cache: Optional[Path]=None):
        self.dir = cache or (cfg.WORKSPACE / "cache"); self.dir.mkdir(parents=True, exist_ok=True)
        self.mem = {}
    def _key(self, sid, scale): return f"{sid}_{scale}"
    def load(self, sid: str, scale: str) -> Optional[np.ndarray]:
        key = self._key(sid, scale)
        if key in self.mem: return self.mem[key]
        pkl = self.dir / f"{key}.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f: arr = pickle.load(f)
            self.mem[key] = sanitize(arr); return self.mem[key]
        src = (cfg.F05 if scale=="0.5" else cfg.F20) / f"{sid}.npy"
        if src.exists():
            arr = sanitize(np.load(src))
            with open(pkl, "wb") as f: pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.mem[key] = arr; return arr
        return None

class VectorCache:
    def __init__(self, dataset: str):
        self.dir = cfg.OUT / dataset / "vec_cache"; self.dir.mkdir(parents=True, exist_ok=True)
    def _safe(self, s): return re.sub(r"[^A-Za-z0-9_.-]", "_", s)
    def path(self, sid, recipe):
        h = hashlib.md5(recipe.encode()).hexdigest()[:8]
        return self.dir / f"{self._safe(sid)}__{self._safe(recipe)}__{h}.npy"
    def get(self, sid, recipe):
        p = self.path(sid, recipe)
        if p.exists():
            try: return sanitize(np.load(p))
            except Exception: return None
        return None
    def put(self, sid, recipe, v):
        np.save(self.path(sid, recipe), sanitize(v))

# =============================== MANIFEST/SPLITS ===========================
def infer_patient(sid: str) -> str:
    s = str(sid)
    if s.startswith("TCGA-"):
        m = re.match(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s)
        if m: return m.group(1)
    t = re.split(r"[-_\.]", s)
    if len(t) >= 3: return "-".join(t[:3])
    if len(t) >= 2: return "-".join(t[:2])
    return s

def load_manifest_base() -> pd.DataFrame:
    mp = cfg.WORKSPACE / "manifests" / "manifest_tcga.csv"
    if not mp.exists(): raise FileNotFoundError(mp)
    df = pd.read_csv(mp)
    if "slide_id" not in df or "cancer_code" not in df:
        raise ValueError("Manifest needs slide_id & cancer_code columns.")
    df["slide_id"] = df["slide_id"].astype(str)
    df["cancer_code"] = df["cancer_code"].astype(str)

    if "case_id" in df: df["group_id"] = df["case_id"].astype(str)
    elif "patient_id" in df: df["group_id"] = df["patient_id"].astype(str)
    else:
        log("[WARN] 'case_id/patient_id' missing — inferring patient IDs from slide_id.")
        df["group_id"] = df["slide_id"].map(infer_patient)

    df["has_05"] = df["slide_id"].map(lambda s: (cfg.F05 / f"{s}.npy").exists())
    df["has_20"] = df["slide_id"].map(lambda s: (cfg.F20 / f"{s}.npy").exists())
    base = df[df["has_05"] & df["has_20"]].reset_index(drop=True)

    log(f"Manifest loaded: {len(df)} rows | base cohort with both scales: {len(base)}")
    return base

def make_splits(mb: pd.DataFrame):
    y = mb["cancer_code"].astype(str).values
    g = mb["group_id"].astype(str).values
    all_s = []
    for r in range(cfg.N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED + r)
        rep=[]
        for tr, va in sgkf.split(np.arange(len(mb)), y, g):
            rep.append((tr.tolist(), va.tolist()))
        all_s.append(rep)
    return all_s

def assert_split_integrity(mb: pd.DataFrame, splits):
    gids = mb["group_id"].astype(str).values
    for r, rep in enumerate(splits, 1):
        for f, (tr, va) in enumerate(rep, 1):
            assert set(tr).isdisjoint(set(va)), f"Index overlap r{r} f{f}"
            assert set(gids[tr]).isdisjoint(set(gids[va])), f"Patient overlap r{r} f{f}"

def load_or_create_splits(mb: pd.DataFrame):
    sp = cfg.OUT / cfg.DATASET / "splits.json"
    if cfg.USE_EXISTING_SPLITS_IF_FOUND and sp.exists():
        log(f"Using existing splits: {sp}")
        s = read_json(sp)
    else:
        if not cfg.REBUILD_IF_MISSING and not sp.exists():
            raise FileNotFoundError("splits.json missing and rebuild disabled.")
        log("Creating patient-level shared splits …")
        s = make_splits(mb); save_json(s, sp); log(f"Wrote splits → {sp}")
    assert_split_integrity(mb, s)
    return s

# =============================== VECTORS (both scales, mean→concat) ========
def mean_pool(F: np.ndarray) -> np.ndarray:
    return sanitize(F).mean(axis=0).astype(np.float32)

def build_vectors_both_scales(mb: pd.DataFrame, recipe: str="cls_both_concat_mean"):
    fcache = FeatureCache(); vcache = VectorCache(cfg.DATASET)
    N = len(mb); X = [None]*N; y = mb["cancer_code"].astype(str).values
    start = time.time(); last = start; done = 0

    def work(i, row):
        sid = row["slide_id"]
        v = vcache.get(sid, recipe)
        if v is None:
            a = fcache.load(sid, "0.5"); b = fcache.load(sid, "2.0")
            if a is None or b is None: return i, None
            v = np.concatenate([mean_pool(a), mean_pool(b)]).astype(np.float32)
            vcache.put(sid, recipe, v)
        return i, sanitize(v)

    with ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 8))) as ex:
        futs = [ex.submit(work, i, row) for i, row in enumerate(mb.to_dict(orient="records"))]
        for fu in as_completed(futs):
            i, v = fu.result()
            if v is None: raise RuntimeError(f"vector missing at index {i}")
            X[i] = v; done += 1
            now = time.time()
            if now - last >= 5 or done == N:
                rate = done / max(1e-9, now - start); eta = (N - done) / max(1e-9, rate)
                log(f"    [build:{recipe}] {done}/{N} | {rate:.1f}/s | ETA ~{int(eta//60)}m{int(eta%60)}s")
                last = now
    X = np.vstack(X)
    log(f"  Built cached vectors: {N} samples, dim={X.shape[1]}")
    return X, y

# =============================== EVALUATION ================================
def run_fold(X, y_enc, n_cls, tr, va):
    Xtr, Xva = X[tr], X[va]; ytr, yva = y_enc[tr], y_enc[va]
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xva = sc.transform(Xva)
    if cfg.PCA_DIM and cfg.PCA_DIM > 0:
        pca = PCA(n_components=min(cfg.PCA_DIM, Xtr.shape[1]), random_state=cfg.SEED)
        Xtr = pca.fit_transform(Xtr); Xva = pca.transform(Xva)
    clf = LogisticRegression(solver="sag", max_iter=500, tol=1e-3,
                             class_weight="balanced", multi_class="auto",
                             random_state=cfg.SEED, n_jobs=1)
    t0 = time.time(); clf.fit(Xtr, ytr); fit_s = time.time() - t0
    y_pred = clf.predict(Xva)
    proba = normalize_proba(getattr(clf, "predict_proba")(Xva))
    acc  = accuracy_score(yva, y_pred)
    bacc = balanced_accuracy_score(yva, y_pred)
    f1m  = f1_score(yva, y_pred, average="macro")
    auc  = roc_auc_score(yva, proba, multi_class="ovr", average="macro") if n_cls > 2 else roc_auc_score(yva, proba[:,1])
    return dict(acc=float(acc), bacc=float(bacc), f1m=float(f1m), auc=float(auc), fit_s=float(fit_s))

def eval_classifier(X, y, splits):
    le = LabelEncoder(); y_enc = le.fit_transform(y); n_cls = len(np.unique(y_enc))
    allm = defaultdict(list)
    tasks = []
    for r, rep in enumerate(splits, 1):
        for f, (tr, va) in enumerate(rep, 1):
            tasks.append((r, f, tr, va, len(rep)))

    res = Parallel(n_jobs=cfg.CV_JOBS, prefer="processes")(
        delayed(run_fold)(X, y_enc, n_cls, tr, va) for (_,_,tr,va,_) in tasks
    )

    for i, m in enumerate(res, 1):
        r, f, _, _, nf = tasks[i-1]
        allm["accuracy"].append(m["acc"])
        allm["balanced_accuracy"].append(m["bacc"])
        allm["f1_macro"].append(m["f1m"])
        allm["auc"].append(m["auc"])
        log(f"  → Fitted r{r}/{len(splits)} f{f}/{nf} [{i}/{len(tasks)}] | "
            f"acc={m['acc']:.4f}, bacc={m['bacc']:.4f}, f1M={m['f1m']:.4f}, auc={m['auc']:.4f} "
            f"(fit {m['fit_s']:.1f}s)")

    out = {}
    for k, v in allm.items():
        arr = np.asarray(v, dtype=float)
        out[k] = dict(
            mean=float(arr.mean()),
            std=float(arr.std()),
            ci_lower=float(np.percentile(arr, 2.5)),
            ci_upper=float(np.percentile(arr, 97.5))
        )
    return out

def write_classifier_csv(metrics):
    rows=[]
    for metric, mv in metrics.items():
        rows.append(dict(
            dataset=cfg.DATASET, ablation="classifier", variant="logistic", metric=metric,
            mean=mv["mean"], std=mv["std"], ci_lower=mv["ci_lower"], ci_upper=mv["ci_upper"]
        ))
    df = pd.DataFrame(rows)
    out_csv = cfg.OUT / cfg.DATASET / "classifier_ablation.csv"
    if out_csv.exists() and not cfg.OVERWRITE_CSV:
        raise RuntimeError(f"{out_csv} exists and OVERWRITE_CSV=False")
    df.to_csv(out_csv, index=False)
    log(f"Saved → {out_csv}")
    return out_csv

# =============================== MAIN =====================================
def main():
    log("="*92)
    log("OPENSLIDEFM — CLASSIFIER RE-RUN (clean, split-aligned, no stale artifacts)")
    log(f"Device: {cfg.DEVICE}  |  Workspace: {cfg.WORKSPACE}")
    log(f"Results: {cfg.OUT}")
    log("="*92)

    # sanity
    if not cfg.F05.exists(): raise FileNotFoundError(cfg.F05)
    if not cfg.F20.exists(): raise FileNotFoundError(cfg.F20)

    # clean stale classifier artifacts (CSV + fold checkpoints)
    if cfg.CLEAR_OLD_CLASSIFIER_ARTIFACTS:
        csv_path = cfg.OUT / cfg.DATASET / "classifier_ablation.csv"
        if csv_path.exists():
            os.remove(csv_path); log(f"[CLEAN] removed {csv_path}")
        ck_root = cfg.OUT / cfg.DATASET / "checkpoints"
        if ck_root.exists():
            for d in ck_root.glob("classifier__*"):
                shutil.rmtree(d, ignore_errors=True); log(f"[CLEAN] removed {d}")

    mb = load_manifest_base()
    with Heartbeat(label=cfg.DATASET, sec=cfg.HEARTBEAT_SEC):
        # Splits: reuse if present (to match other ablations); else create
        splits = load_or_create_splits(mb)

        # Vectors and evaluation
        X, y = build_vectors_both_scales(mb)
        log(f"Sanity: n_samples={len(y)}, X_dim={X.shape[1]}, classes={pd.Series(y).nunique()}")
        metrics = eval_classifier(X, y, splits)

    # Save & print compact summary
    csv_out = write_classifier_csv(metrics)
    log("\n================ CLASSIFIER SUMMARY ================")
    for k in ["accuracy","auc","balanced_accuracy","f1_macro"]:
        mv = metrics[k]
        log(f"{k:<17} → logistic   mean={mv['mean']:.4f} ± {mv['std']:.4f} "
            f"[{mv['ci_lower']:.4f}, {mv['ci_upper']:.4f}]")
    log("====================================================")
    log(f"RUN COMPLETE ✓  (CSV: {csv_out})")
    log("="*92)

if __name__ == "__main__":
    main()
