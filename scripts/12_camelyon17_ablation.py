#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - CAMELYON17 pN Ablation
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# Script 9 — CAMELYON17 pN (κ) ablation (LOCO) 
import os, sys, re, json, math, time, subprocess, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

WS = Path(os.environ.get("WORKSPACE", "./workspace")))
RAW = WS / r"Raw Data" / "CAMELYON17"
EMB_INDEX = WS / "embeddings" / "camelyon17_index.csv"
MANIFEST  = WS / "manifests" / "manifest_camelyon17.csv"
OUTDIR    = WS / "results" / "cam17_pn_eval" / "ablations"
OUTDIR.mkdir(parents=True, exist_ok=True)

def _ensure(pkgs):
    miss=[]
    for name, spec in pkgs:
        try: __import__(name)
        except Exception: miss.append(spec)
    if miss:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])
_ensure([("numpy","numpy>=1.24"),("pandas","pandas>=2.0"),("sklearn","scikit-learn>=1.3")])

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

CFG = {
  "Cs": [0.1, 0.3, 1.0, 3.0, 10.0],
  "random_state": 17,
  "max_iter": 4000,
  "n_jobs": 4,
  "fallback_k": 5,
  "boots": 2000
}

def _now(): return datetime.now().isoformat(timespec="seconds")

def guess_label_csv(root: Path) -> Path|None:
    cands=[]
    for p in root.rglob("*.csv"):
        try: hdr = pd.read_csv(p, nrows=5)
        except: continue
        cols = set(hdr.columns.str.lower())
        if {"patient","pn"} <= cols: cands.append(p); continue
        if {"case","pn"} <= cols: cands.append(p); continue
        if {"patient","stage"} <= cols: cands.append(p); continue
    return min(cands, key=lambda x: len(x.name)) if cands else None

def _pn_to_int(v):
    if pd.isna(v): return None
    s=str(v).lower()
    m=re.search(r"pn\s*([0-3])", s)
    if m: return int(m.group(1))
    if s.isdigit() and int(s) in (0,1,2,3): return int(s)
    return None

def _center_from_any(x):
    if pd.isna(x): return None
    s=str(x)
    m=re.search(r"center[_\-]?(\d+)", s, flags=re.I)
    if m: return int(m.group(1))
    if s.isdigit(): 
        n=int(s); 
        return n if 0<=n<=9 else None
    return None

def load_labels():
    cand = guess_label_csv(RAW)
    if cand is None:
        raise FileNotFoundError("Place a CAMELYON17 labels CSV (patient, pN[, center]) under Raw Data/CAMELYON17/")
    df = pd.read_csv(cand)
    df.columns = [c.lower() for c in df.columns]
    if "case" in df.columns and "patient" not in df.columns:
        df["patient"] = df["case"]
    if "pn" not in df.columns and "stage" in df.columns:
        df["pn"] = df["stage"]
    assert "patient" in df.columns and "pn" in df.columns
    df["patient"] = df["patient"].astype(str)
    df["patient"] = df["patient"].str.extract(r"(patient[_\-]?\d+)", expand=False).fillna(df["patient"])
    df["pn_int"] = df["pn"].apply(_pn_to_int)
    df = df.dropna(subset=["pn_int"]).copy()
    # center
    if "center" not in df.columns and "centerid" in df.columns:
        df["center"] = df["centerid"]
    if "center" in df.columns:
        df["center"] = df["center"].apply(_center_from_any)
    else:
        df["center"] = None
    # try manifest to fill missing
    if MANIFEST.exists() and df["center"].isna().mean() > 0.1:
        man = pd.read_csv(MANIFEST)
        pcol = None
        for c in ["path","filepath","fullpath","filename"]:
            if c in man.columns.str.lower().tolist():
                pcol = man.columns[[cc.lower()==c for cc in man.columns]].tolist()[0]
                break
        if pcol is not None:
            def _pt(s):
                s0=Path(str(s)).stem.lower()
                m=re.search(r"(patient[_\-]?\d+)", s0)
                return m.group(1) if m else s0.split("_node")[0]
            mp = (man.assign(_patient=man[pcol].astype(str).apply(_pt),
                             _center=man[pcol].astype(str).apply(_center_from_any))
                     .dropna(subset=["_patient","_center"])
                     .groupby("_patient")["_center"].agg(lambda s:int(pd.Series(s).mode().iloc[0]))
                     .reset_index().rename(columns={"_patient":"patient","_center":"center"}))
            df = df.merge(mp, on="patient", how="left", suffixes=("","_m"))
            df["center"] = df["center"].fillna(df["center_m"])
            df = df.drop(columns=[c for c in df.columns if c.endswith("_m")])
    df["center"] = df["center"].apply(lambda v: int(v) if pd.notna(v) else -1)
    df["pn_int"] = df["pn_int"].astype(int)
    return df[["patient","center","pn_int"]]

def load_emb_index():
    assert EMB_INDEX.exists(), f"Missing {EMB_INDEX}"
    df = pd.read_csv(EMB_INDEX)
    assert {"slide_id","path_emb"} <= set(df.columns)
    return df

def patient_from_slide(sid: str) -> str:
    s = sid.lower()
    m = re.search(r"(patient[_\-]?\d+)", s)
    return m.group(1) if m else s.split("_node")[0]

def load_embeddings(df_idx):
    X=[]; S=[]; P=[]
    for sid, p in zip(df_idx["slide_id"], df_idx["path_emb"]):
        try:
            v = np.load(p).astype(np.float32)
            if v.ndim!=1 or v.shape[0]!=768: continue
        except: 
            continue
        X.append(v); S.append(str(sid)); P.append(patient_from_slide(str(sid)))
    X = np.stack(X, axis=0) if X else np.zeros((0,768), dtype=np.float32)
    return X, np.array(S), np.array(P)

def aggregate_patient(X, pats):
    uniq = pd.unique(pats)
    P = []; order=[]
    for u in uniq:
        idx = np.where(pats==u)[0]
        P.append(X[idx].mean(axis=0))
        order.append(u)
    return np.stack(P,axis=0) if P else np.zeros((0,768),np.float32), np.array(order)

def fit_predict(model_key, C, Xtr, ytr, Xte):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    if model_key=="logreg_l2":
        clf = LogisticRegression(multi_class="multinomial", solver="saga",
                                 penalty="l2", C=C, max_iter=CFG["max_iter"],
                                 n_jobs=CFG["n_jobs"], class_weight="balanced",
                                 random_state=CFG["random_state"])
    elif model_key=="logreg_l1":
        clf = LogisticRegression(multi_class="multinomial", solver="saga",
                                 penalty="l1", C=C, max_iter=CFG["max_iter"],
                                 n_jobs=CFG["n_jobs"], class_weight="balanced",
                                 random_state=CFG["random_state"])
    elif model_key=="ridge":
        # RidgeClassifier uses alpha, roughly alpha≈1/C
        clf = RidgeClassifier(alpha=1.0/max(C,1e-6), class_weight="balanced", random_state=CFG["random_state"])
    else:
        raise ValueError("unknown model_key")
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    return yhat

def qw_kappa(y, yhat): return cohen_kappa_score(y, yhat, weights="quadratic")

def bootstrap_ci(y, yhat, groups, B=2000, seed=123):
    rng = np.random.default_rng(seed)
    # group by patient
    pts = pd.unique(groups)
    if len(pts)==0: return (float("nan"), float("nan"))
    mapping = {p: np.where(groups==p)[0] for p in pts}
    vals=[]
    for _ in range(B):
        idx=[]
        for _ in range(len(pts)):
            pick = rng.choice(pts)
            idx.extend(mapping[pick])
        idx = np.array(idx, dtype=int)
        vals.append(qw_kappa(y[idx], yhat[idx]))
    return float(np.nanpercentile(vals,2.5)), float(np.nanpercentile(vals,97.5))

# ---------------- Run ----------------
print("== Script 9B — CAMELYON17 pN LOCO ablation ==")
print(json.dumps({"time": _now(), "workspace": str(WS)}, indent=2))

df_idx = load_emb_index()
X_s, slide_ids, pats_s = load_embeddings(df_idx)
Xp, patients = aggregate_patient(X_s, pats_s)
df_lbl = load_labels()

# align
pt2row = {p:i for i,p in enumerate(patients)}
y = np.full((Xp.shape[0],), -1, dtype=int)
c = np.full((Xp.shape[0],), -1, dtype=int)
for _, r in df_lbl.iterrows():
    i = pt2row.get(r["patient"])
    if i is not None:
        y[i] = int(r["pn_int"]); c[i] = int(r["center"])
keep = (y>=0)
Xp = Xp[keep]; y = y[keep]; c = c[keep]; patients = patients[keep]

centers = [int(v) for v in pd.unique(c) if v!=-1]
use_loco = len(centers) >= 2
print(f"[MODE] {'LOCO' if use_loco else str(CFG['fallback_k'])+'-fold CV'}  centers={sorted(centers) if centers else 'NONE'}")
print(f"[DATA] patients={len(patients)}  class_counts=" + str(pd.Series(y).value_counts().sort_index().to_dict()))

models = [("logreg_l2",), ("logreg_l1",), ("ridge",)]
grid = []
for mk in [m[0] for m in models]:
    for C in CFG["Cs"]:
        grid.append((mk, float(C)))

rows=[]
all_preds = {}  # key: (mk,C) -> per-patient predictions (stacked across folds for overall)
for mk, C in grid:
    preds=[]; truths=[]; groups=[]
    per_fold=[]
    if use_loco:
        for cc in sorted(pd.unique(c)):
            if cc==-1: continue
            te = np.where(c==cc)[0]
            tr = np.where(c!=cc)[0]  # include -1 in training
            if len(te)==0 or len(tr)==0: continue
            yhat = fit_predict(mk, C, Xp[tr], y[tr], Xp[te])
            k = qw_kappa(y[te], yhat)
            per_fold.append(("CEN"+str(int(cc)), int(len(te)), float(k)))
            preds.extend(yhat.tolist()); truths.extend(y[te].tolist()); groups.extend(patients[te].tolist())
    else:
        skf = StratifiedKFold(n_splits=CFG["fallback_k"], shuffle=True, random_state=CFG["random_state"])
        fold=0
        for tr, te in skf.split(Xp, y):
            fold+=1
            yhat = fit_predict(mk, C, Xp[tr], y[tr], Xp[te])
            k = qw_kappa(y[te], yhat)
            per_fold.append(("FOLD"+str(fold), int(len(te)), float(k)))
            preds.extend(yhat.tolist()); truths.extend(y[te].tolist()); groups.extend(patients[te].tolist())
    preds = np.array(preds, dtype=int); truths = np.array(truths, dtype=int); groups = np.array(groups)
    mean_k = float(np.mean([r[2] for r in per_fold])) if per_fold else float("nan")
    rows.append({
        "model": mk, "C": C, "kappa_qw_mean": mean_k,
        "folds": len(per_fold),
        "detail": "; ".join([f"{lab}:n={n}|κ={k:.3f}" for lab,n,k in per_fold])
    })
    all_preds[(mk,C)] = (truths, preds, groups)

df = pd.DataFrame(rows).sort_values("kappa_qw_mean", ascending=False)
df.to_csv(OUTDIR / "ablations_summary.csv", index=False)

best = df.iloc[0].to_dict()
bkey = (best["model"], float(best["C"]))
y_true, y_pred, pgroup = all_preds[bkey]
ci_lo, ci_hi = bootstrap_ci(y_true, y_pred, pgroup, B=CFG["boots"])
overall_k = qw_kappa(y_true, y_pred)

# save best predictions
pd.DataFrame({"patient": pgroup, "y_true": y_true, "y_pred": y_pred}).to_csv(OUTDIR/"best_patient_predictions.csv", index=False)
# save meta
meta = {
  "time": _now(),
  "mode": "LOCO" if use_loco else f"{CFG['fallback_k']}-fold-CV",
  "best_model": best["model"],
  "best_C": float(best["C"]),
  "kappa_qw_mean_cv": float(best["kappa_qw_mean"]) if not math.isnan(best["kappa_qw_mean"]) else None,
  "overall_kappa_qw": float(overall_k),
  "kappa_ci95": [float(ci_lo), float(ci_hi)],
  "class_counts": pd.Series(y).value_counts().sort_index().to_dict()
}
(Path(OUTDIR/"best_config.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

# human summary
lines = [
  f"time={meta['time']}",
  f"mode={meta['mode']}",
  f"best={meta['best_model']}  C={meta['best_C']}",
  f"mean_cv_kappa_qw={meta['kappa_qw_mean_cv']:.4f}" if meta["kappa_qw_mean_cv"] is not None else "mean_cv_kappa_qw=nan",
  f"overall_kappa_qw={meta['overall_kappa_qw']:.4f}",
  f"ci95=[{meta['kappa_ci95'][0]:.4f}, {meta['kappa_ci95'][1]:.4f}]",
  f"class_counts={meta['class_counts']}"
]
(Path(OUTDIR/"SUMMARY.txt")).write_text("\n".join(lines), encoding="utf-8")

print("\n== Ablation complete ==")
print(json.dumps(meta, indent=2))
print(f"[OK] ablations_summary.csv → {OUTDIR/'ablations_summary.csv'}")
print(f"[OK] best_config.json     → {OUTDIR/'best_config.json'}")
print(f"[OK] best_patient_predictions.csv → {OUTDIR/'best_patient_predictions.csv'}")
