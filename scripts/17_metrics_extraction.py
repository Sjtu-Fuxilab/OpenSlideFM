#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Metrics from OOF
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# metrics_from_oof.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# === EDIT THIS TO THE RUN YOU CARE ABOUT ===
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "./results")))
OOF_CSV = next((p for p in [
    RESULTS_DIR / "oof_predictions.csv",
    RESULTS_DIR / "oof.csv"
] if p.exists()), None)
assert OOF_CSV and OOF_CSV.exists(), f"Missing OOF file in {RESULTS_DIR}"

df = pd.read_csv(OOF_CSV)
assert "true_isup" in df.columns, "true_isup column not found"

y_true = df["true_isup"].astype(int).values
num_classes = int(max(y_true.max(), 5) + 1)  # expect 6 for PANDA

# ---- get probabilities (prob_* or logit_* -> softmax) ----
prob_cols = [c for c in df.columns if c.startswith("prob_")]
logit_cols = [c for c in df.columns if c.startswith("logit_")]
if prob_cols:
    prob_cols = sorted(prob_cols, key=lambda c: int(c.split("_")[-1]))
    P = df[prob_cols].to_numpy(float)
    # normalize (safety)
    s = P.sum(axis=1, keepdims=True); s[s==0] = 1.0
    P = P / s
elif logit_cols:
    logit_cols = sorted(logit_cols, key=lambda c: int(c.split("_")[-1]))
    Z = df[logit_cols].to_numpy(float)
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
else:
    raise RuntimeError("Neither prob_* nor logit_* columns found in OOF file.")

assert P.shape[1] == num_classes, f"Expected {num_classes} columns, got {P.shape[1]}"

def safe_ovr_macro_auroc(y, prob_mat):
    try:
        return float(roc_auc_score(y, prob_mat, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

def thresh_scores(y, P, thr):
    y_bin = (y >= thr).astype(int)
    s_bin = P[:, thr:].sum(axis=1)
    return y_bin, s_bin

def bin_metrics(y_bin, s_bin):
    auroc = roc_auc_score(y_bin, s_bin)
    aupr  = average_precision_score(y_bin, s_bin)
    return float(auroc), float(aupr)

metrics = {}
metrics["macro_auroc_ovr"] = safe_ovr_macro_auroc(y_true, P)

thresh_list = [1,2,3,4,5]
metrics["thresholds"] = {}
for t in thresh_list:
    yb, sb = thresh_scores(y_true, P, t)
    auroc, aupr = bin_metrics(yb, sb)
    metrics["thresholds"][f">={t}"] = {"auroc": auroc, "auprc": aupr, "pos_rate": float(yb.mean())}

# per-provider (optional)
prov_col = "data_provider" if "data_provider" in df.columns else None
by_prov_rows = []
if prov_col:
    for prov, dsub in df.groupby(prov_col):
        y_sub = dsub["true_isup"].astype(int).values
        if prob_cols:
            P_sub = dsub[prob_cols].to_numpy(float)
            s = P_sub.sum(axis=1, keepdims=True); s[s==0]=1.0
            P_sub /= s
        else:
            Z = dsub[logit_cols].to_numpy(float)
            Z = Z - Z.max(axis=1, keepdims=True)
            P_sub = np.exp(Z); P_sub /= P_sub.sum(axis=1, keepdims=True)
        row = {"provider": prov, "macro_auroc_ovr": safe_ovr_macro_auroc(y_sub, P_sub), "n": int(len(dsub))}
        for t in thresh_list:
            yb, sb = thresh_scores(y_sub, P_sub, t)
            auroc, aupr = bin_metrics(yb, sb)
            row[f"AUROC_>={t}"] = auroc
            row[f"AUPRC_>={t}"] = aupr
        by_prov_rows.append(row)

# save
(RESULTS_DIR / "figures").mkdir(exist_ok=True)
with open(RESULTS_DIR / "metrics_auc.json", "w") as f:
    json.dump(metrics, f, indent=2)
if by_prov_rows:
    pd.DataFrame(by_prov_rows).to_csv(RESULTS_DIR / "metrics_auc_by_provider.csv", index=False)

# print
print("=== PANDA AUROC/AUPRC (from OOF) ===")
print(f"Run dir: {RESULTS_DIR}")
print(f"Macro AUROC (OvR, {num_classes}-class): {metrics['macro_auroc_ovr']:.4f}")
print("\nClinically meaningful thresholds (positive = ISUP ≥ t):")
for t in thresh_list:
    m = metrics["thresholds"][f'>={t}']
    print(f"  ISUP ≥{t}:  AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | prevalence {m['pos_rate']*100:.1f}%")
if by_prov_rows:
    print("\nPer-provider:")
    for row in by_prov_rows:
        extras = " | ".join([f"≥{t}:{row[f'AUROC_>={t}']:.3f}" for t in thresh_list])
        print(f"  {row['provider']:10s} | n={row['n']:4d} | Macro AUROC {row['macro_auroc_ovr']:.4f} | {extras}")

print(f"\nSaved: {RESULTS_DIR/'metrics_auc.json'}")
if by_prov_rows:
    print(f"Saved: {RESULTS_DIR/'metrics_auc_by_provider.csv'}")
