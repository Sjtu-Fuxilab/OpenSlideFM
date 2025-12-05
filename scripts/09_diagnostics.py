#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Post-Pretraining Diagnostics
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# Script 6C — Post-Pretraining Diagnostics

import os, sys, json, math, hashlib, shutil, subprocess, warnings, tempfile
from pathlib import Path
from datetime import datetime, timedelta

# -------- Paths --------
WORKSPACE   = Path(r"D:\个人文件夹\Sanwal\OpenSlide")
LOGS_DIR    = WORKSPACE / "logs"
MODELS_DIR  = WORKSPACE / "models"
FEAT05_DIR  = WORKSPACE / "features" / "scale0p5"
FEAT20_DIR  = WORKSPACE / "features" / "scale2p0"
DIAG_DIR    = WORKSPACE / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV = LOGS_DIR / "script6_train_log.csv"

# -------- Deps (install quietly if missing) --------
def _ensure(pkgs):
    miss=[]
    for name, spec in pkgs:
        try: __import__(name)
        except Exception: miss.append(spec)
    if miss:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])

_ensure([("pandas","pandas>=2.0"), ("numpy","numpy>=1.24")])

import pandas as pd, numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# -------- Helpers --------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    # copy to temp to avoid Windows file-lock while training writes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(path, tmp_path)
        df = pd.read_csv(tmp_path)
    except Exception:
        df = pd.DataFrame()
    finally:
        try: tmp_path.unlink(missing_ok=True)
        except: pass
    return df

def list_ckpts(models_dir: Path):
    exts = (".pt",".pth",".safetensors")
    return sorted([p for p in models_dir.glob("*") if p.suffix.lower() in exts],
                  key=lambda x: x.stat().st_mtime)

def sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def try_torch_load(path: Path):
    if not HAS_TORCH: return False, {"error":"torch not available"}
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        meta = {"type": type(obj).__name__}
        if isinstance(obj, dict): meta["top_keys"] = list(obj.keys())[:8]
        return True, meta
    except Exception as e:
        return False, {"error": str(e)[:180]}

def count_2scale_slides():
    s05 = {p.stem for p in FEAT05_DIR.glob("*.npy")}
    s20 = {p.stem for p in FEAT20_DIR.glob("*.npy")}
    return len(s05 & s20), len(s05), len(s20)

def rolling_median(x: pd.Series, frac=0.1):
    if len(x) == 0: return np.nan
    k = max(3, int(len(x)*frac))
    if k % 2 == 0: k += 1
    return x.rolling(k, center=True, min_periods=max(3,k//3)).median()

# -------- Load logs (robust to missing cols) --------
df = safe_read_csv(LOG_CSV)
diag = {"time": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(WORKSPACE),
        "log_csv_exists": LOG_CSV.exists(),
        "log_rows": int(len(df))}

for c in ["epoch","step","loss","loss_byol","loss_mfr","tps","vram_gb","ts"]:
    if c not in df.columns:
        if c == "ts":
            df[c] = datetime.now().isoformat(timespec="seconds")
        else:
            df[c] = np.nan

# Coerce numeric
for c in ["epoch","step","loss","loss_byol","loss_mfr","tps","vram_gb"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -------- Derive training stats --------
steps_logged = int(df["step"].max()) if len(df) else 0
diag["steps_logged"] = steps_logged

# Loss trend (smoothed medians over first/last ~10%)
if len(df) > 3 and df["loss"].notna().any():
    n = len(df)
    head = df["loss"].dropna().iloc[:max(3, n//10)]
    tail = df["loss"].dropna().iloc[-max(3, n//10):]
    start_med = float(np.median(head)) if len(head) else np.nan
    end_med   = float(np.median(tail)) if len(tail) else np.nan
    rel_impr  = float((start_med - end_med) / start_med) if (start_med and start_med==start_med) else np.nan
else:
    start_med = end_med = rel_impr = np.nan

diag["loss_start_median"] = start_med
diag["loss_end_median"]   = end_med
diag["loss_rel_improvement"] = rel_impr

# TPS (tokens/sec) robust median over recent rows (filter tiny/NaN)
RECENT_ROWS = 200
MIN_VALID_TPS = 100
tps_recent = None
if len(df):
    tail = df.tail(RECENT_ROWS).copy()
    good = tail["tps"].where((tail["tps"] > MIN_VALID_TPS) & np.isfinite(tail["tps"]))
    if good.notna().any():
        tps_recent = float(np.nanmedian(good))
    elif len(df) >= 2:
        # 2-point fallback from last two rows
        r1, r0 = df.iloc[-1], df.iloc[-2]
        try:
            t1 = datetime.fromisoformat(str(r1["ts"]))
            t0 = datetime.fromisoformat(str(r0["ts"]))
            dt = (t1 - t0).total_seconds()
        except Exception:
            dt = None
        dstep = (r1["step"] - r0["step"]) if (np.isfinite(r1["step"]) and np.isfinite(r0["step"])) else 0
        # Use your Script-6 batch sizing (3 slides × (1200+400) tokens)
        TOKENS_PER_BATCH = 3 * (1200 + 400)
        if dt and dt > 0 and dstep > 0:
            tps_recent = float((dstep * TOKENS_PER_BATCH) / dt)

diag["tps_recent_median"] = tps_recent if tps_recent is not None else None
diag["vram_last_gb"] = float(df["vram_gb"].dropna().iloc[-1]) if df["vram_gb"].notna().any() else None

# Staleness
last_ts = None
if len(df):
    try: last_ts = datetime.fromisoformat(str(df["ts"].iloc[-1]))
    except Exception: last_ts = None
diag["last_log_update"] = last_ts.isoformat(timespec="seconds") if last_ts else None
diag["log_stale_over_5min"] = bool((datetime.now() - last_ts) > timedelta(minutes=5)) if last_ts else None

# -------- Feature coverage (2-scale) --------
n_both, n05, n20 = count_2scale_slides()
diag["features_2scale_intersection"] = int(n_both)
diag["features_0p5_count"] = int(n05)
diag["features_2p0_count"] = int(n20)

# -------- Checkpoints --------
ckpts = list_ckpts(MODELS_DIR)
diag["checkpoint_count"] = int(len(ckpts))
ckpt_info = []
for p in ckpts[-6:]:
    ok, meta = try_torch_load(p)
    ckpt_info.append({
        "file": str(p),
        "size_mb": round(p.stat().st_size/(1024**2),2),
        "sha256_12": sha256_12(p),
        "load_ok": bool(ok),
        "meta": meta
    })
diag["checkpoints_recent"] = ckpt_info

# Suggested selection (latest by mtime)
diag["suggest_checkpoint"] = (str(ckpts[-1]) if len(ckpts) else None)

# -------- Gates (PASS/WARN/FAIL) --------
gates = []

# G1: 2-scale coverage
if n_both >= 18000:
    gates.append(("G1_2scale_coverage", "PASS", f"{n_both} slides with both scales"))
elif n_both >= 15000:
    gates.append(("G1_2scale_coverage", "WARN", f"{n_both} < expected; verify features export"))
else:
    gates.append(("G1_2scale_coverage", "FAIL", f"{n_both} very low; investigate features export"))

# G2: loss improvement
if rel_impr == rel_impr:  # not NaN
    if rel_impr >= 0.60:
        gates.append(("G2_loss_improvement", "PASS", f"relative drop {rel_impr:.2f}"))
    elif rel_impr >= 0.30:
        gates.append(("G2_loss_improvement", "WARN", f"modest drop {rel_impr:.2f}"))
    else:
        gates.append(("G2_loss_improvement", "FAIL", f"weak drop {rel_impr:.2f}"))
else:
    gates.append(("G2_loss_improvement", "WARN", "loss trend unavailable"))

# G3: throughput (tokens/sec)
if tps_recent is None:
    gates.append(("G3_throughput", "WARN", "no recent TPS in logs"))
elif tps_recent >= 20000:
    gates.append(("G3_throughput", "PASS", f"{tps_recent:.0f} tok/s"))
elif tps_recent >= 5000:
    gates.append(("G3_throughput", "WARN", f"{tps_recent:.0f} tok/s"))
else:
    gates.append(("G3_throughput", "FAIL", f"{tps_recent:.0f} tok/s"))

# G4: checkpoints presence & loadability
if len(ckpts) == 0:
    gates.append(("G4_checkpoints", "FAIL", "no model files in /models"))
elif any(not c["load_ok"] for c in ckpt_info):
    bad = sum(1 for c in ckpt_info if not c["load_ok"])
    gates.append(("G4_checkpoints", "WARN", f"{bad} recent checkpoint(s) failed to load"))
else:
    gates.append(("G4_checkpoints", "PASS", f"{len(ckpts)} file(s), latest loads OK"))

diag["gates"] = [{"name": n, "status": s, "detail": d} for (n,s,d) in gates]

# -------- Save reports --------
OUT_JSON = DIAG_DIR / "script6b_posttrain_diagnostics.json"
OUT_TXT  = DIAG_DIR / "script6b_posttrain_diagnostics.txt"
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2, ensure_ascii=False)

lines = []
lines.append(f"== Script 6B — Post-Pretraining Diagnostics (no matplotlib) ==")
lines.append(f"time={diag['time']}")
lines.append(f"workspace={diag['workspace']}")
lines.append(f"log_csv_exists={diag['log_csv_exists']} rows={diag['log_rows']} steps_logged={diag['steps_logged']}")
lines.append(f"loss_start_median={diag['loss_start_median']}")
lines.append(f"loss_end_median={diag['loss_end_median']}")
lines.append(f"loss_rel_improvement={diag['loss_rel_improvement']}")
lines.append(f"tps_recent_median={diag['tps_recent_median']}")
lines.append(f"vram_last_gb={diag['vram_last_gb']}")
lines.append(f"last_log_update={diag['last_log_update']}  stale_over_5min={diag['log_stale_over_5min']}")
lines.append(f"features_2scale_intersection={diag['features_2scale_intersection']}  (0.5={diag['features_0p5_count']}, 2.0={diag['features_2p0_count']})")
lines.append(f"checkpoint_count={diag['checkpoint_count']}  suggest_checkpoint={diag['suggest_checkpoint']}")
for c in ckpt_info:
    lines.append(f"  - {c['file']}  size={c['size_mb']} MB  sha256[:12]={c['sha256_12']}  load_ok={c['load_ok']}  meta={c['meta']}")

lines.append("\nGATES:")
for (n,s,d) in gates:
    tag = {"PASS":"[ OK ]", "WARN":"[WARN]", "FAIL":"[FAIL]"}[s]
    lines.append(f" {tag} {n}: {d}")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\n[OK] Saved: {OUT_JSON}")
print(f"[OK] Saved: {OUT_TXT}")
