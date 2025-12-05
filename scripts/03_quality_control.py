#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - QC & Tissue Mask
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# OP_FM — Script 3: QC & Tissue Mask (metrics, exclusions, figures)

import os, sys, time, json, math, traceback, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Local import (avoid global hard fail if missing — you already used it in Script 1)
try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python is required for Script 3. Install and rerun.") from e

# ----------------------------- Inputs & Outputs -----------------------------
assert 'WSI_ROOT' in globals() and 'WORKSPACE' in globals() and 'SUBDIRS' in globals(), \
    "Please run Script 1 first to define WSI_ROOT/WORKSPACE/SUBDIRS."

MANIFEST_PARQUET = SUBDIRS["manifests"] / "manifest_tcga.parquet"
assert MANIFEST_PARQUET.exists(), f"Manifest not found at {MANIFEST_PARQUET}. Run Script 2 first."

QC_METRICS_PARQUET = SUBDIRS["qc"] / "qc_metrics_tcga.parquet"
QC_METRICS_CSV     = SUBDIRS["qc"] / "qc_metrics_tcga.csv"
QC_EXCLUSIONS_CSV  = SUBDIRS["qc"] / "exclusions_tcga.csv"
QC_THUMBS_DIR      = SUBDIRS["qc"] / "thumbs"

FIG_DIR = SUBDIRS["figures"]
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUBDIRS["qc"].mkdir(parents=True, exist_ok=True)
QC_THUMBS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- Config knobs --------------------------------
# Set to None to process all slides; set to a small number for a smoke test
QC_MAX_SLIDES = None  # e.g., 500

# Thumbnail target (max dimension in pixels). Larger = more accurate, slower.
THUMB_MAX_SIDE = 1024

# QC thresholds (tune if you need stricter/looser gating)
MIN_TISSUE_PCT   = 0.10   # exclude if < 10% tissue
MAX_WHITE_PCT    = 0.75   # exclude if > 75% white background
MIN_BLUR_VAR     = 15.0   # exclude if Laplacian var < 15 (thumbnail-level)
MAX_PEN_PCT      = 0.02   # exclude if > 2% pen/ink (blue-ish, high saturation)

# HSV gates (0..255 space from PIL)
HSV_S_TISSUE_MIN = 20     # tissue tends to have some saturation
HSV_V_WHITE_MIN  = 230    # very bright ~white pixels (V high)
HSV_S_WHITE_MAX  = 30     # near-white has low saturation
# Blue ink (pen) heuristic in HSV:
#   - Hue roughly in 180..255 (on PIL 0..255 scale; ~ 255 ~ 360°)
#   - Saturation high to avoid white/gray
HSV_H_BLUE_MIN   = 170
HSV_H_BLUE_MAX   = 255
HSV_S_PEN_MIN    = 60

# Concurrency for QC (safe to run multiple readers; each opens its own slide)
MAX_WORKERS = min(8, (os.cpu_count() or 8))

# ----------------------------- Helpers -------------------------------------
def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_thumbnail(slide_path: Path, max_side: int = 1024) -> Image.Image:
    """Open slide and return a PIL thumbnail with max_side dimension."""
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.dimensions
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    tw, th = int(w / scale), int(h / scale)
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    slide.close()
    return thumb

def to_hsv_np(img_rgb: Image.Image):
    """Return HSV uint8 arrays (H,S,V in 0..255) from an RGB PIL image."""
    hsv = img_rgb.convert("HSV")
    a = np.array(hsv, dtype=np.uint8)
    H, S, V = a[..., 0], a[..., 1], a[..., 2]
    return H, S, V

def laplacian_var(gray_u8: np.ndarray, tissue_mask: np.ndarray = None) -> float:
    """Variance of 3x3 Laplacian (manual conv) over uint8 grayscale. Optionally restricted to tissue."""
    # 3x3 kernel:
    #  0  1  0
    #  1 -4  1
    #  0  1  0
    g = gray_u8.astype(np.float32)
    # pad edges
    p = np.pad(g, 1, mode="reflect")
    c  = -4 * p[1:-1, 1:-1]
    n  = 1 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
    lap = c + n
    if tissue_mask is not None:
        mask = tissue_mask.astype(bool)
        if mask.sum() == 0:
            return 0.0
        vals = lap[mask]
    else:
        vals = lap.ravel()
    # variance
    return float(np.var(vals))

def qc_on_thumbnail(img: Image.Image):
    """Compute QC metrics on a thumbnail image."""
    # RGB -> HSV
    H, S, V = to_hsv_np(img)
    # grayscale for blur
    gray = np.array(img.convert("L"), dtype=np.uint8)

    # Tissue mask: S high AND not pure white
    tissue_mask = (S >= HSV_S_TISSUE_MIN) & (V < HSV_V_WHITE_MIN)

    # White mask: very bright with low saturation
    white_mask = (V >= HSV_V_WHITE_MIN) & (S <= HSV_S_WHITE_MAX)

    # Pen mask: blue-ish + saturated
    pen_mask = (H >= HSV_H_BLUE_MIN) & (H <= HSV_H_BLUE_MAX) & (S >= HSV_S_PEN_MIN)

    total = img.size[0] * img.size[1]
    tissue_pct = float(tissue_mask.sum() / total)
    white_pct  = float(white_mask.sum() / total)
    pen_pct    = float(pen_mask.sum() / total)

    # Blur (variance of Laplacian) only on tissue
    blur_val = laplacian_var(gray, tissue_mask)

    # Simple stats inside tissue
    if tissue_mask.sum() > 0:
        brightness_mean = float(V[tissue_mask].mean())
        saturation_mean = float(S[tissue_mask].mean())
    else:
        brightness_mean = float(V.mean())
        saturation_mean = float(S.mean())

    return {
        "tissue_pct": tissue_pct,
        "white_pct": white_pct,
        "pen_pct": pen_pct,
        "blur_var": blur_val,
        "brightness_mean": brightness_mean,
        "saturation_mean": saturation_mean,
    }, tissue_mask

def qc_reason_flags(m, thresholds):
    """Return list of exclusion reasons (strings). Empty list => keep."""
    reasons = []
    if m["tissue_pct"] < thresholds["min_tissue_pct"]:
        reasons.append(f"low_tissue<{thresholds['min_tissue_pct']:.2f}")
    if m["white_pct"] > thresholds["max_white_pct"]:
        reasons.append(f"white>{thresholds['max_white_pct']:.2f}")
    if m["blur_var"] < thresholds["min_blur_var"]:
        reasons.append(f"blur<{thresholds['min_blur_var']:.1f}")
    if m["pen_pct"] > thresholds["max_pen_pct"]:
        reasons.append(f"pen>{thresholds['max_pen_pct']:.2f}")
    return reasons

def save_thumb_and_mask(slide_id: str, img: Image.Image, tissue_mask: np.ndarray):
    """Save plain thumbnail and a quick tissue overlay for audit."""
    # Save plain thumb (JPEG)
    thumb_path = QC_THUMBS_DIR / f"{slide_id}_thumb.jpg"
    img.save(str(thumb_path), "JPEG", quality=90)

    # Save overlay (red mask)
    overlay = np.array(img).copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    alpha = 0.35
    mask3 = np.stack([tissue_mask]*3, axis=-1)
    overlay = (overlay * (~mask3) + (alpha * overlay + (1 - alpha) * red) * mask3).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    overlay_path = QC_THUMBS_DIR / f"{slide_id}_overlay.jpg"
    overlay_img.save(str(overlay_path), "JPEG", quality=90)

    return str(thumb_path), str(overlay_path)

# ----------------------------- Load manifest --------------------------------
print(f"== OP_FM Script 3: QC & Tissue Mask ==\n[{now_iso()}] Loading manifest:", MANIFEST_PARQUET)
df_manifest = pd.read_parquet(MANIFEST_PARQUET)
df_manifest = df_manifest.copy()

if QC_MAX_SLIDES is not None:
    df_manifest = df_manifest.head(QC_MAX_SLIDES).copy()
print(f"[INFO] Slides to QC: {len(df_manifest)}")

# ----------------------------- Run QC (multi-thread) ------------------------
thresholds = {
    "min_tissue_pct": MIN_TISSUE_PCT,
    "max_white_pct": MAX_WHITE_PCT,
    "min_blur_var": MIN_BLUR_VAR,
    "max_pen_pct": MAX_PEN_PCT,
}

results = []
failures = []
t_start = time.time()

def worker(row):
    slide_path = Path(row["path"])
    slide_id   = str(row["slide_id"])
    cancer     = str(row.get("cancer_code", "UNKNOWN"))
    try:
        img = load_thumbnail(slide_path, THUMB_MAX_SIDE)
        metrics, tissue_mask = qc_on_thumbnail(img)
        reasons = qc_reason_flags(metrics, thresholds)
        thumb_p, overlay_p = save_thumb_and_mask(slide_id, img, tissue_mask)
        rec = {
            "slide_id": slide_id,
            "cancer_code": cancer,
            "path": str(slide_path),
            "tissue_pct": metrics["tissue_pct"],
            "white_pct": metrics["white_pct"],
            "pen_pct": metrics["pen_pct"],
            "blur_var": metrics["blur_var"],
            "brightness_mean": metrics["brightness_mean"],
            "saturation_mean": metrics["saturation_mean"],
            "excluded": int(len(reasons) > 0),
            "reasons": ";".join(reasons) if reasons else "",
            "thumb": thumb_p,
            "overlay": overlay_p,
        }
        return True, rec
    except Exception as e:
        return False, {"slide_id": slide_id, "path": str(slide_path), "error": f"{e.__class__.__name__}: {e}"}

done = 0
last_print = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = [ex.submit(worker, row) for _, row in df_manifest.iterrows()]
    for fut in as_completed(futs):
        ok, payload = fut.result()
        if ok:
            results.append(payload)
        else:
            failures.append(payload)
        done += 1
        now = time.time()
        if now - last_print > 2 or done == len(df_manifest):
            rate = done / (now - t_start + 1e-9)
            print(f"[QC] {done}/{len(df_manifest)} ({rate:.1f} slides/s)")
            last_print = now

elapsed = time.time() - t_start
print(f"[OK] QC completed in {elapsed/60:.1f} min.")

# ----------------------------- Save metrics & exclusions --------------------
df_qc = pd.DataFrame.from_records(results)
df_qc = df_qc.sort_values("slide_id").reset_index(drop=True)
df_qc.to_parquet(QC_METRICS_PARQUET, index=False)
df_qc.to_csv(QC_METRICS_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] QC metrics saved:\n - {QC_METRICS_PARQUET}\n - {QC_METRICS_CSV}")

if failures:
    df_fail = pd.DataFrame(failures)
    fail_path = SUBDIRS["qc"] / "qc_failures.csv"
    df_fail.to_csv(fail_path, index=False, encoding="utf-8-sig")
    print(f"[WARN] {len(failures)} slides failed during QC; see {fail_path}")

# Exclusions file
df_excl = df_qc[df_qc["excluded"] == 1].copy()
df_excl.to_csv(QC_EXCLUSIONS_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Exclusions written: {QC_EXCLUSIONS_CSV} (n={len(df_excl)})")

# ----------------------------- Diagnostics & Figures -----------------------
# 1) Summary prints
n_total = len(df_qc)
n_excl  = len(df_excl)
print("\n== QC Summary ==")
print(f" Total slides QC'd : {n_total:,}")
print(f" Excluded          : {n_excl:,} ({100.0*n_excl/max(1,n_total):.1f}%)")

by_reason = (df_excl["reasons"].str.split(";", expand=True)
             .stack().str.strip().value_counts())
print("\n Exclusions by reason:")
print(by_reason.to_string())

by_cancer_excl = df_excl["cancer_code"].value_counts()
print("\n Exclusions by cancer_code (top 20):")
print(by_cancer_excl.head(20).to_string())

# 2) Histograms of QC metrics
def hist_plot(series, title, xlabel, outname, bins=40):
    plt.figure(figsize=(8,5))
    vals = series.dropna().values
    plt.hist(vals, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    outp = FIG_DIR / outname
    plt.savefig(outp)
    plt.close()
    print(f"[FIG] {outp}")
    return str(outp)

figs = []
figs.append(hist_plot(df_qc["tissue_pct"],     "Tissue % (thumbnail)", "tissue_pct", "qc_tissue_pct_hist.png"))
figs.append(hist_plot(df_qc["white_pct"],      "White % (thumbnail)",  "white_pct",  "qc_white_pct_hist.png"))
figs.append(hist_plot(df_qc["pen_pct"],        "Pen/ink %",            "pen_pct",    "qc_pen_pct_hist.png"))
figs.append(hist_plot(df_qc["blur_var"],       "Blur variance",        "blur_var",   "qc_blur_var_hist.png"))
figs.append(hist_plot(df_qc["brightness_mean"],"Brightness (mean)",    "V_mean",     "qc_brightness_mean_hist.png"))
figs.append(hist_plot(df_qc["saturation_mean"],"Saturation (mean)",    "S_mean",     "qc_saturation_mean_hist.png"))

# 3) Exclusion reason bar
plt.figure(figsize=(10,6))
x = by_reason.index.tolist()
y = by_reason.values.tolist()
plt.bar(x, y)
plt.xticks(rotation=70, ha="right")
plt.ylabel("Excluded slides")
plt.title("Exclusions by reason")
plt.tight_layout()
p_exr = FIG_DIR / "qc_exclusions_by_reason.png"
plt.savefig(p_exr)
plt.close()
print(f"[FIG] {p_exr}")

# 4) Exclusion rate by cancer_code (top 30)
rates = (df_excl["cancer_code"].value_counts() / df_qc["cancer_code"].value_counts()).fillna(0)
rates = rates.sort_values(ascending=False)
plt.figure(figsize=(12,6))
top_rates = rates.head(30)
plt.bar(top_rates.index.astype(str), (100.0*top_rates.values))
plt.xticks(rotation=70, ha="right")
plt.ylabel("Exclusion rate (%)")
plt.title("Exclusion rate by cancer_code (Top 30)")
plt.tight_layout()
p_exrate = FIG_DIR / "qc_exclusion_rate_by_cancer.png"
plt.savefig(p_exrate)
plt.close()
print(f"[FIG] {p_exrate}")

# ----------------------------- Update compute-passport ---------------------
compute_path = SUBDIRS["compute"] / "compute_passport.json"
try:
    with compute_path.open("r", encoding="utf-8") as f:
        cp = json.load(f)
except Exception:
    cp = {"stages": []}

stage_entry = {
    "stage": "qc_tcga",
    "timestamp": now_iso(),
    "inputs": {
        "manifest_parquet": str(MANIFEST_PARQUET),
        "thumb_max_side": THUMB_MAX_SIDE,
    },
    "outputs": {
        "qc_metrics_parquet": str(QC_METRICS_PARQUET),
        "qc_metrics_csv": str(QC_METRICS_CSV),
        "qc_exclusions_csv": str(QC_EXCLUSIONS_CSV),
        "qc_thumbs_dir": str(QC_THUMBS_DIR),
        "figures": figs + [str(p_exr), str(p_exrate)],
    },
    "thresholds": {
        "min_tissue_pct": MIN_TISSUE_PCT,
        "max_white_pct": MAX_WHITE_PCT,
        "min_blur_var": MIN_BLUR_VAR,
        "max_pen_pct": MAX_PEN_PCT,
        "hsv": {
            "S_tissue_min": HSV_S_TISSUE_MIN,
            "V_white_min": HSV_V_WHITE_MIN,
            "S_white_max": HSV_S_WHITE_MAX,
            "H_blue_min": HSV_H_BLUE_MIN,
            "H_blue_max": HSV_H_BLUE_MAX,
            "S_pen_min": HSV_S_PEN_MIN,
        }
    },
    "stats": {
        "n_qc": int(n_total),
        "n_excluded": int(n_excl),
        "elapsed_minutes": float(elapsed / 60.0),
        "failures": int(len(failures)),
        "workers": MAX_WORKERS,
    }
}
cp.setdefault("stages", []).append(stage_entry)

tmp = compute_path.with_suffix(".json.tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(cp, f, ensure_ascii=False, indent=2)
tmp.replace(compute_path)
print(f"\n[OK] Compute-Passport updated: {compute_path}")

print("\nScript 3 complete. Next: Script 4 (Two-scale tiling & token budget).")
