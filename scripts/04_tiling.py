#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Two-scale Tiling & Token Budget
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# OP_FM — Script 4: Two-scale Tiling & Token Budget 

import os, sys, math, json, time, random, datetime, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python is required for tiling. Install and rerun.") from e

# ----------------------------- Prereqs -----------------------------
assert 'WSI_ROOT' in globals() and 'WORKSPACE' in globals() and 'SUBDIRS' in globals(), \
    "Please run Script 1 first to define WSI_ROOT/WORKSPACE/SUBDIRS."

MANIFEST_PARQUET = SUBDIRS["manifests"] / "manifest_tcga.parquet"
QC_METRICS_PARQUET = SUBDIRS["qc"] / "qc_metrics_tcga.parquet"
TILES_DIR = SUBDIRS["tiles"] / "manifests"
TILES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = SUBDIRS["figures"]

# ----------------------------- Config ------------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

# QC policy: 'strict' | 'medium' | 'lenient' | 'none'
QC_POLICY = "medium"  # default, safer after the aggressive pen hits

# Target scales (μm/px)
TARGET_SCALES = [0.5, 2.0]

# Tile geometry
TILE_SIZE = 256     # pixels at the chosen level
OVERLAP   = 32      # pixels
STRIDE    = TILE_SIZE - OVERLAP

# Token budgets (per slide per scale)
MAX_TOKENS = {0.5: 1200, 2.0: 400}

# Tile acceptance
MIN_TILE_TISSUE_COVERAGE = 0.30  # fraction of tile area that must be tissue

# Low-res mask rendering for each slide (thumbnail)
MASK_MAX_SIDE = 2048         # make a thumbnail up to this max side
HSV_S_TISSUE_MIN = 20        # same basics as QC (scaled 0..255)
HSV_V_WHITE_MIN  = 230

# Sampling method for downselecting to budget: 'uniform' or 'variance_topk'
SAMPLING_METHOD = "uniform"

# Execution controls
MAX_WORKERS = min(6, (os.cpu_count() or 8))  # worker = one slide at a time
FORCE_REDO = False  # if True, re-generate even if manifest exists

# Optional quick heatmaps (for 2 random slides)
QUICK_HEATMAPS = True
N_HEATMAPS = 2

# ----------------------------- Helpers ------------------------------
def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def choose_level_for_target_mpp(slide, target_mpp, fallback_base_mpp=0.25):
    """Return (level, approx_mpp) closest to target μm/px."""
    props = slide.properties
    base_mpp = None
    # Try openslide props
    for k in ("openslide.mpp-x", "aperio.MPP"):
        if k in props:
            try:
                base_mpp = float(props.get(k))
                break
            except Exception:
                pass
    if base_mpp is None:
        base_mpp = fallback_base_mpp  # fallback

    best_level = 0
    best_mpp = base_mpp
    for lvl in range(slide.level_count):
        mpp = base_mpp * slide.level_downsamples[lvl]
        if abs(mpp - target_mpp) < abs(best_mpp - target_mpp):
            best_mpp = mpp
            best_level = lvl
    return best_level, float(best_mpp)

def make_tissue_mask(slide, max_side=MASK_MAX_SIDE):
    """Return RGB thumbnail and boolean tissue mask at thumbnail scale."""
    w, h = slide.dimensions
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    tw, th = int(w / scale), int(h / scale)
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")

    hsv = thumb.convert("HSV")
    a = np.array(hsv, dtype=np.uint8)
    H, S, V = a[..., 0], a[..., 1], a[..., 2]
    tissue = (S >= HSV_S_TISSUE_MIN) & (V < HSV_V_WHITE_MIN)
    return thumb, tissue

def grid_positions(level_w, level_h, tile=TILE_SIZE, stride=STRIDE):
    xs = list(range(0, max(level_w - tile, 0) + 1, stride))
    ys = list(range(0, max(level_h - tile, 0) + 1, stride))
    return xs, ys

def coverage_from_mask(mask, level, level_to_mask_scale, x, y, tile=TILE_SIZE):
    """
    Estimate tissue coverage of the tile (x,y,level) using the low-res mask.
    level_to_mask_scale = (sx, sy): multiplies level coords to mask coords.
    """
    sx, sy = level_to_mask_scale
    mx0, my0 = int(x * sx), int(y * sy)
    mx1, my1 = int((x + tile) * sx), int((y + tile) * sy)
    mx0, my0 = max(mx0, 0), max(my0, 0)
    mx1, my1 = min(mx1, mask.shape[1]-1), min(my1, mask.shape[0]-1)
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    roi = mask[my0:my1, mx0:mx1]
    return float(roi.mean())  # True=1, False=0

def sample_tiles_uniform(coords, k, rng):
    if len(coords) <= k:
        return coords
    idx = rng.choice(len(coords), size=k, replace=False)
    return [coords[i] for i in idx]

# Optional variance-based sampler (needs quick per-tile Laplacian on low-res)
def sample_tiles_variance(mask_rgb, coords, k, rng):
    if len(coords) <= k:
        return coords
    # Simple variance proxy on grayscale thumbnail region (downscaled)
    gray = np.array(mask_rgb.convert("L"), dtype=np.float32) / 255.0
    scores = []
    for (x, y) in coords:
        # Take a tiny patch around the mapped region center on the thumbnail
        # This is a rough heuristic; we keep it light
        cx, cy = int(x), int(y)
        # Already in level coords — but we need to work in mask space; caller should pass coords in mask space for this method
        # To keep Script 4 straightforward, we won’t use variance_topk by default.
        scores.append(0.0)
    # Fallback to uniform if not implemented
    return sample_tiles_uniform(coords, k, rng)

def write_parquet(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

# ----------------------------- Load inputs ------------------------------
print(f"== OP_FM Script 4: Two-scale Tiling & Token Budget ==\n[{now_iso()}] Loading:", MANIFEST_PARQUET)
df_manifest = pd.read_parquet(MANIFEST_PARQUET)

# Try to load QC metrics; optional
df_qc = None
if QC_POLICY in ("strict", "medium") and QC_METRICS_PARQUET.exists():
    df_qc = pd.read_parquet(QC_METRICS_PARQUET)
    df_qc = df_qc[["slide_id", "tissue_pct", "white_pct", "pen_pct", "reasons"]].copy()

# ----------------------------- Select slides ------------------------------
if QC_POLICY == "strict" and df_qc is not None:
    keep = df_manifest.merge(df_qc[["slide_id", "reasons"]], on="slide_id", how="left")
    keep = keep[keep["reasons"].isna() | (keep["reasons"] == "")]
elif QC_POLICY == "medium" and df_qc is not None:
    keep = df_manifest.merge(df_qc, on="slide_id", how="left")
    # Ignore 'pen'; only exclude if clearly unusable
    keep = keep[
        (keep["tissue_pct"].fillna(1.0) >= 0.05) &
        (keep["white_pct"].fillna(0.0) <= 0.95)
    ].copy()
else:
    keep = df_manifest.copy()

keep = keep.reset_index(drop=True)
print(f"[INFO] Slides selected under QC policy '{QC_POLICY}': {len(keep):,} out of {len(df_manifest):,}")

# ----------------------------- Per-slide worker ---------------------------
def process_slide(row):
    slide_path = Path(row["path"])
    slide_id   = str(row["slide_id"])
    cancer     = str(row.get("cancer_code", "UNKNOWN"))

    outputs = []
    errors = []
    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        return slide_id, cancer, None, [f"OpenSlideError: {e}"]

    # Build once per slide: thumbnail mask
    try:
        thumb_rgb, mask = make_tissue_mask(slide, MASK_MAX_SIDE)
    except Exception as e:
        slide.close()
        return slide_id, cancer, None, [f"MaskBuildError: {e}"]

    # Dimensions at each level
    level_dims = [slide.level_dimensions[i] for i in range(slide.level_count)]
    base_w, base_h = level_dims[0]

    for target in TARGET_SCALES:
        # output path
        out_path = TILES_DIR / f"{slide_id}_scale{str(target).replace('.','p')}.parquet"
        if out_path.exists() and not FORCE_REDO:
            outputs.append({"scale": target, "manifest": str(out_path), "n_tiles": None, "skipped": True})
            continue

        try:
            level, approx_mpp = choose_level_for_target_mpp(slide, target)
            level_w, level_h = level_dims[level]
            # mapping from level coords to mask coords
            # mask is a thumbnail of base level; compute scale factors
            # mask.shape = (th, tw), thumb corresponds to base (w,h) scaled
            tw, th = thumb_rgb.size
            sx = tw / base_w
            sy = th / base_h
            # level to base downsample
            ds = slide.level_downsamples[level]
            # final: level->mask multiply by (ds * s(mask/base))
            level_to_mask_scale = (sx * ds, sy * ds)

            xs, ys = grid_positions(level_w, level_h, TILE_SIZE, STRIDE)
            # gather candidate coordinates with coverage check
            cand = []
            for y in ys:
                for x in xs:
                    cov = coverage_from_mask(mask, level, level_to_mask_scale, x, y, TILE_SIZE)
                    if cov >= MIN_TILE_TISSUE_COVERAGE:
                        cand.append((x, y))
            n_cand = len(cand)

            # Downselect to budget
            budget = MAX_TOKENS.get(target, 0)
            rng = np.random.default_rng(SEED + hash(slide_id) % (2**16) + int(target*100))
            if SAMPLING_METHOD == "uniform":
                chosen = sample_tiles_uniform(cand, budget, rng)
            else:
                chosen = sample_tiles_uniform(cand, budget, rng)  # keep uniform default

            # Build dataframe
            # Note: store mm-scale positional approximations if needed later (optional)
            data = []
            for idx, (x, y) in enumerate(chosen):
                data.append({
                    "slide_id": slide_id,
                    "cancer_code": cancer,
                    "scale_um_per_px": float(target),
                    "level": int(level),
                    "x": int(x),
                    "y": int(y),
                    "tile_size": TILE_SIZE,
                    "overlap": OVERLAP,
                    "approx_mpp": approx_mpp,
                    "tile_idx": int(idx),
                    "seed": int(SEED),
                })
            df_tiles = pd.DataFrame.from_records(data)

            # Write manifest
            write_parquet(df_tiles, out_path)
            outputs.append({"scale": target, "manifest": str(out_path), "n_tiles": len(df_tiles), "skipped": False})

        except Exception as e:
            errors.append(f"TilingError(scale={target}): {e}")

    slide.close()
    return slide_id, cancer, outputs, errors

# ----------------------------- Run (multi-slide) ---------------------------
t0 = time.time()
done = 0
errors_all = []
per_slide_counts = []

print(f"[{now_iso()}] Starting tiling on {len(keep)} slides with {MAX_WORKERS} workers...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = [ex.submit(process_slide, row) for _, row in keep.iterrows()]
    for fut in as_completed(futs):
        slide_id, cancer, outputs, errs = fut.result()
        done += 1
        if errs:
            for e in errs:
                errors_all.append({"slide_id": slide_id, "error": e})
        if outputs:
            for rec in outputs:
                if rec is None: 
                    continue
                if not rec.get("skipped", False):
                    per_slide_counts.append({
                        "slide_id": slide_id,
                        "cancer_code": cancer,
                        "scale_um_per_px": rec["scale"],
                        "n_tiles": rec["n_tiles"],
                        "manifest": rec["manifest"],
                    })
        if done % 25 == 0 or done == len(keep):
            rate = done / (time.time() - t0 + 1e-9)
            print(f"[TILING] {done}/{len(keep)} ({rate:.2f} slides/s)")

elapsed = time.time() - t0
print(f"[OK] Tiling finished in {elapsed/60:.1f} min.")

# ----------------------------- Summaries & Figures -------------------------
df_sum = pd.DataFrame.from_records(per_slide_counts)
sum_path = SUBDIRS["tiles"] / "tiling_summary_tcga.parquet"
df_sum.to_parquet(sum_path, index=False)
print(f"[OK] Tiling summary: {sum_path}")

if errors_all:
    df_err = pd.DataFrame.from_records(errors_all)
    err_path = SUBDIRS["tiles"] / "tiling_errors_tcga.csv"
    df_err.to_csv(err_path, index=False, encoding="utf-8-sig")
    print(f"[WARN] {len(df_err)} tiling errors logged at {err_path}")

# Figures: tokens/slide per scale
FIG_DIR.mkdir(parents=True, exist_ok=True)
for scale in TARGET_SCALES:
    df_sc = df_sum[df_sum["scale_um_per_px"] == scale]
    if len(df_sc) == 0:
        continue
    plt.figure(figsize=(8,5))
    plt.hist(df_sc["n_tiles"].dropna().values, bins=40)
    plt.xlabel(f"Tokens per slide @ {scale} μm/px")
    plt.ylabel("Slides")
    plt.title(f"Token Distribution @ {scale} μm/px")
    plt.tight_layout()
    outp = FIG_DIR / f"tiling_tokens_dist_scale{str(scale).replace('.','p')}.png"
    plt.savefig(outp)
    plt.close()
    print(f"[FIG] {outp}")

# Optional quick heatmaps for a couple of slides
if QUICK_HEATMAPS and len(df_sum) > 0:
    sample_ids = df_sum["slide_id"].drop_duplicates().sample(min(N_HEATMAPS, df_sum["slide_id"].nunique()), random_state=SEED).tolist()
    for sid in sample_ids:
        try:
            row0 = keep[keep["slide_id"] == sid].iloc[0]
            slide = openslide.OpenSlide(str(row0["path"]))
            thumb_rgb, mask = make_tissue_mask(slide, MASK_MAX_SIDE)
            tw, th = thumb_rgb.size
            # overlay sampled tile centers for 0.5 μm/px only (if present)
            df_s = df_sum[(df_sum["slide_id"] == sid) & (df_sum["scale_um_per_px"] == 0.5)]
            if len(df_s):
                rec = df_s.iloc[0]
                # Recompute level/mapping to draw tile centers
                level, approx_mpp = choose_level_for_target_mpp(slide, 0.5)
                base_w, base_h = slide.level_dimensions[0]
                ds = slide.level_downsamples[level]
                sx = tw / base_w
                sy = th / base_h
                level_to_mask_scale = (sx * ds, sy * ds)

                # Load that slide's manifest
                man_path = Path(rec["manifest"])
                df_tiles = pd.read_parquet(man_path)
                # Draw centers
                overlay = np.array(thumb_rgb).copy()
                for _, t in df_tiles.iterrows():
                    mx = int((t["x"] + TILE_SIZE//2) * level_to_mask_scale[0])
                    my = int((t["y"] + TILE_SIZE//2) * level_to_mask_scale[1])
                    if 0 <= mx < overlay.shape[1] and 0 <= my < overlay.shape[0]:
                        # small dot
                        y0, y1 = max(my-1,0), min(my+2, overlay.shape[0])
                        x0, x1 = max(mx-1,0), min(mx+2, overlay.shape[1])
                        overlay[y0:y1, x0:x1, :] = [255, 0, 0]
                outp = FIG_DIR / f"tiling_heatmap_{sid}.png"
                Image.fromarray(overlay).save(outp)
                print(f"[FIG] {outp}")
            slide.close()
        except Exception as e:
            print(f"[WARN] Heatmap for {sid} failed: {e}")

# ----------------------------- Update compute-passport ---------------------
compute_path = SUBDIRS["compute"] / "compute_passport.json"
try:
    with compute_path.open("r", encoding="utf-8") as f:
        cp = json.load(f)
except Exception:
    cp = {"stages": []}

stage_entry = {
    "stage": "tiling_tcga",
    "timestamp": now_iso(),
    "inputs": {
        "manifest_parquet": str(MANIFEST_PARQUET),
        "qc_metrics_parquet": str(QC_METRICS_PARQUET) if QC_METRICS_PARQUET.exists() else None,
        "qc_policy": QC_POLICY,
        "target_scales_um_per_px": TARGET_SCALES,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "min_tile_tissue_coverage": MIN_TILE_TISSUE_COVERAGE,
        "mask_max_side": MASK_MAX_SIDE,
        "sampling_method": SAMPLING_METHOD,
        "seed": SEED,
    },
    "outputs": {
        "tiling_summary_parquet": str(sum_path),
        "manifests_dir": str(TILES_DIR),
        "figures_dir": str(FIG_DIR),
    },
    "stats": {
        "n_slides_considered": int(len(df_manifest)),
        "n_slides_selected": int(len(keep)),
        "n_slide_scale_entries": int(len(df_sum)),
        "elapsed_minutes": float(elapsed / 60.0),
        "errors": int(len(errors_all)),
    }
}
cp.setdefault("stages", []).append(stage_entry)

tmp = compute_path.with_suffix(".json.tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(cp, f, ensure_ascii=False, indent=2)
tmp.replace(compute_path)
print(f"\n[OK] Compute-Passport updated: {compute_path}")

print("\nScript 4 complete. Next: Script 5 (Frozen-backbone feature extraction to 768-D).")
