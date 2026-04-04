#!/usr/bin/env python3
"""
OpenSlideFM — Complete Pipeline
A computationally efficient multi-scale foundation model for computational
pathology. Dual-scale (0.5 + 2.0 μm/pixel) transformer architecture with
BYOL + Masked Feature Reconstruction pre-training.

Architecture:
  - Backbone: ConvNeXt-Tiny (28M parameters, 768-d features)
  - Aggregator: 6-layer Transformer encoder (43M parameters)
  - Total: 71M parameters
  - Token budget: 1,600 (1,200 high-res + 400 low-res, 3:1 ratio)

Pipeline:
  Script 01: Environment setup & paths
  Script 02: Dataset manifest & provenance
  Script 03: Quality control & tissue masking
  Script 04: Two-scale tiling & token budget
  Script 05: Feature extraction (ConvNeXt-Tiny, frozen)
  Script 06: Self-supervised pre-training (BYOL + MFR)
  Script 06B: Encoder checkpoint finalization
  Script 06C: Post-pretraining diagnostics
  Script 07: TCGA 31-class pan-cancer evaluation
  Script 08: Slide embeddings export
  Script 09: CAMELYON17 pN staging (LOCO-CV)
  Script 10: PANDA feature processing
  Script 11: PANDA Gleason grading (MIL)
  Script 12: PANDA out-of-fold metrics

Usage:
  1. Set environment variables:
       export WORKSPACE=/path/to/workspace
       export WSI_ROOT=/path/to/wsi/slides
       export PANDA_ROOT=/path/to/panda/data    (optional)
  2. Run scripts sequentially or use as importable modules.


"""

import os
from pathlib import Path

# CONFIGURATION — Set these paths for your environment
WORKSPACE = Path(os.environ.get("WORKSPACE", "./workspace"))
WSI_ROOT  = Path(os.environ.get("WSI_ROOT", "./data/wsi"))
PANDA_ROOT = Path(os.environ.get("PANDA_ROOT", "./data/PANDA"))


# SCRIPT 01: ENVIRONMENT SETUP & PATHS
# OP_FM — Script 1: Environment, Paths, and Compute-Passport
import os, sys, json, time, math, platform, shutil, socket, datetime
from pathlib import Path
from typing import Any, Dict, Optional

# USER PATHS
# READ-ONLY: your TCGA WSI root (no writes will ever be performed here)
# WSI_ROOT — set in CONFIGURATION block above
# WORKSPACE: all pipeline outputs go here (and only here)
# WORKSPACE — set in CONFIGURATION block above
# SUBFOLDER LAYOUT
SUBDIRS = {
    "compute": WORKSPACE / "compute",
    "logs": WORKSPACE / "logs",
    "figures": WORKSPACE / "figures",
    "qc": WORKSPACE / "qc",
    "tiles": WORKSPACE / "tiles",
    "features": WORKSPACE / "features",
    "embeddings": WORKSPACE / "embeddings",
    "attn": WORKSPACE / "attn",
    "leak_audit": WORKSPACE / "leak_audit",
    "preanalytics": WORKSPACE / "preanalytics",
    "artifacts": WORKSPACE / "artifacts",
    "manifests": WORKSPACE / "manifests",
    "hashes": WORKSPACE / "hashes",
    "ckpt": WORKSPACE / "ckpt",
}

# Create folders in workspace (and only workspace)
for name, p in SUBDIRS.items():
    p.mkdir(parents=True, exist_ok=True)

# IMPORTS
OPENS = None
PIL_Image = None
TORCH = None

def safe_imports():
    """Import optional deps gracefully; keep the notebook runnable."""
    global OPENS, PIL_Image, TORCH
    try:
        import openslide
        OPENS = openslide
    except Exception as e:
        print("[WARN] openslide-python not available:", e)
        OPENS = None
    try:
        from PIL import Image
        PIL_Image = Image
    except Exception as e:
        print("[WARN] Pillow (PIL) not available:", e)
        PIL_Image = None
    try:
        import torch
        TORCH = torch
    except Exception as e:
        print("[WARN] PyTorch not available:", e)
        TORCH = None

safe_imports()

# HELPERS
def now_iso() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def gb(nbytes: int) -> float:
    return round(nbytes / (1024**3), 2)

def safe_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Atomic-ish write for JSON (write .tmp then replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def list_wsi_files(root: Path):
    """Find slides by typical extensions (case-insensitive)."""
    if not root.exists():
        return []
    exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn")
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))

def open_and_probe_wsi(path: Path):
    """
    Return (width, height, mpp_x, mpp_y, vendor, thumb_path or None).
    Reads WSI read-only; writes thumbnail only under WORKSPACE/figures.
    """
    if OPENS is None:
        return None
    slide = OPENS.OpenSlide(str(path))
    props = slide.properties
    w, h = slide.dimensions
    mpp_x = props.get('openslide.mpp-x') or props.get('aperio.MPP') or None
    mpp_y = props.get('openslide.mpp-y') or props.get('aperio.MPP') or None
    vendor = props.get('openslide.vendor') or props.get('aperio.AppMag') or 'unknown'

    # Write a small thumbnail into WORKSPACE/figures (proof-of-life)
    thumb_path = None
    try:
        if hasattr(slide, "get_thumbnail") and PIL_Image is not None:
            max_side = 768
            scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
            tw, th = int(w/scale), int(h/scale)
            thumb = slide.get_thumbnail((tw, th))
            thumb_path = SUBDIRS["figures"] / f"sample_thumb_{path.stem}.jpg"
            thumb.save(str(thumb_path), "JPEG", quality=90)
    except Exception as e:
        print(f"[WARN] Could not write thumbnail for {path.name}: {e}")
        thumb_path = None
    finally:
        slide.close()

    return (w, h, mpp_x, mpp_y, vendor, thumb_path)

# ENV SUMMARY
def get_env_summary() -> Dict[str, Any]:
    info = {
        "timestamp": now_iso(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "workspace": str(WORKSPACE),
        "wsi_root": str(WSI_ROOT),
        "paths_note": "All writes occur ONLY under 'workspace'. 'wsi_root' is read-only.",
    }
    # Disk at workspace
    try:
        total, used, free = shutil.disk_usage(WORKSPACE)
        info.update({
            "disk_total_gb": gb(total),
            "disk_used_gb": gb(used),
            "disk_free_gb": gb(free),
        })
    except Exception as e:
        info["disk_error"] = str(e)

    # Torch / CUDA
    if TORCH is not None:
        info["torch_version"] = TORCH.__version__
        info["cuda_available"] = TORCH.cuda.is_available()
        if TORCH.cuda.is_available():
            try:
                dev = TORCH.cuda.current_device()
                prop = TORCH.cuda.get_device_properties(dev)
                info["cuda_device"] = {
                    "index": dev,
                    "name": prop.name,
                    "total_vram_gb": round(prop.total_memory / (1024**3), 2),
                    "multi_processor_count": getattr(prop, "multi_processor_count", None),
                }
                info["cuda_runtime_version"] = TORCH.version.cuda
                info["cudnn_version"] = TORCH.backends.cudnn.version()
            except Exception as e:
                info["cuda_error"] = str(e)
    else:
        info["torch_version"] = None

    # OpenSlide
    info["openslide_version"] = getattr(OPENS, "__version__", None) if OPENS else None
    return info

# RUNTIME START
print("== OP_FM Script 1: Environment & Compute-Passport ==")
print(f"[{now_iso()}] Workspace: {WORKSPACE}")
print(f"[{now_iso()}] WSI Root (read-only): {WSI_ROOT}")

if not WSI_ROOT.exists():
    print(f"[WARN] WSI_ROOT does not exist yet: {WSI_ROOT}")

# Save environment summary & compute-passport init
env = get_env_summary()

compute_passport = {
    "run_id": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    "created_at": now_iso(),
    "workspace": str(WORKSPACE),
    "wsi_root": str(WSI_ROOT),
    "environment": env,
    "stages": [],   # subsequent scripts will append stage entries here
}

compute_path = SUBDIRS["compute"] / "compute_passport.json"
safe_write_json(compute_path, compute_passport)
print(f"[OK] Compute-Passport initialized at: {compute_path}")

# SANITY: FIND & PROBE ONE WSI
wsi_files = list_wsi_files(WSI_ROOT)
print(f"[INFO] Detected {len(wsi_files)} WSI files in WSI_ROOT.")

sample_report: Dict[str, Any] = {}
if wsi_files and OPENS is not None:
    # deterministic sample (first sorted) for reproducibility
    sample_path = wsi_files[0]
    try:
        probe = open_and_probe_wsi(sample_path)
        if probe:
            w, h, mpp_x, mpp_y, vendor, thumb_path = probe
            sample_report = {
                "slide_path": str(sample_path),
                "width": w, "height": h,
                "mpp_x": mpp_x, "mpp_y": mpp_y,
                "vendor": vendor,
                "thumbnail": str(thumb_path) if thumb_path else None,
            }
            print("\n== Sample WSI Probe ==")
            print(" Path   :", sample_report["slide_path"])
            print(" Size   :", f"{w} x {h}")
            print(" MPP    :", f"x={mpp_x}  y={mpp_y}")
            print(" Vendor :", vendor)
            print(" Thumb  :", sample_report['thumbnail'] or "(not created)")
        else:
            print("[WARN] OpenSlide not available; skipping probe.")
    except Exception as e:
        print(f"[WARN] Could not open sample slide: {e}")
else:
    if not wsi_files:
        print("[INFO] No WSI files detected (check WSI_ROOT path).")
    if OPENS is None:
        print("[WARN] openslide-python missing; install:\n  pip install openslide-python\nand system OpenSlide libs.")

# LOG SUMMARY TO WORKSPACE
log = {
    "timestamp": now_iso(),
    "env": env,
    "sample_probe": sample_report,
}
log_path = SUBDIRS["logs"] / "env_summary.json"
safe_write_json(log_path, log)
print(f"[OK] Environment summary written to: {log_path}")

# Human-readable TXT for Methods appendix
txt_path = SUBDIRS["logs"] / "env_summary.txt"
with txt_path.open("w", encoding="utf-8") as f:
    f.write("OP_FM — Environment Summary\n")
    f.write(f"Timestamp: {now_iso()}\n\n")
    for k, v in env.items():
        f.write(f"{k}: {v}\n")
    if sample_report:
        f.write("\nSample WSI Probe:\n")
        for k, v in sample_report.items():
            f.write(f"  {k}: {v}\n")
print(f"[OK] Human-readable summary written to: {txt_path}")

# Diagnostics Checklist
print("\n== Diagnostics Checklist (Script 1) ==")
print(" - [", "OK" if OPENS else "!!", "] openslide-python import")
print(" - [", "OK" if PIL_Image else "!!", "] Pillow import")
if TORCH:
    cuda_line = f"CUDA={TORCH.cuda.is_available()}"
    dev_line = ""
    if TORCH.cuda.is_available():
        try:
            prop = TORCH.cuda.get_device_properties(0)
            dev_line = f" | GPU={prop.name} VRAM={round(prop.total_memory/(1024**3),2)} GB"
        except Exception:
            pass
    print(" - [ OK ] PyTorch", TORCH.__version__, cuda_line, dev_line)
else:
    print(" - [ !! ] PyTorch not available")
print(" - [ OK ] All outputs confined to:", WORKSPACE)
print(" - [ INFO ] Compute-Passport at:", compute_path)
print(" - [ INFO ] Logs at:", log_path, "and", txt_path)
print("\nScript 1 complete. Proceed to Script 2 (Manifest & Provenance).")


# SCRIPT 02: DATASET MANIFEST & PROVENANCE
# OP_FM — Script 2: Dataset Manifest & Provenance
import os, sys, json, time, math, hashlib, traceback, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional heavy deps (graceful if missing)
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise RuntimeError("Please install pandas and numpy (pip install pandas numpy)") from e

# Matplotlib only (no seaborn as per your rules)
import matplotlib
import matplotlib.pyplot as plt

# Reuse imports/vars from Script 1
assert 'WSI_ROOT' in globals() and 'WORKSPACE' in globals() and 'SUBDIRS' in globals(), \
    "Please run Script 1 first to define WSI_ROOT/WORKSPACE/SUBDIRS."

# Config knobs
MANIFEST_OUT = SUBDIRS["manifests"] / "manifest_tcga.parquet"
MANIFEST_CSV = SUBDIRS["manifests"] / "manifest_tcga.csv"
FAILED_CSV   = SUBDIRS["manifests"] / "failed_slides.csv"
HASH_INDEX   = SUBDIRS["hashes"]   / "hash_index_tcga.csv"

# Fast fingerprint mode: "size_only" | "sha1_quick" | "sha1_full"
# - "sha1_quick": hash first 8 MiB + last 8 MiB + size (fast & stable for dedup)
# - "sha1_full":  hash whole file (very slow on 20k WSIs)
# - "size_only":  just uses file size (weak dedup; fastest)
CHECKSUM_MODE = "sha1_quick"

# Concurrency (opening WSIs and reading small file regions in parallel)
MAX_WORKERS = min(12, (os.cpu_count() or 8))

# Utilities
def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def file_times(p: Path):
    st = p.stat()
    # Windows returns st_ctime as "creation" time
    created = datetime.datetime.fromtimestamp(getattr(st, "st_ctime", st.st_mtime)).strftime("%Y-%m-%d %H:%M:%S")
    modified = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return created, modified

def quick_fingerprint(path: Path, mode="sha1_quick", chunk=8*1024*1024):
    """Return (fingerprint, sha1_full_or_None)."""
    size = path.stat().st_size
    if mode == "size_only":
        return f"SIZE:{size}", None

    if mode == "sha1_quick":
        h = hashlib.sha1()
        with path.open("rb") as f:
            # first chunk
            h.update(f.read(chunk))
            # last chunk
            if size > chunk:
                f.seek(max(size - chunk, 0))
                h.update(f.read(chunk))
        h.update(str(size).encode("utf-8"))
        return f"QSHA1:{h.hexdigest()}", None

    if mode == "sha1_full":
        h = hashlib.sha1()
        with path.open("rb") as f:
            while True:
                b = f.read(1024*1024)
                if not b:
                    break
                h.update(b)
        return f"SHA1:{h.hexdigest()}", h.hexdigest()

    raise ValueError(f"Unknown CHECKSUM_MODE: {mode}")

def list_wsi_files(root: Path):
    exts = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn")
    out = []
    for ext in exts:
        out.extend(root.rglob(f"*{ext}"))
        out.extend(root.rglob(f"*{ext.upper()}"))
    # deduplicate
    return sorted(set(out))

def cancer_code_from_path(p: Path, root: Path):
    rel = p.relative_to(root)
    # Expect structure: <CANCER_CODE>/<filename>
    return rel.parts[0] if len(rel.parts) >= 2 else "UNKNOWN"

def open_and_probe(path: Path):
    """Open WSI with openslide to get dimensions + properties."""
    import openslide  # local import to isolate any import errors
    slide = openslide.OpenSlide(str(path))
    props = slide.properties
    width, height = slide.dimensions
    level_count = slide.level_count

    # Try to read common metadata keys
    mpp_x = props.get('openslide.mpp-x') or props.get('aperio.MPP') or None
    mpp_y = props.get('openslide.mpp-y') or props.get('aperio.MPP') or None
    vendor = props.get('openslide.vendor') or 'unknown'
    obj_pow = props.get('aperio.AppMag') or props.get('openslide.objective-power') or None

    slide.close()
    return {
        "width": int(width),
        "height": int(height),
        "level_count": int(level_count),
        "mpp_x": float(mpp_x) if mpp_x not in (None, "") else None,
        "mpp_y": float(mpp_y) if mpp_y not in (None, "") else None,
        "vendor": str(vendor),
        "objective_power": float(obj_pow) if (obj_pow is not None and str(obj_pow).replace('.','',1).isdigit()) else str(obj_pow) if obj_pow else None,
    }

# Scan & collect
start = time.time()
print(f"== OP_FM Script 2: Manifest & Provenance ==\n[{now_iso()}] Scanning WSI root (read-only): {WSI_ROOT}")

slides = list_wsi_files(WSI_ROOT)
n_total = len(slides)
print(f"[INFO] Found {n_total} candidate WSI files.")

records = []
failures = []

def process_one(path: Path):
    rec = {
        "path": str(path),
        "filename": path.name,
        "slide_id": path.stem,  # generic; downstream can parse TCGA ids if needed
        "cancer_code": cancer_code_from_path(path, WSI_ROOT),
        "size_bytes": path.stat().st_size,
    }
    # timestamps
    created, modified = file_times(path)
    rec["created_time"] = created
    rec["modified_time"] = modified

    # checksum / fingerprint
    try:
        fp, sha1_full = quick_fingerprint(path, mode=CHECKSUM_MODE)
        rec["fingerprint"] = fp
        rec["sha1_full"] = sha1_full
    except Exception as e:
        rec["fingerprint"] = None
        rec["sha1_full"] = None

    # attempt to open and read properties
    try:
        meta = open_and_probe(path)
        rec.update(meta)
        rec["error"] = None
    except Exception as e:
        rec.update({
            "width": None, "height": None, "level_count": None,
            "mpp_x": None, "mpp_y": None, "vendor": None, "objective_power": None,
            "error": f"{e.__class__.__name__}: {e}"
        })
    return rec

# Parallel pass
t0 = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(process_one, p): p for p in slides}
    done = 0
    last_print = t0
    for fut in as_completed(futures):
        rec = fut.result()
        records.append(rec)
        if rec.get("error"):
            failures.append({"path": rec["path"], "error": rec["error"]})
        done += 1
        # light progress print every ~2 seconds
        now = time.time()
        if now - last_print > 2 or done == n_total:
            rate = done / (now - t0 + 1e-9)
            print(f"[SCAN] {done}/{n_total} ({rate:.1f} files/s)")
            last_print = now

elapsed_scan = time.time() - start
print(f"[OK] Scanned {n_total} slides in {elapsed_scan/60:.1f} min.")

# DataFrame & save
df = pd.DataFrame.from_records(records)

# Ensure consistent types
num_cols = ["size_bytes", "width", "height", "level_count", "mpp_x", "mpp_y"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Save manifest (Parquet + CSV)
SUBDIRS["manifests"].mkdir(parents=True, exist_ok=True)
df.to_parquet(MANIFEST_OUT, index=False)
df.to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Manifest saved:\n - {MANIFEST_OUT}\n - {MANIFEST_CSV}")

# Save failures (if any)
if failures:
    pd.DataFrame(failures).to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")
    print(f"[WARN] {len(failures)} slides failed to open; see {FAILED_CSV}")

# Save a light hash index (path, size, fingerprint) for quick dedup debugging
pd.DataFrame(df[["path", "size_bytes", "fingerprint"]]).to_csv(HASH_INDEX, index=False, encoding="utf-8-sig")
print(f"[OK] Hash index written: {HASH_INDEX}")

# Diagnostics (print)
print("\n== Diagnostics (Manifest) ==")
total_bytes = df["size_bytes"].sum(skipna=True)
print(f" Total slides: {len(df):,}")
print(f" Total size  : {total_bytes / (1024**3):.2f} GB")
by_cancer = df["cancer_code"].value_counts(dropna=False)
print("\n Slides by cancer_code (top 20):")
print(by_cancer.head(20).to_string())

missing_mpp = df[(df["mpp_x"].isna()) | (df["mpp_y"].isna())]
print(f"\n Missing MPP entries: {len(missing_mpp)}")

# Duplicate detection (by fingerprint if available, else by size+filename)
dup_key = "fingerprint" if df["fingerprint"].notna().any() else None
if dup_key:
    dup_groups = df.groupby(dup_key).size().sort_values(ascending=False)
    dup_groups = dup_groups[dup_groups > 1]
    print(f"\n Potential duplicates by {dup_key}: {int(dup_groups.sum()) - len(dup_groups)} extra files in {len(dup_groups)} groups")
else:
    size_groups = df.groupby(["size_bytes", "filename"]).size().sort_values(ascending=False)
    size_groups = size_groups[size_groups > 1]
    print(f"\n Potential duplicates by (size, filename): {int(size_groups.sum()) - len(size_groups)} extra files in {len(size_groups)} groups")

# Top-N largest slides
topN = df.sort_values("size_bytes", ascending=False).head(10)[["filename", "cancer_code", "size_bytes"]].copy()
topN["size_gb"] = topN["size_bytes"] / (1024**3)
print("\n Top-10 largest WSIs (GB):")
print(topN[["filename", "cancer_code", "size_gb"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Diagnostics (plots to file)
fig_dir = SUBDIRS["figures"]
fig_dir.mkdir(parents=True, exist_ok=True)

# 1) Size distribution (GB)
plt.figure(figsize=(8,5))
sizes_gb = (df["size_bytes"] / (1024**3)).dropna()
plt.hist(sizes_gb.values, bins=40)
plt.xlabel("Slide size (GB)")
plt.ylabel("Count")
plt.title("WSI Size Distribution (TCGA)")
plt.tight_layout()
p1 = fig_dir / "manifest_size_distribution.png"
plt.savefig(p1)
plt.close()
print(f"[FIG] {p1}")

# 2) Width/Height distributions (log10)
plt.figure(figsize=(8,5))
wh = df[["width", "height"]].dropna()
vals = np.log10(wh.values.clip(min=1))
plt.hist(vals.flatten(), bins=40)
plt.xlabel("log10(pixels)")
plt.ylabel("Count")
plt.title("WSI Width/Height Distribution (log10)")
plt.tight_layout()
p2 = fig_dir / "manifest_wh_log_distribution.png"
plt.savefig(p2)
plt.close()
print(f"[FIG] {p2}")

# 3) Slides by cancer_code (bar, top 30)
plt.figure(figsize=(10,6))
top_codes = by_cancer.head(30)
plt.bar(top_codes.index.astype(str), top_codes.values)
plt.xticks(rotation=80, ha="right")
plt.ylabel("Slides")
plt.title("Slides per cancer_code (Top 30)")
plt.tight_layout()
p3 = fig_dir / "manifest_counts_by_cancer.png"
plt.savefig(p3)
plt.close()
print(f"[FIG] {p3}")

# 4) MPP completeness (% with both mpp_x & mpp_y)
mpp_complete = df["mpp_x"].notna() & df["mpp_y"].notna()
pct_mpp = 100.0 * mpp_complete.mean()
plt.figure(figsize=(4,4))
plt.bar(["MPP complete", "MPP missing"], [pct_mpp, 100.0 - pct_mpp])
plt.title("MPP Availability (%)")
plt.tight_layout()
p4 = fig_dir / "manifest_mpp_availability.png"
plt.savefig(p4)
plt.close()
print(f"[FIG] {p4} (MPP complete: {pct_mpp:.1f}%)")

# Append compute-passport
compute_path = SUBDIRS["compute"] / "compute_passport.json"
try:
    with compute_path.open("r", encoding="utf-8") as f:
        cp = json.load(f)
except Exception:
    cp = {"stages": []}

stage_entry = {
    "stage": "manifest_tcga",
    "timestamp": now_iso(),
    "inputs": {"wsi_root": str(WSI_ROOT)},
    "outputs": {
        "manifest_parquet": str(MANIFEST_OUT),
        "manifest_csv": str(MANIFEST_CSV),
        "failed_csv": str(FAILED_CSV) if failures else None,
        "hash_index_csv": str(HASH_INDEX),
        "figures": [str(p1), str(p2), str(p3), str(p4)],
    },
    "stats": {
        "n_files_found": int(n_total),
        "n_records": int(len(df)),
        "n_failures": int(len(failures)),
        "total_gb": float(total_bytes / (1024**3)),
        "elapsed_minutes": float(elapsed_scan / 60.0),
        "checksum_mode": CHECKSUM_MODE,
    }
}
cp.setdefault("stages", []).append(stage_entry)

# Write back atomically
tmp = compute_path.with_suffix(".json.tmp")
with tmp.open("w", encoding="utf-8") as f:
    json.dump(cp, f, ensure_ascii=False, indent=2)
tmp.replace(compute_path)
print(f"\n[OK] Compute-Passport updated: {compute_path}")

print("\nScript 2 complete. Next: Script 3 (QC & Tissue Mask).")


# SCRIPT 03: QUALITY CONTROL & TISSUE MASKING
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

# Inputs & Outputs
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

# Config knobs
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

# Helpers
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

# Load manifest
print(f"== OP_FM Script 3: QC & Tissue Mask ==\n[{now_iso()}] Loading manifest:", MANIFEST_PARQUET)
df_manifest = pd.read_parquet(MANIFEST_PARQUET)
df_manifest = df_manifest.copy()

if QC_MAX_SLIDES is not None:
    df_manifest = df_manifest.head(QC_MAX_SLIDES).copy()
print(f"[INFO] Slides to QC: {len(df_manifest)}")

# Run QC (multi-thread)
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

# Save metrics & exclusions
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

# Diagnostics & Figures
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

# Update compute-passport
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


# SCRIPT 04: TWO-SCALE TILING & TOKEN BUDGET
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

# Prereqs
assert 'WSI_ROOT' in globals() and 'WORKSPACE' in globals() and 'SUBDIRS' in globals(), \
    "Please run Script 1 first to define WSI_ROOT/WORKSPACE/SUBDIRS."

MANIFEST_PARQUET = SUBDIRS["manifests"] / "manifest_tcga.parquet"
QC_METRICS_PARQUET = SUBDIRS["qc"] / "qc_metrics_tcga.parquet"
TILES_DIR = SUBDIRS["tiles"] / "manifests"
TILES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = SUBDIRS["figures"]

# Config
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

# Helpers
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

# Load inputs
print(f"== OP_FM Script 4: Two-scale Tiling & Token Budget ==\n[{now_iso()}] Loading:", MANIFEST_PARQUET)
df_manifest = pd.read_parquet(MANIFEST_PARQUET)

# Try to load QC metrics; optional
df_qc = None
if QC_POLICY in ("strict", "medium") and QC_METRICS_PARQUET.exists():
    df_qc = pd.read_parquet(QC_METRICS_PARQUET)
    df_qc = df_qc[["slide_id", "tissue_pct", "white_pct", "pen_pct", "reasons"]].copy()

# Select slides
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

# Per-slide worker
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

# Run (multi-slide)
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

# Summaries & Figures
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

# Update compute-passport
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


# SCRIPT 05: FROZEN-BACKBONE FEATURE EXTRACTION (CONVNEXT-TINY)
# Script 5 — OpenSlide extractor
import os, sys, json, time, math, random, shutil, subprocess, platform, gc
from pathlib import Path
from datetime import datetime
from time import perf_counter

# Paths (strict)
# WORKSPACE — set in CONFIGURATION block above
# WSI_ROOT — set in CONFIGURATION block above
SUBDIRS = {
    "features": WORKSPACE / "features",
    "tiles":    WORKSPACE / "tiles",
    "logs":     WORKSPACE / "logs",
    "figures":  WORKSPACE / "figures",
}
for p in SUBDIRS.values(): p.mkdir(parents=True, exist_ok=True)

TSUM = SUBDIRS["tiles"] / "tiling_summary_tcga.parquet"
assert TSUM.exists(), f"Missing tiling summary: {TSUM}"

# Quiet-install deps (no admin)
def ensure(pkg): 
    try:
        __import__(pkg.split('[')[0].replace('-','_').split('==')[0])
    except Exception:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

ensure("openslide_python")
ensure("openslide_bin")           # provides DLLs on Windows
ensure("torch>=2.1")
ensure("torchvision")
ensure("pandas")
ensure("pyarrow")
ensure("Pillow")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
import openslide

# Config
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16 if DEVICE=="cuda" else torch.bfloat16
TILE_SIZE = 256       # manifest tile size (Script 4 default)
MODEL_IN  = 224
SELFTEST_SECONDS = 60
TARGET_TILES_PER_SEC = 50.0       # <-- your target gate
RANDOM_SEED = 13
SAVE_DTYPE = np.float16

random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if hasattr(torch.backends,"cudnn"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch,"set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

# Transforms
IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
_to_tensor = T.ToTensor()
_resize    = T.Resize((MODEL_IN, MODEL_IN), interpolation=T.InterpolationMode.BILINEAR)
_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
def to_model_tensor(img: Image.Image) -> torch.Tensor:
    if img.size != (MODEL_IN, MODEL_IN):
        img = _resize(img)
    t = _to_tensor(img); t = _normalize(t)
    return t

# Model (ConvNeXt-Tiny → 768D)
class ConvNeXtTinyFeats(nn.Module):
    def __init__(self):
        super().__init__()
        w = tvm.ConvNeXt_Tiny_Weights.DEFAULT
        m = tvm.convnext_tiny(weights=w)
        self.features = m.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters(): p.requires_grad=False
        self.eval()
    @torch.no_grad()
    def forward(self, x):              # [N,3,224,224]
        x = self.features(x)           # [N,768,H,W]
        x = self.gap(x).flatten(1)     # [N,768]
        return x

def build_model():
    m = ConvNeXtTinyFeats().to(DEVICE)
    if DEVICE=="cuda":
        m = m.to(memory_format=torch.channels_last)
        # short warmup
        d = torch.randn(256,3,MODEL_IN,MODEL_IN, device=DEVICE).to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True):
            _ = m(d)
        torch.cuda.synchronize()
    return m

MODEL = build_model()

# Tiling summary & manifest helpers
df_sum = pd.read_parquet(TSUM)   # columns include: slide_id, manifest, scale_um_per_px, n_tiles, ...
assert "manifest" in df_sum.columns and "slide_id" in df_sum.columns

SLIDE_INDEX_PATH = SUBDIRS["logs"] / "slide_path_index.json"
def index_slide_paths(root: Path) -> dict:
    print("[INDEX] Building slide path map (once) ...")
    mp={}
    for ext in ("*.svs","*.ndpi","*.tif","*.mrxs","*.scn"):
        for p in root.rglob(ext):
            mp[p.stem] = str(p)
    return mp
if SLIDE_INDEX_PATH.exists():
    slide_map = json.loads(SLIDE_INDEX_PATH.read_text(encoding="utf-8"))
else:
    slide_map = index_slide_paths(WSI_ROOT)
    SLIDE_INDEX_PATH.write_text(json.dumps(slide_map, indent=2), encoding="utf-8")

def slide_path_from_id(slide_id:str, manifest_df:pd.DataFrame|None=None):
    # Prefer manifest-sourced path if present
    if manifest_df is not None:
        for cand in ("path","source_path","slide_path","wsi_path"):
            if cand in manifest_df.columns:
                p = manifest_df[cand].iloc[0]
                if isinstance(p,str) and Path(p).exists():
                    return p
    # Fallback to index
    if slide_id in slide_map: return slide_map[slide_id]
    base = slide_id.split(".")[0]
    return slide_map.get(base, None)

def load_manifest(man_path:Path):
    m = pd.read_parquet(man_path)
    # column normalization
    lower = {c.lower():c for c in m.columns}
    def pick(*names):
        for n in names:
            if n in m.columns: return n
            if n.lower() in lower: return lower[n.lower()]
        raise KeyError(f"Missing columns {names} in {man_path.name}")
    xcol   = pick("x","px_x","x_level")
    ycol   = pick("y","px_y","y_level")
    lvlcol = pick("level","lvl")
    # tile size if present
    tsize = TILE_SIZE
    for n in ("tile_size","tile_px","size"):
        if n in m.columns:
            try: tsize = int(m[n].iloc[0])
            except: pass
            break
    return m, xcol, ycol, lvlcol, tsize

# OpenSlide reader (level coords → level-0 coords)
class SlideReader:
    def __init__(self, path:str):
        self.path = path
        self.osr  = openslide.OpenSlide(path)
        self.down = list(self.osr.level_downsamples)  # float
    def read_tile(self, level:int, x_level:int, y_level:int, size:int):
        # convert level coords to level-0 pixels
        ds = self.down[level]
        bx = int(round(x_level * ds))
        by = int(round(y_level * ds))
        img = self.osr.read_region((bx,by), level, (size,size)).convert("RGB")
        return img
    def close(self):
        try: self.osr.close()
        except: pass

# Batching & forward
def iter_batches_from_manifest(reader:SlideReader, man_df, xcol, ycol, lvlcol, tile_px, max_batch=4096):
    # Serial, contiguous batches (HDD-friendly), no multiprocessing
    buf=[]
    for r in man_df[[xcol,ycol,lvlcol]].itertuples(index=False, name=None):
        x,y,lvl = map(int, r)
        img = reader.read_tile(lvl, x, y, tile_px)
        t = to_model_tensor(img)                    # [3,H,W]
        buf.append(t)
        if len(buf) >= max_batch:
            batch = torch.stack(buf,0).to(memory_format=torch.channels_last)
            yield batch
            buf.clear()
    if buf:
        batch = torch.stack(buf,0).to(memory_format=torch.channels_last)
        yield batch

def forward_batches(model, batches_iter):
    outs=[]
    for cpu_batch in batches_iter:
        with torch.no_grad():
            chunk = cpu_batch.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE=="cuda")):
                out = model(chunk)                 # [N,768]
            outs.append(out.detach().cpu())
        del cpu_batch
    feats = torch.cat(outs,0).contiguous().numpy()
    return feats

# Output paths
OUT05 = SUBDIRS["features"] / "scale0p5"; OUT20 = SUBDIRS["features"] / "scale2p0"
OUT05.mkdir(parents=True, exist_ok=True); OUT20.mkdir(parents=True, exist_ok=True)
def out_paths(slide_id:str, scale:float, ext="npy"):
    d = OUT05 if math.isclose(scale,0.5,abs_tol=1e-6) else OUT20
    return d / f"{slide_id}.{ext}", d / f"{slide_id}_meta.parquet"

# Env print
env = {
    "time": datetime.now().isoformat(timespec="seconds"),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "device": DEVICE,
    "torch": torch.__version__,
    "amp_dtype": str(AMP_DTYPE),
    "openslide_vendor": None
}
try:
    # peek one slide to get vendor
    ptest = next(iter(slide_map.values()), None)
    if ptest:
        osr = openslide.OpenSlide(ptest); env["openslide_vendor"] = osr.properties.get("openslide.vendor","?"); osr.close()
except Exception: pass
(SUBDIRS["logs"] / "script5_env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
print("[ENV]\n" + json.dumps(env, indent=2))

# Build pending groups (both scales per slide)
done_map={}
for sc in (0.5, 2.0):
    sub = df_sum[np.isclose(df_sum["scale_um_per_px"], sc)]
    for sid in sub["slide_id"].unique():
        npy, meta = out_paths(sid, sc)
        done_map[(sid, sc)] = npy.exists() and meta.exists()

groups=[]
for sid, g in df_sum.groupby("slide_id", sort=False):
    entries=[]
    for _, row in g.sort_values("n_tiles",ascending=False).iterrows():
        sc = float(row["scale_um_per_px"])
        if not done_map.get((sid, sc), False):
            entries.append({"scale": sc, "manifest": Path(row["manifest"])})
    if entries:
        groups.append({"slide_id": sid, "entries": entries})
print(f"[INFO] Slides pending (≥1 scale): {len(groups)}")

# Self-test (60 s)
def selftest(seconds=SELFTEST_SECONDS, target=TARGET_TILES_PER_SEC):
    # pick smallest total tiles to minimize seek overhead during test
    cand=[]
    for sid, g in df_sum.groupby("slide_id"):
        n=int(g["n_tiles"].sum()); man=Path(g.sort_values("n_tiles").iloc[-1]["manifest"])
        cand.append((n, sid, man))
    cand.sort(key=lambda x:x[0])
    pick = cand[:min(12, len(cand))]

    # open readers once
    readers={}
    for _, sid, manp in pick:
        m, xcol, ycol, lvlcol, tpx = load_manifest(manp)
        fn = slide_path_from_id(sid, m)
        if not fn or not Path(fn).exists(): continue
        readers[sid] = (SlideReader(fn), m[[xcol,ycol,lvlcol]].copy(), xcol, ycol, lvlcol, tpx)

    tiles_done=0; t0=perf_counter(); stop=t0+seconds
    while perf_counter()<stop and readers:
        for sid,(sr, m, xcol,ycol,lvlcol,tpx) in list(readers.items()):
            # take ~512 tiles per sid per turn
            take = m.iloc[:512]
            if take.empty:
                del readers[sid]; sr.close(); continue
            batches = iter_batches_from_manifest(sr, take, xcol,ycol,lvlcol, tpx, max_batch=2048)
            with torch.no_grad():
                for cpu_batch in batches:
                    chunk = cpu_batch.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE=="cuda")):
                        _ = MODEL(chunk)
                    tiles_done += chunk.size(0)
                    del cpu_batch, chunk
                    if perf_counter()>=stop: break
            m = m.iloc[len(take):]
            readers[sid]=(sr, m, xcol,ycol,lvlcol,tpx)
            if perf_counter()>=stop: break

    dt = perf_counter()-t0
    rate = tiles_done / max(dt,1e-6)
    print(f"[SELFTEST] tiles={tiles_done}  time={dt:.1f}s  tiles/s={rate:.1f}")
    print(("[PASS] " if rate>=target else "[FAIL] ")+f"{rate:.1f} tiles/s (target ≥ {target:.0f})")
    (SUBDIRS["logs"]/ "script5_selftest.json").write_text(json.dumps({
        "time": datetime.now().isoformat(timespec="seconds"),
        "tiles": tiles_done, "seconds": round(dt,2), "tiles_per_s": round(rate,2),
        "target": target, "pass": rate>=target
    }, indent=2), encoding="utf-8")
    # close readers
    for sid,(sr, *_rest) in readers.items(): sr.close()
    return rate

rate = selftest()
if rate < TARGET_TILES_PER_SEC:
    print("[ABORT] Below target. This cell stops here (no full run).")
    raise SystemExit(0)

# Full run (only if self-test passed)
PROG = SUBDIRS["logs"] / "script5_progress.jsonl"
def log_progress(**kw):
    kw["ts"]=datetime.now().isoformat(timespec="seconds")
    with open(PROG,"a",encoding="utf-8") as f: f.write(json.dumps(kw,ensure_ascii=False)+"\n")

for i, grp in enumerate(groups, 1):
    sid = grp["slide_id"]
    # open once per slide
    # load one manifest to discover path (prefer scale 0.5 if exists)
    man_pref = min(grp["entries"], key=lambda e: abs(e["scale"]-0.5))
    m_probe, xcol, ycol, lvlcol, tpx = load_manifest(man_pref["manifest"])
    fn = slide_path_from_id(sid, m_probe)
    if not fn or not Path(fn).exists():
        print(f"[WARN] slide path not found: {sid} — skipped")
        continue
    reader = SlideReader(fn)

    for e in grp["entries"]:
        sc = float(e["scale"])
        npy_path, meta_path = out_paths(sid, sc)
        if npy_path.exists() and meta_path.exists(): 
            continue

        man_df, xcol, ycol, lvlcol, tpx = load_manifest(e["manifest"])
        if man_df.empty:
            print(f"[WARN] empty manifest: {e['manifest']} — skip")
            continue

        # forward
        t0 = perf_counter()
        batches = iter_batches_from_manifest(reader, man_df, xcol,ycol,lvlcol, tpx, max_batch=4096)
        feats = forward_batches(MODEL, batches)         # [N,768]
        if DEVICE=="cuda": torch.cuda.synchronize()
        dt = perf_counter()-t0

        # save
        np.save(npy_path, feats.astype(SAVE_DTYPE))
        md = man_df.copy()
        md["slide_id"]=sid; md["scale_um_per_px"]=sc
        md.to_parquet(meta_path, index=False)

        N = int(feats.shape[0])
        tiles_per_s = N / max(dt,1e-6)
        vram = (torch.cuda.max_memory_allocated()/(1024**3)) if DEVICE=="cuda" else 0.0
        print(f"[OK] {i}/{len(groups)} | {sid} @{sc:.1f} µm/px → ({N},768) | {tiles_per_s:.1f} tiles/s | VRAM~{vram:.2f} GB")
        log_progress(slide_id=sid, scale=sc, tiles=N, seconds=round(dt,2), tps=round(tiles_per_s,2), vram_gb=round(vram,2))

        del feats; gc.collect()
        if DEVICE=="cuda": torch.cuda.empty_cache()

    reader.close()

print("[DONE] All pending entries processed.")


# SCRIPT 06: SELF-SUPERVISED PRE-TRAINING (BYOL + MFR)
# Script 6 — Two-Scale Feature-Space Pretraining
import os, sys, json, math, random, gc, subprocess, platform
from pathlib import Path
from time import perf_counter
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

# Workspace
# WORKSPACE — set in CONFIGURATION block above
FEATURES05 = WORKSPACE / "features" / "scale0p5"
FEATURES20 = WORKSPACE / "features" / "scale2p0"
LOGS       = WORKSPACE / "logs"
WEIGHTS    = WORKSPACE / "weights"
FIGS       = WORKSPACE / "figures"
EMBED      = WORKSPACE / "embeddings" / "student_final"
for p in [LOGS, WEIGHTS, FIGS, EMBED]:
    p.mkdir(parents=True, exist_ok=True)
    assert str(p).startswith(str(WORKSPACE)), f"Output path escapes WORKSPACE: {p}"

# Robust deps (no hard failures for optional libs)
def _pip(*pkgs):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    except Exception as e:
        print(f"[WARN] pip install failed for {pkgs}: {e}")

try:
    import numpy as np
    import pandas as pd
except Exception:
    _pip("numpy>=1.24","pandas>=2.0"); import numpy as np, pandas as pd

try:
    import torch, torch.nn as nn, torch.nn.functional as F
except Exception:
    _pip("torch>=2.1"); import torch, torch.nn as nn, torch.nn.functional as F

try:
    from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
except Exception:
    _pip("safetensors>=0.4.0"); from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors

# Matplotlib is optional; plotting will be skipped if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# Config
CONFIG = {
    "seed": 13,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype_amp": "float16",                 # "float16" on CUDA; "float32" on CPU
    "token_budget_0p5": 1200,               # tokens from 0.5 μm per slide
    "token_budget_2p0":  400,               # tokens from 2.0 μm per slide
    "mask_frac": 0.25,                      # fraction of tokens masked for MFR
    "lambda_mfr": 0.5,                      # weight for MFR loss
    "d_model": 768,
    "n_heads": 8,
    "n_layers": 6,
    "ff_mult": 4,
    "dropout": 0.1,
    "batch_slides": 3,                      # fits 24 GB with defaults
    "grad_accum": 2,                        # effective batch = batch_slides * grad_accum
    "epochs": 4,
    "steps_per_epoch_cap": None,            # None = full pass; or int to cap
    "lr": 1.5e-4,
    "weight_decay": 1e-4,
    "ema_tau": 0.996,
    "warmup_steps": 500,
    "save_every_steps": 1000,
    "log_every_steps": 50,
    "resume_if_available": True,            # resume from weights/latest.txt if present
    "export_embeddings_after_train": True,  # export per-slide g-embeddings after training
    "export_use_budget": True               # True: budgets; False: all tokens (slower)
}

# Reproducibility
SEED = CONFIG["seed"]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if hasattr(torch.backends,"cudnn"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DEVICE = CONFIG["device"]
AMP_DTYPE = (torch.float16 if (DEVICE=="cuda" and CONFIG["dtype_amp"]=="float16") else
             torch.bfloat16 if (DEVICE=="cuda" and CONFIG["dtype_amp"]=="bfloat16") else
             torch.float32)

# Slide inventory (require both scales)
def _collect(dir_path: Path) -> Dict[str, Path]:
    mp = {}
    for p in dir_path.glob("*.npy"):
        mp[p.stem] = p
    return mp

mp05 = _collect(FEATURES05)
mp20 = _collect(FEATURES20)
common_ids = sorted(set(mp05.keys()) & set(mp20.keys()))
assert len(common_ids)>0, "No slides found that have both 0.5 and 2.0 μm features. Check Script 5 outputs."

@dataclass
class SlideRec:
    slide_id: str
    npy05: Path
    meta05: Path
    npy20: Path
    meta20: Path

def meta_path(npy_path: Path) -> Path:
    return npy_path.with_name(npy_path.stem + "_meta.parquet")

slides: List[SlideRec] = []
for sid in common_ids:
    p05 = mp05[sid]; p20 = mp20[sid]
    m05 = meta_path(p05); m20 = meta_path(p20)
    if m05.exists() and m20.exists():
        slides.append(SlideRec(sid, p05, m05, p20, m20))
assert len(slides)>0, "Found slides but *_meta.parquet files are missing. Re-run Script 5 or verify meta files."

print(json.dumps({
    "time": datetime.now().isoformat(timespec="seconds"),
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "device": DEVICE,
    "amp_dtype": str(AMP_DTYPE).split(".")[-1],
    "slides_2scale": len(slides)
}, indent=2))

# Meta loading (robust to column names)
_META_CACHE: Dict[Path, pd.DataFrame] = {}
def load_meta(p: Path) -> pd.DataFrame:
    if p in _META_CACHE: return _META_CACHE[p]
    df = pd.read_parquet(p)  # Script 5 produced pyarrow-style parquet
    # normalize columns
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in cols_lower: return cols_lower[n.lower()]
        raise KeyError(f"Missing one of {names} in {p.name}")
    xcol   = pick("x")
    ycol   = pick("y")
    lvlcol = pick("level","lvl")
    sccol  = pick("scale_um_per_px")
    tsize = 256
    for n in ("tile_size","tile_px","size"):
        if n in df.columns:
            try: tsize = int(df[n].iloc[0])
            except: pass
            break
    out = df[[xcol,ycol,lvlcol,sccol]].copy()
    out.columns = ["x","y","level","scale_um_per_px"]
    out["tile_px"] = tsize
    _META_CACHE[p] = out
    return out

def compute_mm_xy(df: pd.DataFrame) -> np.ndarray:
    um_per_px = df["scale_um_per_px"].astype(float).to_numpy()
    mm_per_px = um_per_px / 1000.0
    cx = (df["x"].to_numpy() + df["tile_px"].to_numpy()/2.0) * mm_per_px
    cy = (df["y"].to_numpy() + df["tile_px"].to_numpy()/2.0) * mm_per_px
    return np.stack([cx, cy], axis=1).astype(np.float32)

# MIL model
class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_model)
        )
    def forward(self, mmxy: torch.Tensor, scale_um: torch.Tensor):
        x = torch.cat([mmxy, scale_um], dim=-1)  # [B,T,3]
        return self.proj(x)

class MILTransformer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_layers=6, ff_mult=4, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=int(ff_mult*d_model),
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln  = nn.LayerNorm(d_model)
        self.pos = PositionalEncoder(d_model)
        self.proj_global = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.proj_token  = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pred_global = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pred_token  = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, feats: torch.Tensor, mmxy: torch.Tensor, scale_um: torch.Tensor, pad_mask: torch.Tensor):
        """
        feats   : [B,T,768]
        mmxy    : [B,T,2]
        scale_um: [B,T,1]
        pad_mask: [B,T] (True for PADs)
        """
        B,T,_ = feats.shape
        pos = self.pos(mmxy, scale_um)
        x = feats + pos
        cls = self.cls.expand(B,1,-1)
        x = torch.cat([cls, x], dim=1)  # [B,1+T,D]
        pad = torch.zeros(B,1, dtype=torch.bool, device=pad_mask.device)
        key_padding = torch.cat([pad, pad_mask], dim=1)
        x = self.enc(x, src_key_padding_mask=key_padding)
        x = self.ln(x)
        g = x[:,0,:]
        t = x[:,1:,:]
        g_proj = self.proj_global(g)
        t_proj = self.proj_token(t)
        g_pred = self.pred_global(g_proj)
        t_pred = self.pred_token(t_proj)
        return g_proj, t_proj, g_pred, t_pred

# Losses & EMA
def cosine_loss(p: torch.Tensor, z: torch.Tensor):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    return (1.0 - (p * z).sum(dim=-1)).mean()

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=(1.0 - tau))

# Build models/opt
student = MILTransformer(
    d_model=CONFIG["d_model"], n_heads=CONFIG["n_heads"],
    n_layers=CONFIG["n_layers"], ff_mult=CONFIG["ff_mult"], dropout=CONFIG["dropout"]
).to(DEVICE)

teacher = MILTransformer(
    d_model=CONFIG["d_model"], n_heads=CONFIG["n_heads"],
    n_layers=CONFIG["n_layers"], ff_mult=CONFIG["ff_mult"], dropout=CONFIG["dropout"]
).to(DEVICE)
teacher.load_state_dict(student.state_dict())
for p in teacher.parameters(): p.requires_grad = False

opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

# Token sampling & batching
def _sample(n: int, k: int) -> np.ndarray:
    if n <= k: return np.arange(n, dtype=np.int64)
    return np.random.choice(n, size=k, replace=False).astype(np.int64)

def load_tokens_for_slide(rec: SlideRec, budget05: int, budget20: int):
    """Return (feats[T,768], mmxy[T,2], scl[T,1]) with T = budget05 + budget20."""
    # 0.5 μm
    f05 = np.load(rec.npy05, mmap_mode='r')                     # [N05,768]
    assert f05.shape[1] == CONFIG["d_model"], f"Feature dim {f05.shape[1]}≠{CONFIG['d_model']} for {rec.slide_id} @0.5"
    m05 = load_meta(rec.meta05)
    idx05 = _sample(f05.shape[0], budget05)
    mm05  = compute_mm_xy(m05.iloc[idx05])                      # [budget05,2]
    sc05  = m05["scale_um_per_px"].iloc[idx05].to_numpy(np.float32).reshape(-1,1)

    # 2.0 μm
    f20 = np.load(rec.npy20, mmap_mode='r')                     # [N20,768]
    assert f20.shape[1] == CONFIG["d_model"], f"Feature dim {f20.shape[1]}≠{CONFIG['d_model']} for {rec.slide_id} @2.0"
    m20 = load_meta(rec.meta20)
    idx20 = _sample(f20.shape[0], budget20)
    mm20  = compute_mm_xy(m20.iloc[idx20])                      # [budget20,2]
    sc20  = m20["scale_um_per_px"].iloc[idx20].to_numpy(np.float32).reshape(-1,1)

    feats = np.concatenate([f05[idx05], f20[idx20]], axis=0).astype(np.float32)  # [T,768]
    mmxy  = np.concatenate([mm05, mm20], axis=0).astype(np.float32)              # [T,2]
    scl   = np.concatenate([sc05, sc20], axis=0).astype(np.float32)              # [T,1]
    return feats, mmxy, scl

def make_batch(batch_recs: List[SlideRec], budget05: int, budget20: int, mask_frac: float):
    feats_list=[]; mmxy_list=[]; sc_list=[]; mask_tiles=[]
    for rec in batch_recs:
        f, mm, sc = load_tokens_for_slide(rec, budget05, budget20)
        Tn = f.shape[0]
        feats_list.append(torch.from_numpy(f))
        mmxy_list.append(torch.from_numpy(mm))
        sc_list.append(torch.from_numpy(sc))
        mcount = max(1, int(round(mask_frac*Tn)))
        mask_idx = np.random.choice(Tn, size=mcount, replace=False).astype(np.int64)
        mask_tiles.append(torch.from_numpy(mask_idx))

    T = max(t.shape[0] for t in feats_list)
    B = len(batch_recs); D = feats_list[0].shape[1]
    feats = torch.zeros(B, T, D, dtype=torch.float32)
    mmxy  = torch.zeros(B, T, 2, dtype=torch.float32)
    scl   = torch.zeros(B, T, 1, dtype=torch.float32)
    pad   = torch.ones(B, T, dtype=torch.bool)
    for i in range(B):
        n = feats_list[i].shape[0]
        feats[i,:n] = feats_list[i]
        mmxy[i,:n]  = mmxy_list[i]
        scl[i,:n]   = sc_list[i]
        pad[i,:n]   = False

    mfr_index = []
    for b, idx in enumerate(mask_tiles):
        mfr_index.append(torch.stack([torch.full_like(idx, b), idx], dim=1))
    mfr_index = torch.cat(mfr_index, dim=0)  # [M,2]

    return {
        "feats": feats.to(DEVICE, non_blocking=True),
        "mmxy":  mmxy.to(DEVICE, non_blocking=True),
        "scl":   scl.to(DEVICE, non_blocking=True),
        "pad":   pad.to(DEVICE, non_blocking=True),
        "mfr_index": mfr_index.to(DEVICE, non_blocking=True)
    }

# Cosine scheduler w/ warmup
class CosineWarmup:
    def __init__(self, optimizer, warmup, max_steps, base_lr):
        self.opt=optimizer; self.warmup=warmup; self.max=max_steps; self.base=base_lr; self.t=0
    def step(self):
        self.t += 1
        if self.t <= self.warmup:
            lr = self.base * self.t / max(1,self.warmup)
        else:
            p = (self.t - self.warmup) / max(1, self.max - self.warmup)
            lr = self.base * 0.5*(1+math.cos(math.pi*p))
        for g in self.opt.param_groups: g["lr"]=lr
        return lr

# Logging & checkpoints
LOG_CSV = LOGS / "script6_train_log.csv"
if not LOG_CSV.exists():
    LOG_CSV.write_text("ts,epoch,step,lr,loss,loss_byol,loss_mfr,tokens_per_s,vram_gb\n", encoding="utf-8")
LOG_JL  = LOGS / "script6_train_log.jsonl"

def log_row(d: dict):
    d2 = d.copy(); d2["ts"]=datetime.now().isoformat(timespec="seconds")
    with open(LOG_JL,"a",encoding="utf-8") as f: f.write(json.dumps(d2,ensure_ascii=False)+"\n")
    with open(LOG_CSV,"a",encoding="utf-8") as f:
        f.write(f'{d2["ts"]},{d2.get("epoch",0)},{d2.get("step",0)},'
                f'{d2.get("lr",0):.6f},{d2.get("loss",0):.6f},{d2.get("loss_byol",0):.6f},'
                f'{d2.get("loss_mfr",0):.6f},{d2.get("tps",0):.2f},{d2.get("vram_gb",0):.2f}\n')

def save_ckpt(tag: str):
    fn = WEIGHTS / f"script6_student_{tag}.safetensors"
    state = {k: v.detach().cpu() for k,v in student.state_dict().items()}
    save_safetensors(state, str(fn))
    (WEIGHTS / "latest.txt").write_text(fn.name, encoding="utf-8")
    print(f"[SAVE] {fn.name}")

def try_resume():
    if not CONFIG["resume_if_available"]: return False
    txt = WEIGHTS / "latest.txt"
    if not txt.exists(): return False
    ck = WEIGHTS / txt.read_text(encoding="utf-8").strip()
    if not ck.exists(): return False
    print(f"[RESUME] Loading {ck.name}")
    sd = load_safetensors(str(ck))
    student.load_state_dict(sd, strict=True)
    teacher.load_state_dict(sd, strict=False)  # teacher weights will sync by EMA
    return True

# Training loop
total_steps = CONFIG["epochs"] * (len(slides)//CONFIG["batch_slides"] + 1)
if CONFIG["steps_per_epoch_cap"]:
    total_steps = CONFIG["epochs"] * CONFIG["steps_per_epoch_cap"]
sched = CosineWarmup(opt, warmup=CONFIG["warmup_steps"], max_steps=total_steps, base_lr=CONFIG["lr"])

resumed = try_resume()
print(f"[TRAIN] slides={len(slides)} | batch_slides={CONFIG['batch_slides']} | grad_accum={CONFIG['grad_accum']} | epochs={CONFIG['epochs']} | resume={resumed}")

global_step=0
for epoch in range(1, CONFIG["epochs"]+1):
    random.shuffle(slides)
    steps_this_epoch = 0
    max_steps_epoch = (CONFIG["steps_per_epoch_cap"] or (len(slides)//CONFIG["batch_slides"] + 1))

    i = 0
    while steps_this_epoch < max_steps_epoch and i < len(slides):
        batch_recs = slides[i : i+CONFIG["batch_slides"]]
        i += CONFIG["batch_slides"]

        try:
            b = make_batch(batch_recs, CONFIG["token_budget_0p5"], CONFIG["token_budget_2p0"], CONFIG["mask_frac"])
        except AssertionError as ae:
            print(f"[SKIP] {batch_recs[0].slide_id} assert: {ae}"); continue
        except Exception as e:
            print(f"[SKIP] Batch error: {e}"); continue

        feats, mmxy, scl, pad, mfr_index = b["feats"], b["mmxy"], b["scl"], b["pad"], b["mfr_index"]
        tokens_total = int((~pad).sum().item())

        opt.zero_grad(set_to_none=True)
        t0 = perf_counter()

        # teacher forward
        with torch.no_grad():
            g_t, t_t, _, _ = teacher(feats, mmxy, scl, pad)

        # student forward + losses
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE=="cuda" and AMP_DTYPE!=torch.float32)):
            g_s, t_s, g_sp, t_sp = student(feats, mmxy, scl, pad)
            loss_byol = cosine_loss(g_sp, g_t)
            bi = mfr_index
            t_s_mask = t_sp[bi[:,0], bi[:,1], :]
            t_t_mask = t_t[bi[:,0], bi[:,1], :]
            loss_mfr = cosine_loss(t_s_mask, t_t_mask)
            loss = loss_byol + CONFIG["lambda_mfr"] * loss_mfr

        scaler.scale(loss / CONFIG["grad_accum"]).backward()

        if ((steps_this_epoch+1) % CONFIG["grad_accum"] == 0):
            scaler.step(opt)
            scaler.update()
            ema_update(teacher, student, tau=CONFIG["ema_tau"])
            lr = sched.step()
        else:
            lr = sched.opt.param_groups[0]["lr"]

        if DEVICE=="cuda":
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated()/(1024**3)
            torch.cuda.reset_peak_memory_stats()
        else:
            vram = 0.0

        dt = perf_counter()-t0
        tps = tokens_total/max(dt,1e-6)

        global_step += 1
        steps_this_epoch += 1

        if global_step % CONFIG["log_every_steps"] == 0:
            print(f"[E{epoch} S{global_step}] loss={loss.item():.4f} (byol {loss_byol.item():.4f} | mfr {loss_mfr.item():.4f}) "
                  f"| tokens={tokens_total} | {tps:.1f} tok/s | lr={lr:.2e} | VRAM~{vram:.2f} GB")
            log_row({"epoch":epoch, "step":global_step, "lr":lr,
                     "loss":float(loss.item()), "loss_byol":float(loss_byol.item()),
                     "loss_mfr":float(loss_mfr.item()), "tps":float(tps), "vram_gb":float(vram)})

        if global_step % CONFIG["save_every_steps"] == 0:
            save_ckpt(f"e{epoch}_s{global_step}")

        # light periodic cleanup
        if (global_step % 200) == 0:
            del feats, mmxy, scl, pad, mfr_index, g_t, t_t, g_s, t_s, g_sp, t_sp
            gc.collect()
            if DEVICE=="cuda": torch.cuda.empty_cache()

    save_ckpt(f"e{epoch}")

print("[TRAIN] Finished Script 6 pretraining.")

# Optional: quick curve (skips if matplotlib missing)
try:
    df_plot = pd.read_csv(LOG_CSV)
    if HAS_MPL and not df_plot.empty:
        plt.figure(figsize=(8,5))
        plt.plot(df_plot["step"], df_plot["loss"], label="loss")
        if "loss_byol" in df_plot: plt.plot(df_plot["step"], df_plot["loss_byol"], label="BYOL")
        if "loss_mfr" in df_plot:  plt.plot(df_plot["step"], df_plot["loss_mfr"],  label="MFR")
        plt.xlabel("step"); plt.ylabel("loss"); plt.grid(True, alpha=0.3); plt.legend()
        outp = FIGS / "script6_training_curves.png"
        plt.tight_layout(); plt.savefig(outp, dpi=150); plt.close()
        print(f"[FIG] {outp}")
    else:
        print("[SKIP] Plotting not available or log empty.")
except Exception as e:
    print(f"[WARN] Plotting skipped: {e}")

# Optional: export slide embeddings
def export_embeddings(ckpt_name: Optional[str]=None, use_budget=True):
    if ckpt_name is None:
        txt = (WEIGHTS / "latest.txt")
        assert txt.exists(), "Missing weights/latest.txt"
        ckpt_name = txt.read_text(encoding="utf-8").strip()
    ckpt_path = WEIGHTS / ckpt_name
    print(f"[EXPORT] Loading {ckpt_path.name}")
    sd = load_safetensors(str(ckpt_path))
    student.load_state_dict(sd, strict=True)
    student.eval()

    count=0; t0=perf_counter()
    for rec in slides:
        outn = EMBED / f"{rec.slide_id}.npy"
        if outn.exists(): continue
        if use_budget:
            f, mm, sc = load_tokens_for_slide(rec, CONFIG["token_budget_0p5"], CONFIG["token_budget_2p0"])
        else:
            f05 = np.load(rec.npy05, mmap_mode='r'); m05 = load_meta(rec.meta05)
            f20 = np.load(rec.npy20, mmap_mode='r'); m20 = load_meta(rec.meta20)
            f = np.concatenate([f05, f20], axis=0).astype(np.float32)
            mm = np.concatenate([compute_mm_xy(m05), compute_mm_xy(m20)], axis=0).astype(np.float32)
            sc = np.concatenate([
                m05["scale_um_per_px"].to_numpy(dtype=np.float32).reshape(-1,1),
                m20["scale_um_per_px"].to_numpy(dtype=np.float32).reshape(-1,1)
            ], axis=0).astype(np.float32)
        feats = torch.from_numpy(f).unsqueeze(0).to(DEVICE)
        mmxy  = torch.from_numpy(mm).unsqueeze(0).to(DEVICE)
        scl   = torch.from_numpy(sc).unsqueeze(0).to(DEVICE)
        pad   = torch.zeros(1, feats.size(1), dtype=torch.bool, device=DEVICE)
        with torch.no_grad():
            g_proj, _, _, _ = student(feats, mmxy, scl, pad)
        emb = g_proj.squeeze(0).detach().cpu().numpy().astype(np.float32)
        np.save(outn, emb)
        count += 1
        if count % 200 == 0:
            print(f"[EMB] {count}/{len(slides)} saved...")
    dt = perf_counter()-t0
    print(f"[EMB] Done: {count} slides in {dt/60:.1f} min")

if CONFIG["export_embeddings_after_train"]:
    export_embeddings(ckpt_name=None, use_budget=CONFIG["export_use_budget"])

print("[DONE] Script 6 complete.")


# SCRIPT 06B: ENCODER CHECKPOINT FINALIZATION
# Script 6B — Finalize & Save Encoder Checkpoint
import os, sys, json, time, random, shutil, subprocess, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# Paths
WORKSPACE   = WORKSPACE
FEAT05_DIR  = WORKSPACE / "features" / "scale0p5"
FEAT20_DIR  = WORKSPACE / "features" / "scale2p0"
MODELS_DIR  = WORKSPACE / "models"
LOGS_DIR    = WORKSPACE / "logs"
for p in (MODELS_DIR, LOGS_DIR): p.mkdir(parents=True, exist_ok=True)

STUDENT_OUT = MODELS_DIR / "openslidefm_student.pt"
TEACHER_OUT = MODELS_DIR / "openslidefm_teacher_ema.pt"
MANIFEST    = MODELS_DIR / "openslidefm_checkpoint_manifest.json"
TRAIN_LOG   = LOGS_DIR / "script6c_finalize_log.csv"

# Deps
def _ensure(pkgs):
    miss=[]
    for name, spec in pkgs:
        try: __import__(name)
        except Exception: miss.append(spec)
    if miss:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])

_ensure([("numpy","numpy>=1.24"), ("torch","torch>=2.1")])

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16 if DEVICE=="cuda" else torch.bfloat16

# Config
CFG = {
    "token_dim": 768,
    "budget_0p5": 1200,       # target tokens @ 0.5 µm/px
    "budget_2p0": 400,        # target tokens @ 2.0 µm/px
    "mask_frac": 0.25,
    "d_model": 768,
    "nhead": 8,
    "nlayers": 6,
    "dropout": 0.1,
    "proj_dim": 256,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "total_steps": 400,       # short top-up to materialize weights
    "ema_decay": 0.996,
    "batch_slides": 3,
    "print_every": 20,
    "seed": 1337,
}

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(CFG["seed"])
torch.set_float32_matmul_precision("high")

# Utilities
def list_slide_ids():
    s05 = {p.stem for p in FEAT05_DIR.glob("*.npy")}
    s20 = {p.stem for p in FEAT20_DIR.glob("*.npy")}
    inter = sorted(s05 & s20)
    return inter

def _sample_idx(n_avail: int, k: int) -> np.ndarray:
    if n_avail <= 0:
        return np.zeros((0,), dtype=np.int64)
    replace = n_avail < k
    return np.random.choice(n_avail, size=k, replace=replace).astype(np.int64)

def load_tokens_fixed(slide_id: str, k05: int, k20: int) -> np.ndarray:
    """Always returns shape [(k05+k20), 768]. Uses replacement if needed."""
    f05 = np.load(FEAT05_DIR / f"{slide_id}.npy", mmap_mode="r")  # [N05,768] float32
    f20 = np.load(FEAT20_DIR / f"{slide_id}.npy", mmap_mode="r")  # [N20,768]
    i05 = _sample_idx(int(f05.shape[0]), k05)
    i20 = _sample_idx(int(f20.shape[0]), k20)
    x05 = f05[i05]
    x20 = f20[i20]
    # Guard against any unexpected dtype/shape issues
    x05 = x05.astype(np.float32, copy=False).reshape(k05, CFG["token_dim"])
    x20 = x20.astype(np.float32, copy=False).reshape(k20, CFG["token_dim"])
    x   = np.concatenate([x05, x20], axis=0)  # [(k05+k20), 768]
    return x

def feature_view(x: np.ndarray, drop_p=0.1, noise_std=0.02) -> np.ndarray:
    """Simple feature-space augmentation (keeps shape)."""
    if drop_p > 0:
        m = (np.random.rand(*x.shape) > drop_p).astype(np.float32)
        x = x * m
    if noise_std > 0:
        x = x + np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return x

def write_log_row(step:int, loss:float, l_byol:float, l_mfr:float, tps:int, vram_gb:float):
    header = ["ts","step","loss","loss_byol","loss_mfr","tps","vram_gb"]
    if not TRAIN_LOG.exists():
        TRAIN_LOG.write_text(",".join(header) + "\n", encoding="utf-8")
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "step": step,
        "loss": round(float(loss),6),
        "loss_byol": round(float(l_byol),6),
        "loss_mfr": round(float(l_mfr),6),
        "tps": int(tps),
        "vram_gb": round(float(vram_gb),2),
    }
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(",".join(str(row[h]) for h in header) + "\n")

# Model
class TransformerMIL(nn.Module):
    def __init__(self, d_model=768, nhead=8, nlayers=6, dropout=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.ln  = nn.LayerNorm(d_model)
    def forward(self, tokens: torch.Tensor):  # [B,T,768]
        B, T, D = tokens.shape
        cls = self.cls.expand(B, -1, -1)         # [B,1,D]
        x = torch.cat([cls, tokens], dim=1)      # [B,1+T,D]
        x = self.enc(x)                          # [B,1+T,D]
        x = self.ln(x)
        cls_emb = x[:,0]                         # [B,D]
        tok_emb = x[:,1:]                        # [B,T,D]
        return cls_emb, tok_emb

class BYOLHead(nn.Module):
    def __init__(self, d_model=768, proj_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, proj_dim)
        )
        self.pred = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
    def forward(self, h):  # [B,D]
        z = F.normalize(self.proj(h), dim=-1)
        p = F.normalize(self.pred(z), dim=-1)
        return z, p

class EncoderWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = TransformerMIL(cfg["d_model"], cfg["nhead"], cfg["nlayers"], cfg["dropout"])
        self.head     = BYOLHead(cfg["d_model"], cfg["proj_dim"])
    def forward(self, tokens):  # [B,T,768]
        cls_emb, tok_emb = self.backbone(tokens)
        z, p = self.head(cls_emb)
        return cls_emb, tok_emb, z, p

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, decay: float):
    for t, s in zip(teacher.parameters(), student.parameters()):
        t.data.mul_(decay).add_(s.data, alpha=1.0 - decay)

def byol_loss(p_s, z_t):
    return 2.0 - 2.0 * (p_s * z_t.detach()).sum(dim=-1).mean()

def mfr_loss(tok_s, tok_t, mask):
    # mask: [B,T] bool — random subset; we always have full tokens (fixed shape), so no padding mask needed.
    if mask is None or mask.sum() == 0:
        return torch.tensor(0.0, device=tok_s.device)
    diff = tok_s[mask] - tok_t.detach()[mask]
    return (diff*diff).mean()

# Main
if STUDENT_OUT.exists() and TEACHER_OUT.exists():
    print(f"[OK] Checkpoints already exist:\n - {STUDENT_OUT}\n - {TEACHER_OUT}")
else:
    slide_ids = list_slide_ids()
    assert len(slide_ids) >= 100, f"Too few 2-scale slides: {len(slide_ids)}"
    print(f"[INFO] Slides with both scales: {len(slide_ids)}")
    print(f"[INFO] Device={DEVICE}, AMP={AMP_DTYPE}")

    model_s = EncoderWrapper(CFG).to(DEVICE)
    model_t = EncoderWrapper(CFG).to(DEVICE)
    model_t.load_state_dict(model_s.state_dict())

    opt    = torch.optim.AdamW(model_s.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE=="cuda"))

    T = CFG["budget_0p5"] + CFG["budget_2p0"]
    B = CFG["batch_slides"]
    tokens_per_batch = B * T

    step = 0
    t0 = time.time()

    while step < CFG["total_steps"]:

        # batch: fixed-shape tokens for all slides
        batch_ids = random.sample(slide_ids, B)
        xs, xs2, xt = [], [], []
        for sid in batch_ids:
            base = load_tokens_fixed(sid, CFG["budget_0p5"], CFG["budget_2p0"])  # [T,768], fixed shape
            xs.append(feature_view(base, drop_p=0.1, noise_std=0.02))
            xs2.append(feature_view(base, drop_p=0.1, noise_std=0.02))
            xt.append(base)

        x1 = torch.from_numpy(np.stack(xs,  axis=0)).to(DEVICE, non_blocking=True)  # [B,T,768]
        x2 = torch.from_numpy(np.stack(xs2, axis=0)).to(DEVICE, non_blocking=True)
        xt = torch.from_numpy(np.stack(xt,  axis=0)).to(DEVICE, non_blocking=True)

        # random mask for MFR (same shape for all)
        mask = (torch.rand((B, T), device=DEVICE) < CFG["mask_frac"])

        with torch.amp.autocast(device_type=("cuda" if DEVICE=="cuda" else "cpu"), dtype=AMP_DTYPE, enabled=True):
            cls1, tok1, z1, p1 = model_s(x1)
            cls2, tok2, z2, p2 = model_s(x2)
            with torch.no_grad():
                cls_t, tok_t, zt, _ = model_t(xt)

            L_byol = 0.5 * byol_loss(p1, zt) + 0.5 * byol_loss(p2, zt)
            L_mfr  = 0.5 * mfr_loss(tok1, tok_t, mask) + 0.5 * mfr_loss(tok2, tok_t, mask)
            loss   = L_byol + 0.5 * L_mfr

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        ema_update(model_t, model_s, CFG["ema_decay"])

        step += 1
        if step == 1 or step % CFG["print_every"] == 0 or step == CFG["total_steps"]:
            dt  = max(1e-6, time.time() - t0)
            tps = int((step * tokens_per_batch) / dt)
            vram = torch.cuda.max_memory_allocated() / (1024**3) if DEVICE=="cuda" else 0.0
            print(f"[S{step:05d}] loss={loss.item():.4f} (byol {L_byol.item():.4f} | mfr {L_mfr.item():.4f}) | "
                  f"tps={tps:,} | VRAM~{vram:.2f} GB")
            write_log_row(step, loss.item(), L_byol.item(), L_mfr.item(), tps, vram)

    # Save final checkpoints
    torch.save(model_s.state_dict(), STUDENT_OUT)
    torch.save(model_t.state_dict(), TEACHER_OUT)

    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE,
        "dtype": str(AMP_DTYPE).split(".")[-1],
        "slides_2scale": len(slide_ids),
        "config": CFG,
        "student_path": str(STUDENT_OUT),
        "teacher_path": str(TEACHER_OUT),
    }
    MANIFEST.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n[OK] Saved:")
    print(" -", STUDENT_OUT)
    print(" -", TEACHER_OUT)
    print(" -", MANIFEST)

print(f"\n[CHECK] checkpoints_present = {STUDENT_OUT.exists() and TEACHER_OUT.exists()}")


# SCRIPT 07: TCGA 31-CLASS PAN-CANCER EVALUATION
# COMPLETE TCGA EVALUATION PIPELINE
# Trains and evaluates cancer classification on TCGA dataset to establish baseline performance for comparison with external validation (CAM16/17/PANDA).
import os
import sys
import json
import warnings
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix,
    balanced_accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# CONFIGURATION
class Config:
    # Paths
    OPENSLIDE = WORKSPACE
    DL_V2 = WSI_ROOT.parent
    
    # Data
    EMBEDDINGS = DL_V2 / "artifacts" / "embeddings" / "patient_means_clean_run_20250908_020405_emb_openclip_vitb16_turbo.parquet"
    LABELS = DL_V2 / "artifacts" / "labels" / "labels.csv"
    MANIFEST = OPENSLIDE / "manifests" / "manifest_tcga.csv"
    
    # Output
    OUTPUT = OPENSLIDE / "results" / "tcga_baseline_evaluation"
    
    # Model
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 64
    PATIENCE = 10
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = Config()

# UTILITIES
def print_header(text):
    print(f"\n{'='*80}")
    print(f" {text}")
    print('='*80)

def print_subheader(text):
    print(f"\n{'-'*80}")
    print(f" {text}")
    print('-'*80)

# DATA LOADING
def load_data():
    """Load embeddings, labels, and manifest"""
    print_header("1. LOADING DATA")
    
    # Load embeddings
    print("\n📦 Loading embeddings...")
    df_emb = pd.read_parquet(CFG.EMBEDDINGS)
    print(f"  ✓ Embeddings: {df_emb.shape}")
    print(f"    Patients: {len(df_emb)}")
    print(f"    Features: {df_emb.shape[1]}")
    
    # Load labels
    print("\n📋 Loading labels...")
    df_labels = pd.read_csv(CFG.LABELS)
    print(f"  ✓ Labels: {df_labels.shape}")
    
    # Check split distribution
    if 'split' in df_labels.columns:
        split_dist = df_labels['split'].value_counts()
        print(f"\n  Split distribution:")
        for split, count in split_dist.items():
            print(f"    {split}: {count}")
    
    # Load manifest for cancer codes
    print("\n🗂️  Loading manifest...")
    df_manifest = pd.read_csv(CFG.MANIFEST)
    print(f"  ✓ Manifest: {df_manifest.shape}")
    print(f"    Total slides: {len(df_manifest)}")
    print(f"    Cancer types: {df_manifest['cancer_code'].nunique()}")
    
    return df_emb, df_labels, df_manifest

def prepare_dataset(df_emb, df_labels, df_manifest):
    """Prepare train/test datasets with labels"""
    print_header("2. PREPARING DATASET")
    
    # Extract patient IDs from embeddings index
    print("\n🔗 Mapping patients to cancer types...")
    
    # Get patient-to-cancer mapping from manifest
    # Extract patient ID from slide_id (e.g., TCGA-02-0001-01A-01-TS1 -> TCGA-02-0001)
    df_manifest['patient_id'] = df_manifest['slide_id'].str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', expand=False)
    
    # Get unique patient-cancer mapping
    patient_cancer_map = df_manifest.groupby('patient_id')['cancer_code'].first().to_dict()
    
    # Map embeddings to cancer types
    df_emb['patient_id'] = df_emb.index
    df_emb['cancer_type'] = df_emb['patient_id'].map(patient_cancer_map)
    
    # Remove patients without cancer labels
    df_emb_labeled = df_emb[df_emb['cancer_type'].notna()].copy()
    print(f"  ✓ Patients with labels: {len(df_emb_labeled)}")
    print(f"    Removed {len(df_emb) - len(df_emb_labeled)} patients without labels")
    
    # Add split information from labels.csv if available
    if 'split' in df_labels.columns:
        # Create patient-split mapping
        df_labels['patient_id'] = df_labels['patient']
        patient_split_map = df_labels.set_index('patient_id')['split'].to_dict()
        df_emb_labeled['split'] = df_emb_labeled['patient_id'].map(patient_split_map)
        
        # Use patients with defined splits
        df_emb_labeled = df_emb_labeled[df_emb_labeled['split'].notna()].copy()
        print(f"  ✓ Patients with train/test split: {len(df_emb_labeled)}")
    else:
        # Create random split if none exists
        print("  ⚠️  No split found, creating 80/10/10 split...")
        from sklearn.model_selection import train_test_split
        patients = df_emb_labeled['patient_id'].values
        train_val, test = train_test_split(patients, test_size=0.1, random_state=42)
        train, val = train_test_split(train_val, test_size=0.111, random_state=42)  # 0.111 * 0.9 ≈ 0.1
        
        split_map = {}
        for p in train: split_map[p] = 'train'
        for p in val: split_map[p] = 'val'
        for p in test: split_map[p] = 'test'
        df_emb_labeled['split'] = df_emb_labeled['patient_id'].map(split_map)
    
    # Show cancer type distribution
    print(f"\n📊 Cancer type distribution:")
    cancer_counts = df_emb_labeled['cancer_type'].value_counts()
    print(f"  Total cancer types: {len(cancer_counts)}")
    print(f"  Top 10:")
    for cancer, count in cancer_counts.head(10).items():
        print(f"    {cancer}: {count}")
    
    # Show split distribution
    print(f"\n📊 Split distribution:")
    for split in ['train', 'val', 'test']:
        count = (df_emb_labeled['split'] == split).sum()
        print(f"  {split}: {count}")
    
    # Prepare feature matrix and labels
    feature_cols = [c for c in df_emb_labeled.columns if c.startswith('f')]
    X = df_emb_labeled[feature_cols].values.astype(np.float32)
    
    # Encode cancer types
    le = LabelEncoder()
    y = le.fit_transform(df_emb_labeled['cancer_type'].values)
    
    print(f"\n✓ Feature matrix: {X.shape}")
    print(f"✓ Number of classes: {len(le.classes_)}")
    
    # Split data
    train_mask = df_emb_labeled['split'] == 'train'
    val_mask = df_emb_labeled['split'] == 'val'
    test_mask = df_emb_labeled['split'] == 'test'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\n✓ Train: {X_train.shape}, {len(np.unique(y_train))} classes")
    print(f"✓ Val:   {X_val.shape}, {len(np.unique(y_val))} classes")
    print(f"✓ Test:  {X_test.shape}, {len(np.unique(y_test))} classes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), le, df_emb_labeled

# MODEL
class CancerClassifier(nn.Module):
    """Simple MLP for cancer classification"""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# TRAINING
def train_model(train_data, val_data, num_classes):
    """Train cancer classifier"""
    print_header("3. TRAINING MODEL")
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = CancerClassifier(
        input_dim=input_dim,
        hidden_dim=CFG.HIDDEN_DIM,
        num_classes=num_classes,
        dropout=CFG.DROPOUT
    ).to(CFG.DEVICE)
    
    print(f"\n🧠 Model architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {CFG.HIDDEN_DIM}")
    print(f"  Output classes: {num_classes}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"  Epochs: {CFG.EPOCHS}")
    print(f"  Batch size: {CFG.BATCH_SIZE}")
    print(f"  Learning rate: {CFG.LEARNING_RATE}")
    print(f"  Device: {CFG.DEVICE}")
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, CFG.EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(CFG.DEVICE)
            y_batch = y_batch.to(CFG.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(CFG.DEVICE)
                y_batch = y_batch.to(CFG.DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CFG.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Ensure output directory exists
            CFG.OUTPUT.mkdir(parents=True, exist_ok=True)
            
            # Save best model (workaround for Unicode path issue)
            # Save to temp file, then copy using pure Python binary I/O
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pth') as tmp:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, tmp)
                tmp_path = tmp.name
            
            # Copy using pure Python binary I/O (handles Unicode)
            final_path = CFG.OUTPUT / "best_model.pth"
            try:
                with open(tmp_path, 'rb') as src:
                    with open(final_path, 'wb') as dst:
                        dst.write(src.read())
            finally:
                os.unlink(tmp_path)  # Delete temp file
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                print(f"\n⏸️  Early stopping triggered at epoch {epoch}")
                break
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    
    # Load best model (workaround for Unicode path)
    model_path = str(CFG.OUTPUT / "best_model.pth")
    with open(model_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=CFG.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history

# EVALUATION
@torch.no_grad()
def evaluate_model(model, test_data, label_encoder):
    """Evaluate on test set"""
    print_header("4. EVALUATING ON TEST SET")
    
    X_test, y_test = test_data
    
    # Create dataloader
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    # Predict
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []
    
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(CFG.DEVICE)
        outputs = model(X_batch)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_true.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs)
    all_true = np.array(all_true)
    
    # Calculate metrics
    print("\n📊 Test Set Performance:")
    
    # Overall accuracy
    acc = accuracy_score(all_true, all_preds)
    print(f"\n  Overall Accuracy: {acc:.4f}")
    
    # Balanced accuracy
    bal_acc = balanced_accuracy_score(all_true, all_preds)
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    
    # Macro F1
    macro_f1 = f1_score(all_true, all_preds, average='macro')
    print(f"  Macro F1: {macro_f1:.4f}")
    
    # Weighted F1
    weighted_f1 = f1_score(all_true, all_preds, average='weighted')
    print(f"  Weighted F1: {weighted_f1:.4f}")
    
    # Multi-class AUC (one-vs-rest)
    try:
        auc_ovr = roc_auc_score(all_true, all_probs, multi_class='ovr', average='macro')
        print(f"  Macro AUC (OvR): {auc_ovr:.4f}")
    except:
        auc_ovr = None
        print(f"  Macro AUC (OvR): N/A")
    
    # Per-class metrics
    print(f"\n📋 Classification Report:")
    class_names = label_encoder.classes_
    report = classification_report(
        all_true, all_preds,
        target_names=class_names,
        digits=3
    )
    print(report)
    
    # Save detailed metrics
    results = {
        'overall': {
            'accuracy': float(acc),
            'balanced_accuracy': float(bal_acc),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'macro_auc_ovr': float(auc_ovr) if auc_ovr is not None else None,
            'num_samples': int(len(all_true)),
            'num_classes': int(len(class_names))
        },
        'per_class': classification_report(
            all_true, all_preds,
            target_names=class_names,
            output_dict=True
        )
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_true, all_preds)
    
    return results, cm, all_preds, all_probs, all_true

# VISUALIZATION
def plot_results(history, cm, label_encoder):
    """Create visualization plots"""
    print_header("5. CREATING VISUALIZATIONS")
    
    fig = plt.figure(figsize=(20, 5))
    
    # Training curves
    ax1 = plt.subplot(1, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(epochs, history['val_acc'], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix (top 15 classes by support)
    ax3 = plt.subplot(1, 3, 3)
    
    # Get top classes
    class_support = cm.sum(axis=1)
    top_indices = np.argsort(class_support)[-15:][::-1]
    cm_top = cm[np.ix_(top_indices, top_indices)]
    class_names_top = label_encoder.classes_[top_indices]
    
    sns.heatmap(cm_top, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names_top, yticklabels=class_names_top,
                ax=ax3, cbar_kws={'label': 'Count'})
    ax3.set_title('Confusion Matrix (Top 15 Classes)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=12)
    ax3.set_ylabel('True', fontsize=12)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plot_path = str(CFG.OUTPUT / 'training_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: training_results.png")
    plt.close()

# SAVE RESULTS
def save_results(results, cm, label_encoder, df_labeled, y_pred, y_true):
    """Save all results to disk"""
    print_header("6. SAVING RESULTS")
    
    # Save metrics JSON
    with open(str(CFG.OUTPUT / 'test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved: test_metrics.json")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    cm_df.to_csv(str(CFG.OUTPUT / 'confusion_matrix.csv'))
    print(f"  ✓ Saved: confusion_matrix.csv")
    
    # Save per-class metrics
    per_class_df = pd.DataFrame(results['per_class']).T
    per_class_df.to_csv(str(CFG.OUTPUT / 'per_class_metrics.csv'))
    print(f"  ✓ Saved: per_class_metrics.csv")
    
    # Save predictions
    test_mask = df_labeled['split'] == 'test'
    test_patients = df_labeled[test_mask]['patient_id'].values
    
    pred_df = pd.DataFrame({
        'patient_id': test_patients,
        'true_label': label_encoder.inverse_transform(y_true),
        'pred_label': label_encoder.inverse_transform(y_pred),
        'correct': y_true == y_pred
    })
    pred_df.to_csv(str(CFG.OUTPUT / 'test_predictions.csv'), index=False)
    print(f"  ✓ Saved: test_predictions.csv")
    
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'TCGA',
        'num_patients': len(df_labeled),
        'num_classes': len(label_encoder.classes_),
        'train_size': int((df_labeled['split'] == 'train').sum()),
        'val_size': int((df_labeled['split'] == 'val').sum()),
        'test_size': int((df_labeled['split'] == 'test').sum()),
        'model': {
            'type': 'MLP',
            'hidden_dim': CFG.HIDDEN_DIM,
            'dropout': CFG.DROPOUT,
            'learning_rate': CFG.LEARNING_RATE,
            'weight_decay': CFG.WEIGHT_DECAY
        },
        'results': results['overall']
    }
    
    with open(str(CFG.OUTPUT / 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: summary.json")
    
    print(f"\n✓ All results saved to: {CFG.OUTPUT}")

# MAIN
def main():
    """Main execution"""
    print("="*80)
    print(" TCGA BASELINE EVALUATION PIPELINE")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {CFG.OUTPUT}")
    
    # Ensure output directory exists
    CFG.OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    df_emb, df_labels, df_manifest = load_data()
    
    # 2. Prepare dataset
    train_data, val_data, test_data, label_encoder, df_labeled = prepare_dataset(
        df_emb, df_labels, df_manifest
    )
    
    # 3. Train model
    model, history = train_model(train_data, val_data, num_classes=len(label_encoder.classes_))
    
    # 4. Evaluate
    results, cm, y_pred, y_probs, y_true = evaluate_model(model, test_data, label_encoder)
    
    # 5. Visualize
    plot_results(history, cm, label_encoder)
    
    # 6. Save
    save_results(results, cm, label_encoder, df_labeled, y_pred, y_true)
    
    # Final summary
    print_header("SUMMARY")
    print(f"\n✅ TCGA Baseline Evaluation Complete!")
    print(f"\n📊 Key Metrics:")
    print(f"  Test Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {results['overall']['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {results['overall']['macro_f1']:.4f}")
    if results['overall']['macro_auc_ovr'] is not None:
        print(f"  Macro AUC (OvR): {results['overall']['macro_auc_ovr']:.4f}")
    
    print(f"\n📁 Results saved to: {CFG.OUTPUT}")
    print(f"\n💡 Next Steps:")
    print(f"  1. Compare TCGA test metrics with CAM16/17/PANDA")
    print(f"  2. Calculate performance drop: (CAM16 - TCGA test)")
    print(f"  3. Include in publication tables")
    
    print("\n" + "="*80)
    print(" COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()

# SCRIPT 06C: POST-PRETRAINING DIAGNOSTICS
# Script 6C — Post-Pretraining Diagnostics
import os, sys, json, math, hashlib, shutil, subprocess, warnings, tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Paths
WORKSPACE   = WORKSPACE
LOGS_DIR    = WORKSPACE / "logs"
MODELS_DIR  = WORKSPACE / "models"
FEAT05_DIR  = WORKSPACE / "features" / "scale0p5"
FEAT20_DIR  = WORKSPACE / "features" / "scale2p0"
DIAG_DIR    = WORKSPACE / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV = LOGS_DIR / "script6_train_log.csv"

# Deps (install quietly if missing)
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

# Helpers
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

# Load logs (robust to missing cols)
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

# Derive training stats
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

# Feature coverage (2-scale)
n_both, n05, n20 = count_2scale_slides()
diag["features_2scale_intersection"] = int(n_both)
diag["features_0p5_count"] = int(n05)
diag["features_2p0_count"] = int(n20)

# Checkpoints
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

# Gates (PASS/WARN/FAIL)
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

# Save reports
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


# SCRIPT 08: SLIDE EMBEDDINGS EXPORT
# Script8 — Slide Embeddings Export
import os, sys, json, time, math, shutil, gc
from pathlib import Path
from datetime import datetime
import subprocess, warnings
warnings.filterwarnings("ignore")

# Workspace
# WORKSPACE — set in CONFIGURATION block above
FEAT05 = WORKSPACE / "features" / "scale0p5"
FEAT20 = WORKSPACE / "features" / "scale2p0"
EMB_DIR = WORKSPACE / "embeddings"
MANIFESTS = WORKSPACE / "manifests"
DIAG = WORKSPACE / "diagnostics"
for p in [EMB_DIR, DIAG]:
    p.mkdir(parents=True, exist_ok=True)

# Deps
def _ensure(pkgs):
    miss=[]
    for name, spec in pkgs:
        try: __import__(name)
        except Exception: miss.append(spec)
    if miss:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])

_ensure([
    ("numpy","numpy>=1.24"),
    ("pandas","pandas>=2.0"),
])

import numpy as np
import pandas as pd

# Load manifests (dataset tagging)
def load_slide_sets():
    sets = {}
    def _load_csv(name):
        p = MANIFESTS / f"manifest_{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            # Support both columns we've used: ('slide_id','path') or ('filename', etc.)
            sid_col = "slide_id" if "slide_id" in df.columns else ("filename" if "filename" in df.columns else None)
            if sid_col is None:
                return set()
            sids = set((df["slide_id"] if "slide_id" in df.columns else pd.Series([Path(x).stem for x in df["filename"]])))
            return sids
        return set()

    sets["tcga"] = _load_csv("tcga")  # from Script 2
    sets["camelyon16"] = _load_csv("camelyon16")
    sets["camelyon17"] = _load_csv("camelyon17")
    return sets

SLIDESETS = load_slide_sets()

# Discover features present at both scales
def available_two_scale_ids():
    s05 = set([p.stem for p in FEAT05.glob("*.npy")])
    s20 = set([p.stem for p in FEAT20.glob("*.npy")])
    both = sorted(list(s05 & s20))
    return both

TWO_SCALE_IDS = available_two_scale_ids()

# Decide dataset for each slide_id
def dataset_of(slide_id: str) -> str:
    # Priority: camelyon16 / camelyon17 / tcga / other
    if slide_id in SLIDESETS.get("camelyon16", set()): return "CAMELYON16"
    if slide_id in SLIDESETS.get("camelyon17", set()): return "CAMELYON17"
    if slide_id in SLIDESETS.get("tcga", set()):       return "TCGA"
    return "OTHER"

# Export logic
def embed_one(slide_id: str) -> dict:
    f05 = FEAT05 / f"{slide_id}.npy"
    f20 = FEAT20 / f"{slide_id}.npy"
    if not (f05.exists() and f20.exists()):
        return {"slide_id": slide_id, "ok": False, "reason": "missing_feature_file"}

    try:
        a = np.load(f05, mmap_mode="r")  # [T1,768]
        b = np.load(f20, mmap_mode="r")  # [T2,768]
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 768 or b.shape[1] != 768:
            return {"slide_id": slide_id, "ok": False, "reason": f"bad_shape a{tuple(a.shape)} b{tuple(b.shape)}"}
        v05 = a.mean(axis=0).astype(np.float32)
        v20 = b.mean(axis=0).astype(np.float32)
        emb = ((v05 + v20) * 0.5).astype(np.float32)  # [768]
        ds = dataset_of(slide_id)
        out_dir = EMB_DIR / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slide_id}.npy"
        np.save(out_path, emb)

        return {
            "slide_id": slide_id,
            "dataset": ds,
            "ok": True,
            "path_emb": str(out_path),
            "t05": int(a.shape[0]),
            "t20": int(b.shape[0]),
            "norm": float(np.linalg.norm(emb)),
        }
    except Exception as e:
        return {"slide_id": slide_id, "ok": False, "reason": f"{type(e).__name__}:{e}"}

# Driver
CONFIG = {
    "only_datasets": ["CAMELYON16","CAMELYON17"],  # << do these first; set to None to do ALL (incl. TCGA)
    "workers": 16,     # Threaded I/O; safe in notebook on Windows
    "print_every": 200 # progress print interval (slides)
}

def main():
    print("== Script 8 — Slide Embeddings Export ==")
    print(json.dumps({
        "time": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(WORKSPACE),
        "features_0p5": str(FEAT05),
        "features_2p0": str(FEAT20),
        "emb_out": str(EMB_DIR),
        "two_scale_ids": len(TWO_SCALE_IDS),
        "sets": {k: len(v) for k,v in SLIDESETS.items()}
    }, indent=2))

    # Filter IDs by dataset if requested
    target_ids = []
    for sid in TWO_SCALE_IDS:
        ds = dataset_of(sid)
        if CONFIG["only_datasets"] is None or ds in CONFIG["only_datasets"]:
            target_ids.append(sid)

    print(f"[PLAN] Slides with 2-scale features in target sets: {len(target_ids)}")
    if len(target_ids) == 0:
        print("[EXIT] Nothing to export for chosen sets.")
        return

    # Export with threads
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0=time.time(); done=0; ok=0; bad=0
    rows=[]
    print_every = max(1, CONFIG["print_every"])
    with ThreadPoolExecutor(max_workers=CONFIG["workers"]) as ex:
        futs = {ex.submit(embed_one, sid): sid for sid in target_ids}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            ok += int(r.get("ok", False))
            bad += int(not r.get("ok", False))
            if (done % print_every)==0:
                dt = time.time() - t0
                sps = done / max(1e-6, dt)
                print(f"[{done:6d}/{len(target_ids)}] ok={ok} bad={bad}  {sps:.2f} slides/s")

    # Save per-dataset indices
    df = pd.DataFrame(rows)
    df["dataset"] = df["dataset"].fillna("UNKNOWN")
    for ds, g in df[df["ok"]==True].groupby("dataset"):
        out_csv = EMB_DIR / f"{ds.lower()}_index.csv"
        g[["slide_id","path_emb","t05","t20","norm"]].to_csv(out_csv, index=False)
        print(f"[OK] Index for {ds}: {len(g)} → {out_csv}")

    # Diagnostics
    diag = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "total_attempted": int(len(target_ids)),
        "ok": int((df["ok"]==True).sum()) if len(df) else 0,
        "bad": int((df["ok"]==False).sum()) if len(df) else 0,
        "by_dataset": df.groupby("dataset")["ok"].sum().to_dict() if len(df) else {},
        "examples_bad": df[df["ok"]==False].head(10).to_dict(orient="records"),
    }
    (DIAG / "script8_embeddings_summary.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    (DIAG / "script8_embeddings_summary.txt").write_text(
        "\n".join([f"{k}: {v}" for k,v in diag.items()]), encoding="utf-8")

    print("\n== Summary ==")
    print(json.dumps(diag, indent=2))
    print("\n[DONE] Script 8 complete.")
    print(f"Embeddings dir: {EMB_DIR}")

if __name__ == "__main__":
    main()


# SCRIPT 09: CAMELYON17 PN STAGING (LOCO-CV)
# Script 9 — CAMELYON17 pN (κ) ablation (LOCO)
import os, sys, re, json, math, time, subprocess, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# WORKSPACE — set in CONFIGURATION block above
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

# Run
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


# SCRIPT 10: PANDA FEATURE PROCESSING
# PANDA Processing Pipeline
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

# Configuration
# PANDA_ROOT — set in CONFIGURATION block above
# WORKSPACE — set in CONFIGURATION block above
OUTPUT_DIRS = {
    "features_05": WORKSPACE / "features" / "panda" / "scale0p5",
    "features_20": WORKSPACE / "features" / "panda" / "scale2p0", 
    "results": WORKSPACE / "results" / "panda",
    "logs": WORKSPACE / "logs" / "panda"
}
for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Optimization settings
N_WORKERS = min(cpu_count() - 1, 8)  # Leave one CPU free
BATCH_SIZE = 128  # Increased batch size
PREFETCH_TILES = 4  # Prefetch multiple tiles
USE_MIXED_PRECISION = True
CACHE_SIZE = 1000  # Cache recent tiles in memory

print(f"System info: {cpu_count()} CPUs available, using {N_WORKERS} workers")

def check_already_processed(image_id):
    """Quick check if slide is already processed"""
    feat_05 = OUTPUT_DIRS["features_05"] / f"{image_id}.npy"
    feat_20 = OUTPUT_DIRS["features_20"] / f"{image_id}.npy"
    
    if feat_05.exists() and feat_20.exists():
        # Verify files are valid
        try:
            f05 = np.load(feat_05, mmap_mode='r')
            f20 = np.load(feat_20, mmap_mode='r')
            if f05.shape[1] == 768 and f20.shape[1] == 768:
                return True
        except:
            # Corrupted files, will reprocess
            pass
    return False

def get_pending_slides(df, max_slides=None):
    """Get list of slides that need processing"""
    pending = []
    
    for idx, row in df.iterrows():
        if not row['image_exists']:
            continue
            
        image_id = row['image_id']
        
        # Skip if already processed
        if check_already_processed(image_id):
            continue
        
        pending.append(row)
        
        if max_slides and len(pending) >= max_slides:
            break
    
    return pending

def process_single_slide(args):
    """Process a single slide - can be run in parallel"""
    row, device_id = args
    
    # Import heavy libraries only in worker process
    import torch
    import torchvision.models as tvm
    import torch.nn as nn
    from PIL import Image
    import openslide
    
    # Set device for this worker
    if torch.cuda.is_available():
        device = f"cuda:{device_id % torch.cuda.device_count()}"
    else:
        device = "cpu"
    
    image_id = row['image_id']
    image_path = Path(row['image_path'])
    
    # Double-check if already processed
    if check_already_processed(image_id):
        return image_id, "skipped", 0
    
    # Build model
    class ConvNeXtTinyFeats(nn.Module):
        def __init__(self):
            super().__init__()
            weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT
            model = tvm.convnext_tiny(weights=weights)
            self.features = model.features
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.eval()
            for p in self.parameters(): 
                p.requires_grad = False
        
        @torch.no_grad()
        def forward(self, x):
            x = self.features(x)
            x = self.gap(x).flatten(1)
            return x
    
    try:
        model = ConvNeXtTinyFeats().to(device)
        if device != "cpu":
            model = model.to(memory_format=torch.channels_last)
        
        # Open slide
        slide = openslide.OpenSlide(str(image_path))
        
        # Configuration
        TILE_SIZE = 256
        STRIDE = 224
        SCALES = [0.5, 2.0]
        MAX_TILES = {0.5: 1200, 2.0: 400}
        
        tiles_extracted = 0
        
        # Process each scale
        for scale in SCALES:
            scale_dir = OUTPUT_DIRS[f"features_{str(scale).replace('.','').replace('p','')}"]
            feat_path = scale_dir / f"{image_id}.npy"
            
            if feat_path.exists():
                continue
            
            # Determine level
            base_mpp = 0.5
            target_downsample = scale / base_mpp
            level = slide.get_best_level_for_downsample(target_downsample)
            actual_downsample = slide.level_downsamples[level]
            
            # Get dimensions
            level_w, level_h = slide.level_dimensions[level]
            
            # Collect tiles efficiently
            tiles = []
            tile_batch = []
            
            for y in range(0, level_h - TILE_SIZE + 1, STRIDE):
                for x in range(0, level_w - TILE_SIZE + 1, STRIDE):
                    if len(tiles) >= MAX_TILES[scale]:
                        break
                    
                    # Read tile
                    x0 = int(x * actual_downsample)
                    y0 = int(y * actual_downsample)
                    tile = slide.read_region((x0, y0), level, (TILE_SIZE, TILE_SIZE)).convert('RGB')
                    
                    # Quick tissue check
                    tile_np = np.array(tile)
                    if tile_np.mean() < 235 and tile_np.std() > 15:
                        # Resize immediately
                        tile_224 = tile.resize((224, 224), Image.BILINEAR)
                        tile_batch.append(tile_224)
                        
                        # Process batch when full
                        if len(tile_batch) >= BATCH_SIZE:
                            batch_features = process_batch(tile_batch, model, device)
                            tiles.extend(batch_features)
                            tile_batch = []
                            tiles_extracted += len(batch_features)
                
                if len(tiles) >= MAX_TILES[scale]:
                    break
            
            # Process remaining tiles
            if tile_batch:
                batch_features = process_batch(tile_batch, model, device)
                tiles.extend(batch_features)
                tiles_extracted += len(batch_features)
            
            # Save features
            if tiles:
                all_features = np.vstack(tiles).astype(np.float16)
                np.save(feat_path, all_features)
            else:
                np.save(feat_path, np.zeros((0, 768), dtype=np.float16))
        
        slide.close()
        
        # Clean up GPU memory
        if device != "cpu":
            torch.cuda.empty_cache()
        
        return image_id, "success", tiles_extracted
        
    except Exception as e:
        return image_id, f"error: {str(e)}", 0

def process_batch(tile_batch, model, device):
    """Process a batch of tiles through the model"""
    import torch
    
    # Convert tiles to tensors
    tensors = []
    for tile in tile_batch:
        tile_array = np.array(tile).astype(np.float32) / 255.0
        tensor = torch.from_numpy(tile_array).permute(2, 0, 1)
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensors.append(tensor)
    
    # Batch inference
    batch_tensor = torch.stack(tensors).to(device, non_blocking=True)
    if device != "cpu":
        batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
    
    with torch.no_grad():
        if USE_MIXED_PRECISION and device != "cpu":
            with torch.cuda.amp.autocast():
                features = model(batch_tensor)
        else:
            features = model(batch_tensor)
        
        features = features.cpu().numpy()
    
    return features

def extract_features_parallel(df, max_slides=None):
    """Extract features using multiple workers"""
    # Import torch here just for CUDA check
    import torch
    
    print("\n" + "="*80)
    print("PARALLEL FEATURE EXTRACTION")
    print("="*80)
    
    # Get pending slides
    pending_slides = get_pending_slides(df[df['image_exists']], max_slides)
    
    if not pending_slides:
        print("All slides already processed!")
        return
    
    print(f"Found {len(pending_slides)} slides to process")
    print(f"Using {N_WORKERS} parallel workers")
    
    # Check CUDA availability once
    cuda_available = torch.cuda.is_available()
    
    # Prepare arguments for workers
    worker_args = []
    for i, row in enumerate(pending_slides):
        device_id = i % N_WORKERS if cuda_available else 0
        worker_args.append((row, device_id))
    
    # Process in parallel
    results = []
    failed = []
    successful = 0
    skipped = 0
    
    start_time = time.time()
    last_print = start_time
    
    # Use ThreadPoolExecutor for I/O-bound parts, ProcessPoolExecutor for CPU-bound
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_single_slide, args): args[0]['image_id'] 
                  for args in worker_args}
        
        for future in as_completed(futures):
            image_id = futures[future]
            try:
                slide_id, status, tiles = future.result()
                
                if status == "success":
                    successful += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed.append((slide_id, status))
                
                # Progress update
                current_time = time.time()
                if current_time - last_print > 5:  # Print every 5 seconds
                    elapsed = current_time - start_time
                    processed = successful + skipped + len(failed)
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (len(pending_slides) - processed) / rate if rate > 0 else 0
                    
                    print(f"Progress: {processed}/{len(pending_slides)} | "
                          f"Rate: {rate:.2f} slides/sec | "
                          f"ETA: {eta/60:.1f} min")
                    last_print = current_time
                    
            except Exception as e:
                failed.append((image_id, str(e)))
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"Extraction complete in {elapsed/60:.1f} minutes")
    print(f"Successful: {successful}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {len(failed)}")
    print(f"Average: {successful/elapsed:.2f} slides/sec")
    
    if failed:
        failed_df = pd.DataFrame(failed, columns=['image_id', 'error'])
        failed_df.to_csv(OUTPUT_DIRS["logs"] / "failed_extractions.csv", index=False)

def main_optimized():
    """Optimized main pipeline"""
    import torch
    
    print("="*80)
    print("OPTIMIZED PANDA PROCESSING PIPELINE")
    print(f"Workers: {N_WORKERS} | Batch size: {BATCH_SIZE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print("="*80)
    
    # Load manifest
    manifest_path = OUTPUT_DIRS["logs"] / "panda_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
    else:
        # Create manifest
        train_csv = PANDA_ROOT / "train.csv"
        if not train_csv.exists():
            print("ERROR: train.csv not found!")
            return
        
        df = pd.read_csv(train_csv)
        df['image_path'] = df['image_id'].apply(
            lambda x: str(PANDA_ROOT / "train_images" / f"{x}.tiff")
        )
        df['image_exists'] = df['image_path'].apply(lambda x: Path(x).exists())
        df.to_csv(manifest_path, index=False)
    
    print(f"Total slides in dataset: {len(df)}")
    print(f"Slides with images: {df['image_exists'].sum()}")
    
    # Check already processed
    already_done = sum(1 for _, row in df.iterrows() 
                      if row['image_exists'] and check_already_processed(row['image_id']))
    print(f"Already processed: {already_done}")
    
    # Options
    print("\n" + "="*60)
    print("EXTRACTION OPTIONS:")
    print("="*60)
    print(f"1. Quick test (10 slides)")
    print(f"2. Small batch (100 slides)")
    print(f"3. Medium batch (1000 slides)")
    print(f"4. Large batch (5000 slides)")
    print(f"5. Full dataset (all {df['image_exists'].sum()} slides)")
    print(f"6. Skip extraction")
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == "6":
        print("Skipping extraction")
        return
    
    max_slides_map = {
        "1": 10,
        "2": 100,
        "3": 1000,
        "4": 5000,
        "5": None
    }
    max_slides = max_slides_map.get(choice, 100)
    
    # Run extraction
    extract_features_parallel(df, max_slides)
    
    print("\nDone! Features saved to:")
    print(f"  {OUTPUT_DIRS['features_05']}")
    print(f"  {OUTPUT_DIRS['features_20']}")

if __name__ == "__main__":
    main_optimized()

# SCRIPT 11: PANDA GLEASON GRADING (MIL)
#PANDA Gleason Grading 
import os, sys, json, time, math, platform, random, tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Utils
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

# Config
@dataclass
class Config:
    # Paths
    WORKSPACE: Path = WORKSPACE
    PANDA_ROOT: Path = WORKSPACE / "Validation Data" / "PANDA"
    FEAT_05: Path = WORKSPACE / "features" / "panda" / "scale0p5"
    FEAT_20: Path = WORKSPACE / "features" / "panda" / "scale2p0"
    OUTPUT:  Path = WORKSPACE / "results" / "panda_mil_088"

    # Data/model dims
    INPUT_DIM: int = 768
    NUM_CLASSES: int = 6

    # Pooling & selection
    NUM_HEADS: int = 8
    POOL_DROPOUT: float = 0.1
    TOPK_RATIO: float = 0.15     # select 15% tokens
    TOPK_MIN: int = 24           # at least 24 tokens per bag

    # Fusion & head
    HIDDEN_DIM: int = 512
    FUSE_DROPOUT: float = 0.15

    # Training
    EPOCHS: int = 30
    WARMUP_EPOCHS: int = 2
    BATCH_SLIDES: int = 24
    LR: float = 3e-4
    WD: float = 1e-4
    MAX_GRAD_NORM: float = 1.0
    AMP: bool = True
    NUM_WORKERS: int = 0 if platform.system()=="Windows" else max(4, (os.cpu_count() or 8)-2)
    PIN_MEMORY: bool = torch.cuda.is_available()
    PROVIDER_AWARE: bool = True
    PATIENCE: int = 7

    # Loss
    LABEL_SMOOTH: float = 0.05
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    ORDINAL_LAM: float = 0.25      # |argmax - y|
    EXP_LAM: float = 0.02           # (E[class] - y)^2

    # Aug
    NOISE_STD: float = 0.01
    FEAT_DROPOUT_P: float = 0.05

    # Feature budgets
    MAX_TILES_05: int = 1000
    MAX_TILES_20: int = 350

    # CV & seeds
    N_FOLDS: int = 5
    SEEDS: Tuple[int,...] = (42, 777, 1337)

    # EMA
    EMA_DECAY: float = 0.999

    # Misc
    PRINT_EVERY: int = 25

    def __post_init__(self):
        (self.OUTPUT / "models").mkdir(parents=True, exist_ok=True)
        (self.OUTPUT / "logs").mkdir(parents=True, exist_ok=True)

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE.type=="cuda" else "CPU"

# Data
class PANDADatasetIndex:
    def __init__(self, cfg: Config):
        df = pd.read_csv(cfg.PANDA_ROOT / "train.csv")
        df["isup_grade"] = df["isup_grade"].fillna(0).astype(int)
        has05 = df["image_id"].apply(lambda s: (cfg.FEAT_05 / f"{s}.npy").exists())
        has20 = df["image_id"].apply(lambda s: (cfg.FEAT_20 / f"{s}.npy").exists())
        df = df[has05 & has20].reset_index(drop=True)
        self.df = df
        print(f"[DATA] Slides (both scales): {len(df)}")
        print(f"[DATA] ISUP distribution:\n{df['isup_grade'].value_counts().sort_index()}")

    def splits(self):
        n = len(self.df)
        if CFG.PROVIDER_AWARE and "data_provider" in self.df.columns:
            key = self.df["isup_grade"].astype(str) + "_" + self.df["data_provider"].astype(str)
        else:
            key = self.df["isup_grade"]
        skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
        return list(skf.split(np.arange(n), key))

class SlideBagDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self): return len(self.df)

    def _load(self, root: Path, sid: str, budget: int, train: bool):
        arr = np.load(root / f"{sid}.npy", mmap_mode="r")  # [T, D] float32/16
        if arr.ndim != 2 or arr.shape[1] != CFG.INPUT_DIM:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1, CFG.INPUT_DIM)
        T = arr.shape[0]
        if T == 0:
            arr = np.zeros((1, CFG.INPUT_DIM), dtype=np.float32); T = 1
        if T > budget:
            if train:
                idx = np.random.choice(T, budget, replace=False)
                arr = arr[idx]
            else:
                arr = arr[:budget]
        arr = np.asarray(arr, dtype=np.float32).copy(order="C")
        return arr

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        sid = r["image_id"]; y = int(r["isup_grade"])
        f05 = self._load(CFG.FEAT_05, sid, CFG.MAX_TILES_05, self.train)
        f20 = self._load(CFG.FEAT_20, sid, CFG.MAX_TILES_20, self.train)
        if self.train:
            if np.random.rand() < 0.4:
                f05 += np.random.normal(0, CFG.NOISE_STD, f05.shape).astype(np.float32)
                f20 += np.random.normal(0, CFG.NOISE_STD, f20.shape).astype(np.float32)
            if np.random.rand() < 0.3:
                mask05 = (np.random.rand(*f05.shape) > CFG.FEAT_DROPOUT_P).astype(np.float32)
                mask20 = (np.random.rand(*f20.shape) > CFG.FEAT_DROPOUT_P).astype(np.float32)
                f05 *= mask05; f20 *= mask20
        return {
            "feat_05": torch.from_numpy(f05),
            "feat_20": torch.from_numpy(f20),
            "label": torch.tensor(y, dtype=torch.long),
            "id": sid,
            "prov": r.get("data_provider", "NA")
        }

def collate_bags(batch):
    B = len(batch); D = batch[0]["feat_05"].shape[1]
    n1 = [b["feat_05"].shape[0] for b in batch]
    n2 = [b["feat_20"].shape[0] for b in batch]
    N1, N2 = max(n1), max(n2)
    f05 = torch.zeros(B, N1, D, dtype=torch.float32)
    f20 = torch.zeros(B, N2, D, dtype=torch.float32)
    m05 = torch.ones(B, N1, dtype=torch.bool)   # True = pad
    m20 = torch.ones(B, N2, dtype=torch.bool)
    for i,b in enumerate(batch):
        a = b["feat_05"]; f05[i,:a.size(0)] = a; m05[i,:a.size(0)] = False
        c = b["feat_20"]; f20[i,:c.size(0)] = c; m20[i,:c.size(0)] = False
    y   = torch.stack([b["label"] for b in batch],0)
    ids = [b["id"] for b in batch]
    prov= [b["prov"] for b in batch]
    return {"feat_05":f05, "mask_05":m05,
            "feat_20":f20, "mask_20":m20,
            "label":y, "ids":ids, "prov":prov}

# Model
class MultiHeadPool(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pdrop: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=pdrop, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(d_model*2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(pdrop)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Force float32 for stability under AMP
        x = x.float();  # [B,N,D]
        key_mask = mask  # [B,N] bool (True=pad)
        # Self-attention (float32)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_mask, need_weights=False)
        h = self.norm1(x + self.dropout(attn_out))
        f = self.ffn(h)
        h = self.norm2(h + self.dropout(f))      # [B,N,D]

        # Score tokens (float32)
        scores = self.scorer(h).squeeze(-1)      # [B,N]
        # Mask out padding using a large negative float32
        scores = scores.masked_fill(key_mask, -1e4)

        # Per-sample top-k
        with torch.no_grad():
            nonpad = (~key_mask).sum(1)  # [B]
            k = (nonpad.float() * CFG.TOPK_RATIO).floor().clamp(min=CFG.TOPK_MIN)
            k = torch.minimum(k, nonpad.float()).to(torch.long)
            k = torch.clamp(k, min=1)

        pooled = []
        for b in range(x.size(0)):
            k_b = int(k[b].item())
            topv, topi = torch.topk(scores[b, :nonpad[b]], k_b, dim=0)
            sel = h[b, topi]                        # [k_b, D]
            w = F.softmax(topv, dim=0).unsqueeze(1) # [k_b,1]
            pooled.append((sel * w).sum(0))
        pooled = torch.stack(pooled, dim=0)         # [B,D]
        return pooled  # float32

class MILModel(nn.Module):
    def __init__(self, in_dim=CFG.INPUT_DIM, num_classes=CFG.NUM_CLASSES):
        super().__init__()
        self.pool05 = MultiHeadPool(in_dim, CFG.NUM_HEADS, CFG.POOL_DROPOUT)
        self.pool20 = MultiHeadPool(in_dim, CFG.NUM_HEADS, CFG.POOL_DROPOUT)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim*2, CFG.HIDDEN_DIM),
            nn.LayerNorm(CFG.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(CFG.FUSE_DROPOUT),
            nn.Linear(CFG.HIDDEN_DIM, CFG.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(CFG.FUSE_DROPOUT),
        )
        self.classifier = nn.Linear(CFG.HIDDEN_DIM, num_classes)

    def forward(self, f05, m05, f20, m20):
        e05 = self.pool05(f05, m05)   # [B,D]
        e20 = self.pool20(f20, m20)   # [B,D]
        h   = self.fuse(torch.cat([e05, e20], dim=1))  # [B,H]
        logits = self.classifier(h).float()            # [B,K]
        return {"logits": logits, "emb": h}

# Loss
def smooth_one_hot(y: torch.Tensor, num_classes: int, eps: float):
    with torch.no_grad():
        target = torch.full((y.size(0), num_classes), eps/num_classes, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0 - eps + eps/num_classes)
    return target

def focal_ce_loss(logits, targets, class_weights=None, alpha=0.25, gamma=2.0, label_smooth=0.05):
    # CE with label smoothing + focal modulation
    K = logits.size(1)
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    T = smooth_one_hot(targets, K, label_smooth)
    pt = (p * T).sum(dim=1).clamp(min=1e-8)
    ce = -(T * logp).sum(dim=1)
    focal = (alpha * (1 - pt)**gamma) * ce
    if class_weights is not None:
        w = class_weights[targets]  # [B]
        focal = focal * w
    return focal.mean()

def ordinal_penalties(probs, targets):
    # |argmax - y| and (E[class]-y)^2
    pred = probs.argmax(dim=1)
    dist = (pred - targets).abs().float()
    expc = (probs * torch.arange(CFG.NUM_CLASSES, device=probs.device).float()).sum(dim=1)
    mse = (expc - targets.float())**2
    return dist.mean(), mse.mean()

# EMA
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = {}

# Trainer
class Trainer:
    def __init__(self, in_dim, fold, seed, class_weights: torch.Tensor):
        self.fold = fold; self.seed = seed
        self.model = MILModel(in_dim, CFG.NUM_CLASSES).to(DEVICE)
        self.ema = EMA(self.model, decay=CFG.EMA_DECAY)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=CFG.LR, weight_decay=CFG.WD, betas=(0.9, 0.999))
        self.scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP)
        self.best_kappa = -1.0; self.epoch = 0; self.no_improve = 0
        self.cw = class_weights.to(DEVICE) if class_weights is not None else None

        # Schedulers
        self.lr_sched_warm = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=0.1, total_iters=CFG.WARMUP_EPOCHS)
        self.lr_sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(1, CFG.EPOCHS-CFG.WARMUP_EPOCHS), eta_min=1e-6)

        self.ckpt = CFG.OUTPUT / "models" / f"seed{seed}_fold{fold}_best.pth"

        print(f"[Model] params={sum(p.numel() for p in self.model.parameters()):,}")

    def _step_lr(self, ep):
        if ep <= CFG.WARMUP_EPOCHS:
            self.lr_sched_warm.step()
        else:
            self.lr_sched_main.step()

    def run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss = 0.0; y_true=[]; y_pred=[]
        t0 = time.time()
        for i,b in enumerate(loader,1):
            f05 = b["feat_05"].to(DEVICE, non_blocking=True)
            f20 = b["feat_20"].to(DEVICE, non_blocking=True)
            m05 = b["mask_05"].to(DEVICE, non_blocking=True)
            m20 = b["mask_20"].to(DEVICE, non_blocking=True)
            y   = b["label"].to(DEVICE, non_blocking=True)

            if train:
                with torch.cuda.amp.autocast(enabled=CFG.AMP):
                    out = self.model(f05,m05,f20,m20)
                    logits = out["logits"]
                    loss_ce = focal_ce_loss(logits, y, self.cw, alpha=CFG.FOCAL_ALPHA, gamma=CFG.FOCAL_GAMMA, label_smooth=CFG.LABEL_SMOOTH)
                    probs = F.softmax(logits.float(), dim=1)
                    dist_mean, mse_mean = ordinal_penalties(probs, y)
                    loss = loss_ce + CFG.ORDINAL_LAM * dist_mean + CFG.EXP_LAM * mse_mean
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), CFG.MAX_GRAD_NORM)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.ema.update(self.model)
            else:
                # eval with EMA weights
                self.ema.apply_shadow(self.model)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=CFG.AMP):
                    out = self.model(f05,m05,f20,m20)
                    logits = out["logits"]
                    loss_ce = focal_ce_loss(logits, y, self.cw, alpha=CFG.FOCAL_ALPHA, gamma=CFG.FOCAL_GAMMA, label_smooth=0.0)
                    probs = F.softmax(logits.float(), dim=1)
                    dist_mean, mse_mean = ordinal_penalties(probs, y)
                    loss = loss_ce + CFG.ORDINAL_LAM * dist_mean + CFG.EXP_LAM * mse_mean
                self.ema.restore(self.model)

            total_loss += float(loss.detach().item())
            pred = logits.detach().float().softmax(1).argmax(1)
            y_true.extend(y.tolist()); y_pred.extend(pred.tolist())

            if i % CFG.PRINT_EVERY == 0 or i == len(loader):
                rate = i / max(1e-6, (time.time() - t0))
                kappa = qwk(np.array(y_true), np.array(y_pred)) if len(y_true)>8 else 0.0
                phase = "Train" if train else "Val"
                print(f"  [{phase}] {i:4d}/{len(loader)} | {rate:.1f} it/s | Loss {loss.item():.4f} | κ~{kappa:.3f}")

        avg_loss = total_loss / len(loader)
        kappa = qwk(np.array(y_true), np.array(y_pred))
        acc = accuracy_score(np.array(y_true), np.array(y_pred))
        return avg_loss, kappa, acc

    def fit(self, dl_tr: DataLoader, dl_va: DataLoader):
        for ep in range(1, CFG.EPOCHS+1):
            self.epoch = ep
            print(f"\n{'='*79}\nFold {self.fold} | Seed {self.seed} | Epoch {ep}/{CFG.EPOCHS}\n{'='*79}")
            trL, trK, trA = self.run_epoch(dl_tr, train=True)
            vaL, vaK, vaA = self.run_epoch(dl_va, train=False)
            self._step_lr(ep)
            lr = self.opt.param_groups[0]["lr"]
            print(f"[Epoch {ep}] LR {lr:.2e} | Train L={trL:.4f} κ={trK:.4f} Acc={trA:.3f} | Val L={vaL:.4f} κ={vaK:.4f} Acc={vaA:.3f}")

            if vaK > self.best_kappa:
                imp = vaK - self.best_kappa
                self.best_kappa = vaK; self.no_improve = 0
                torch.save({"model": self.model.state_dict(),
                            "ema": self.ema.shadow,
                            "epoch": ep,
                            "kappa": self.best_kappa}, self.ckpt)
                print(f"  ✓ Best improved to κ={self.best_kappa:.4f} (+{imp:.4f})")
            else:
                self.no_improve += 1
                print(f"  No improvement ({self.no_improve}/{CFG.PATIENCE})")
                if self.no_improve >= CFG.PATIENCE:
                    print("  Early stopping")
                    break

    @torch.no_grad()
    def predict_val(self, loader: DataLoader):
        # load best and eval with EMA shadow
        if self.ckpt.exists():
            ck = torch.load(self.ckpt, map_location=DEVICE)
            self.model.load_state_dict(ck["model"])
            # restore EMA shadow
            self.ema.shadow = ck.get("ema", self.ema.shadow)

        self.model.eval()
        self.ema.apply_shadow(self.model)

        preds=[]; probs=[]; labels=[]; ids=[]; provs=[]
        for b in loader:
            f05 = b["feat_05"].to(DEVICE, non_blocking=True)
            f20 = b["feat_20"].to(DEVICE, non_blocking=True)
            m05 = b["mask_05"].to(DEVICE, non_blocking=True)
            m20 = b["mask_20"].to(DEVICE, non_blocking=True)
            y   = b["label"].to(DEVICE, non_blocking=True)
            out = self.model(f05,m05,f20,m20)
            p = F.softmax(out["logits"].float(), dim=1)
            pr = p.cpu().numpy()
            probs.append(pr)
            pred = p.argmax(1).cpu().numpy()
            preds.append(pred)
            labels.extend(y.cpu().tolist())
            ids.extend(b["ids"]); provs.extend(b["prov"])
        self.ema.restore(self.model)
        return {
            "preds": np.concatenate(preds,0),
            "probs": np.concatenate(probs,0),
            "labels": np.array(labels),
            "ids": ids,
            "prov": provs
        }

# Helpers
def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    freq = counts / np.clip(counts.sum(), 1, None)
    inv = 1.0 / np.clip(freq, 1e-6, None)
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)

# Main
def main():
    print("="*79)
    print(" PANDA MIL — MHA+TopK, Focal+Ordinal, EMA, 3-Seed Ensemble ".center(79))
    print("="*79)
    print(f"Start: {now()} | Device: {DEVICE} | GPU: {GPU_NAME}")
    print(f"Paths: FEAT_05={CFG.FEAT_05} | FEAT_20={CFG.FEAT_20}")
    print(f"Compile=False | AMP={CFG.AMP}")
    print("="*79)

    idx = PANDADatasetIndex(CFG)
    splits = idx.splits()
    N = len(idx.df)

    # For ensemble across seeds
    oof_prob_ens = np.zeros((N, CFG.NUM_CLASSES), dtype=np.float32)
    oof_pred_ens = np.zeros(N, dtype=np.int64)
    seed_contribs = []

    for seed in CFG.SEEDS:
        print("\n" + "#"*79)
        print(f"### SEED {seed}".center(79))
        print("#"*79)

        set_seed(seed)
        oof_prob = np.zeros((N, CFG.NUM_CLASSES), dtype=np.float32)
        oof_pred = np.zeros(N, dtype=np.int64)

        for f,(tr,va) in enumerate(splits, start=1):
            print("\n" + "="*79)
            print(f"FOLD {f}/{CFG.N_FOLDS} — seed {seed}".center(79))
            print("="*79)
            df_tr = idx.df.iloc[tr].reset_index(drop=True)
            df_va = idx.df.iloc[va].reset_index(drop=True)

            cw = compute_class_weights(df_tr["isup_grade"].values, CFG.NUM_CLASSES)
            print(f"[Class weights] {cw.numpy()}")

            ds_tr = SlideBagDataset(df_tr, train=True)
            ds_va = SlideBagDataset(df_va, train=False)
            dl_tr = DataLoader(ds_tr, batch_size=CFG.BATCH_SLIDES, shuffle=True,
                               num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
                               collate_fn=collate_bags)
            dl_va = DataLoader(ds_va, batch_size=CFG.BATCH_SLIDES, shuffle=False,
                               num_workers=CFG.NUM_WORKERS, pin_memory=CFG.PIN_MEMORY,
                               collate_fn=collate_bags)

            trainer = Trainer(CFG.INPUT_DIM, f, seed, cw)
            trainer.fit(dl_tr, dl_va)
            out = trainer.predict_val(dl_va)

            kappa = qwk(out["labels"], out["preds"])
            acc   = accuracy_score(out["labels"], out["preds"])
            print(f"\n[FOLD {f}] κ={kappa:.4f} | Acc={acc:.3f}")

            oof_prob[va] = out["probs"]
            oof_pred[va] = out["preds"]

            # free
            del ds_tr, ds_va, dl_tr, dl_va, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # seed-level OOF
        y_true = idx.df["isup_grade"].values
        kappa_seed = qwk(y_true, oof_pred)
        acc_seed   = accuracy_score(y_true, oof_pred)
        print("\n" + "-"*79)
        print(f"OOF (seed {seed}) — κ={kappa_seed:.4f} | Acc={acc_seed:.4f}")
        print("-"*79)

        # Save seed oof for analysis
        pd.DataFrame({
            "image_id": idx.df["image_id"],
            "true_isup": y_true,
            "pred_isup": oof_pred,
            **{f"prob_{i}": oof_prob[:,i] for i in range(CFG.NUM_CLASSES)}
        }).to_csv(CFG.OUTPUT / f"oof_seed{seed}.csv", index=False)

        seed_contribs.append({"seed": seed, "kappa": float(kappa_seed), "acc": float(acc_seed)})
        oof_prob_ens += oof_prob / float(len(CFG.SEEDS))

    # Ensemble OOF by averaging probs across seeds
    oof_pred_ens = oof_prob_ens.argmax(1)
    y_true = idx.df["isup_grade"].values
    kappa_oof = qwk(y_true, oof_pred_ens)
    acc_oof = accuracy_score(y_true, oof_pred_ens)
    cm = confusion_matrix(y_true, oof_pred_ens, labels=list(range(CFG.NUM_CLASSES)))

    print("\n" + "="*79)
    print(" FINAL OOF RESULTS (SEED ENSEMBLE) ".center(79))
    print("="*79)
    print(f"OOF κ_qw: {kappa_oof:.4f}")
    print(f"OOF Acc : {acc_oof:.4f}")
    print("Confusion matrix:\n", cm)

    # provider-wise
    if "data_provider" in idx.df.columns:
        for prov in sorted(idx.df["data_provider"].unique()):
            m = (idx.df["data_provider"]==prov).values
            if m.sum()>0:
                kp = qwk(y_true[m], oof_pred_ens[m]); ap = accuracy_score(y_true[m], oof_pred_ens[m])
                print(f"  {prov:<10} | n={m.sum():>4} | κ={kp:.4f} | acc={ap:.4f}")

    # Save ensemble artifacts
    pd.DataFrame({
        "image_id": idx.df["image_id"],
        "true_isup": y_true,
        "pred_isup": oof_pred_ens,
        **{f"prob_{i}": oof_prob_ens[:,i] for i in range(CFG.NUM_CLASSES)}
    }).to_csv(CFG.OUTPUT / "oof_ensemble.csv", index=False)

    # summary
    with open(CFG.OUTPUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": now(),
            "device": str(DEVICE),
            "gpu": GPU_NAME,
            "config": {
                "epochs": CFG.EPOCHS,
                "batch_slides": CFG.BATCH_SLIDES,
                "lr": CFG.LR,
                "wd": CFG.WD,
                "num_heads": CFG.NUM_HEADS,
                "topk_ratio": CFG.TOPK_RATIO,
                "topk_min": CFG.TOPK_MIN,
                "ema_decay": CFG.EMA_DECAY,
                "label_smooth": CFG.LABEL_SMOOTH,
                "focal_alpha": CFG.FOCAL_ALPHA,
                "focal_gamma": CFG.FOCAL_GAMMA,
                "ordinal_lam": CFG.ORDINAL_LAM,
                "exp_lam": CFG.EXP_LAM,
                "seeds": CFG.SEEDS
            },
            "oof": {"kappa": float(kappa_oof), "accuracy": float(acc_oof)},
            "seed_contribs": seed_contribs,
            "class_distribution": pd.Series(y_true).value_counts().sort_index().to_dict()
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Complete. Artifacts saved to {CFG.OUTPUT}")

if __name__ == "__main__":
    main()


# SCRIPT 12: PANDA OUT-OF-FOLD METRICS
# metrics_from_oof.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# === EDIT THIS TO THE RUN YOU CARE ABOUT ===
RESULTS_DIR = WORKSPACE / "results" / "panda_mil"
OOF_CSV = next((p for p in [
    RESULTS_DIR / "oof_predictions.csv",
    RESULTS_DIR / "oof.csv"
] if p.exists()), None)
assert OOF_CSV and OOF_CSV.exists(), f"Missing OOF file in {RESULTS_DIR}"

df = pd.read_csv(OOF_CSV)
assert "true_isup" in df.columns, "true_isup column not found"

y_true = df["true_isup"].astype(int).values
num_classes = int(max(y_true.max(), 5) + 1)  # expect 6 for PANDA

# get probabilities (prob_* or logit_* -> softmax)
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
