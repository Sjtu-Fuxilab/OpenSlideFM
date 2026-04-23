# OpenSlideFM — Complete Pipeline
# Pipeline:
#   Script 01: Environment setup & paths
#   Script 02: Dataset manifest & provenance
#   Script 03: Quality control & tissue masking
#   Script 04: Two-scale tiling & token budget
#   Script 05: Feature extraction (ConvNeXt-Tiny, frozen)
#   Script 06: Self-supervised pre-training (BYOL + MFR)
#   Script 06C: Post-pretraining diagnostics
#   Script 07: TCGA 31-class pan-cancer evaluation
#   Script 08: Slide embeddings export
#   Script 09: CAMELYON17 pN staging (LOCO-CV)
#   Script 09A: CAMELYON16 metastasis detection (5-fold CV)
#   Script 10: PANDA feature processing
#   Script 11: PANDA Gleason grading (MIL)
#   Script 12: PANDA out-of-fold metrics
# Usage:
#   1. Set environment variables:
#        export WORKSPACE=/path/to/workspace
#        export WSI_ROOT=/path/to/wsi/slides
#        export PANDA_ROOT=/path/to/panda/data    (optional)
#   2. Run scripts sequentially or use as importable modules.

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

try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise RuntimeError("Please install pandas and numpy (pip install pandas numpy)") from e

import matplotlib
import matplotlib.pyplot as plt

assert 'WSI_ROOT' in globals() and 'WORKSPACE' in globals() and 'SUBDIRS' in globals(), \
    "Please run Script 1 first to define WSI_ROOT/WORKSPACE/SUBDIRS."

MANIFEST_OUT = SUBDIRS["manifests"] / "manifest_tcga.parquet"
MANIFEST_CSV = SUBDIRS["manifests"] / "manifest_tcga.csv"
FAILED_CSV   = SUBDIRS["manifests"] / "failed_slides.csv"
HASH_INDEX   = SUBDIRS["hashes"]   / "hash_index_tcga.csv"
CHECKSUM_MODE = "sha1_quick"
MAX_WORKERS = min(12, (os.cpu_count() or 8))

def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def file_times(p: Path):
    st = p.stat()
    created = datetime.datetime.fromtimestamp(getattr(st, "st_ctime", st.st_mtime)).strftime("%Y-%m-%d %H:%M:%S")
    modified = datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return created, modified

def quick_fingerprint(path: Path, mode="sha1_quick", chunk=8*1024*1024):
    size = path.stat().st_size
    if mode == "size_only":
        return f"SIZE:{size}", None
    if mode == "sha1_quick":
        h = hashlib.sha1()
        with path.open("rb") as f:
            h.update(f.read(chunk))
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
    return sorted(set(out))

def cancer_code_from_path(p: Path, root: Path):
    rel = p.relative_to(root)
    return rel.parts[0] if len(rel.parts) >= 2 else "UNKNOWN"

def open_and_probe(path: Path):
    import openslide
    slide = openslide.OpenSlide(str(path))
    props = slide.properties
    width, height = slide.dimensions
    level_count = slide.level_count
    mpp_x = props.get('openslide.mpp-x') or props.get('aperio.MPP') or None
    mpp_y = props.get('openslide.mpp-y') or props.get('aperio.MPP') or None
    vendor = props.get('openslide.vendor') or 'unknown'
    obj_pow = props.get('aperio.AppMag') or props.get('openslide.objective-power') or None
    slide.close()
    return {
        "width": int(width), "height": int(height), "level_count": int(level_count),
        "mpp_x": float(mpp_x) if mpp_x not in (None, "") else None,
        "mpp_y": float(mpp_y) if mpp_y not in (None, "") else None,
        "vendor": str(vendor),
        "objective_power": float(obj_pow) if (obj_pow is not None and str(obj_pow).replace('.','',1).isdigit()) else str(obj_pow) if obj_pow else None,
    }

start = time.time()
print(f"== OP_FM Script 2: Manifest & Provenance ==\n[{now_iso()}] Scanning WSI root (read-only): {WSI_ROOT}")

slides = list_wsi_files(WSI_ROOT)
n_total = len(slides)
print(f"[INFO] Found {n_total} candidate WSI files.")

records = []
failures = []

def process_one(path: Path):
    rec = {
        "path": str(path), "filename": path.name, "slide_id": path.stem,
        "cancer_code": cancer_code_from_path(path, WSI_ROOT),
        "size_bytes": path.stat().st_size,
    }
    created, modified = file_times(path)
    rec["created_time"] = created
    rec["modified_time"] = modified
    try:
        fp, sha1_full = quick_fingerprint(path, mode=CHECKSUM_MODE)
        rec["fingerprint"] = fp
        rec["sha1_full"] = sha1_full
    except Exception as e:
        rec["fingerprint"] = None
        rec["sha1_full"] = None
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
        now = time.time()
        if now - last_print > 2 or done == n_total:
            rate = done / (now - t0 + 1e-9)
            print(f"[SCAN] {done}/{n_total} ({rate:.1f} files/s)")
            last_print = now

elapsed_scan = time.time() - start
print(f"[OK] Scanned {n_total} slides in {elapsed_scan/60:.1f} min.")

df = pd.DataFrame.from_records(records)
num_cols = ["size_bytes", "width", "height", "level_count", "mpp_x", "mpp_y"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

SUBDIRS["manifests"].mkdir(parents=True, exist_ok=True)
df.to_parquet(MANIFEST_OUT, index=False)
df.to_csv(MANIFEST_CSV, index=False, encoding="utf-8-sig")
print(f"[OK] Manifest saved:\n - {MANIFEST_OUT}\n - {MANIFEST_CSV}")

if failures:
    pd.DataFrame(failures).to_csv(FAILED_CSV, index=False, encoding="utf-8-sig")
    print(f"[WARN] {len(failures)} slides failed to open; see {FAILED_CSV}")

pd.DataFrame(df[["path", "size_bytes", "fingerprint"]]).to_csv(HASH_INDEX, index=False, encoding="utf-8-sig")
print(f"[OK] Hash index written: {HASH_INDEX}")

print("\n== Diagnostics (Manifest) ==")
total_bytes = df["size_bytes"].sum(skipna=True)
print(f" Total slides: {len(df):,}")
print(f" Total size  : {total_bytes / (1024**3):.2f} GB")
by_cancer = df["cancer_code"].value_counts(dropna=False)
print("\n Slides by cancer_code (top 20):")
print(by_cancer.head(20).to_string())

missing_mpp = df[(df["mpp_x"].isna()) | (df["mpp_y"].isna())]
print(f"\n Missing MPP entries: {len(missing_mpp)}")

dup_key = "fingerprint" if df["fingerprint"].notna().any() else None
if dup_key:
    dup_groups = df.groupby(dup_key).size().sort_values(ascending=False)
    dup_groups = dup_groups[dup_groups > 1]
    print(f"\n Potential duplicates by {dup_key}: {int(dup_groups.sum()) - len(dup_groups)} extra files in {len(dup_groups)} groups")
else:
    size_groups = df.groupby(["size_bytes", "filename"]).size().sort_values(ascending=False)
    size_groups = size_groups[size_groups > 1]
    print(f"\n Potential duplicates by (size, filename): {int(size_groups.sum()) - len(size_groups)} extra files in {len(size_groups)} groups")

topN = df.sort_values("size_bytes", ascending=False).head(10)[["filename", "cancer_code", "size_bytes"]].copy()
topN["size_gb"] = topN["size_bytes"] / (1024**3)
print("\n Top-10 largest WSIs (GB):")
print(topN[["filename", "cancer_code", "size_gb"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

fig_dir = SUBDIRS["figures"]
fig_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8,5))
sizes_gb = (df["size_bytes"] / (1024**3)).dropna()
plt.hist(sizes_gb.values, bins=40)
plt.xlabel("Slide size (GB)"); plt.ylabel("Count"); plt.title("WSI Size Distribution (TCGA)")
plt.tight_layout()
p1 = fig_dir / "manifest_size_distribution.png"; plt.savefig(p1); plt.close()

plt.figure(figsize=(8,5))
wh = df[["width", "height"]].dropna()
vals = np.log10(wh.values.clip(min=1))
plt.hist(vals.flatten(), bins=40)
plt.xlabel("log10(pixels)"); plt.ylabel("Count"); plt.title("WSI Width/Height Distribution (log10)")
plt.tight_layout()
p2 = fig_dir / "manifest_wh_log_distribution.png"; plt.savefig(p2); plt.close()

plt.figure(figsize=(10,6))
top_codes = by_cancer.head(30)
plt.bar(top_codes.index.astype(str), top_codes.values)
plt.xticks(rotation=80, ha="right"); plt.ylabel("Slides"); plt.title("Slides per cancer_code (Top 30)")
plt.tight_layout()
p3 = fig_dir / "manifest_counts_by_cancer.png"; plt.savefig(p3); plt.close()

mpp_complete = df["mpp_x"].notna() & df["mpp_y"].notna()
pct_mpp = 100.0 * mpp_complete.mean()
plt.figure(figsize=(4,4))
plt.bar(["MPP complete", "MPP missing"], [pct_mpp, 100.0 - pct_mpp])
plt.title("MPP Availability (%)")
plt.tight_layout()
p4 = fig_dir / "manifest_mpp_availability.png"; plt.savefig(p4); plt.close()

compute_path = SUBDIRS["compute"] / "compute_passport.json"
try:
    with compute_path.open("r", encoding="utf-8") as f:
        cp = json.load(f)
except Exception:
    cp = {"stages": []}
stage_entry = {
    "stage": "manifest_tcga", "timestamp": now_iso(),
    "inputs": {"wsi_root": str(WSI_ROOT)},
    "outputs": {
        "manifest_parquet": str(MANIFEST_OUT), "manifest_csv": str(MANIFEST_CSV),
        "failed_csv": str(FAILED_CSV) if failures else None,
        "hash_index_csv": str(HASH_INDEX), "figures": [str(p1), str(p2), str(p3), str(p4)],
    },
    "stats": {
        "n_files_found": int(n_total), "n_records": int(len(df)), "n_failures": int(len(failures)),
        "total_gb": float(total_bytes / (1024**3)), "elapsed_minutes": float(elapsed_scan / 60.0),
        "checksum_mode": CHECKSUM_MODE,
    }
}
cp.setdefault("stages", []).append(stage_entry)
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

try:
    import openslide
except Exception as e:
    raise RuntimeError("openslide-python is required for Script 3. Install and rerun.") from e

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

QC_MAX_SLIDES = None
THUMB_MAX_SIDE = 1024
MIN_TISSUE_PCT   = 0.10
MAX_WHITE_PCT    = 0.75
MIN_BLUR_VAR     = 15.0
MAX_PEN_PCT      = 0.02
HSV_S_TISSUE_MIN = 20
HSV_V_WHITE_MIN  = 230
HSV_S_WHITE_MAX  = 30
HSV_H_BLUE_MIN   = 170
HSV_H_BLUE_MAX   = 255
HSV_S_PEN_MIN    = 60
MAX_WORKERS = min(8, (os.cpu_count() or 8))

def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_thumbnail(slide_path: Path, max_side: int = 1024) -> Image.Image:
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.dimensions
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    tw, th = int(w / scale), int(h / scale)
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    slide.close()
    return thumb

def to_hsv_np(img_rgb: Image.Image):
    hsv = img_rgb.convert("HSV")
    a = np.array(hsv, dtype=np.uint8)
    return a[..., 0], a[..., 1], a[..., 2]

def laplacian_var(gray_u8: np.ndarray, tissue_mask: np.ndarray = None) -> float:
    g = gray_u8.astype(np.float32)
    p = np.pad(g, 1, mode="reflect")
    c  = -4 * p[1:-1, 1:-1]
    n  = 1 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
    lap = c + n
    if tissue_mask is not None:
        mask = tissue_mask.astype(bool)
        if mask.sum() == 0: return 0.0
        vals = lap[mask]
    else:
        vals = lap.ravel()
    return float(np.var(vals))

def qc_on_thumbnail(img: Image.Image):
    H, S, V = to_hsv_np(img)
    gray = np.array(img.convert("L"), dtype=np.uint8)
    tissue_mask = (S >= HSV_S_TISSUE_MIN) & (V < HSV_V_WHITE_MIN)
    white_mask = (V >= HSV_V_WHITE_MIN) & (S <= HSV_S_WHITE_MAX)
    pen_mask = (H >= HSV_H_BLUE_MIN) & (H <= HSV_H_BLUE_MAX) & (S >= HSV_S_PEN_MIN)
    total = img.size[0] * img.size[1]
    tissue_pct = float(tissue_mask.sum() / total)
    white_pct  = float(white_mask.sum() / total)
    pen_pct    = float(pen_mask.sum() / total)
    blur_val = laplacian_var(gray, tissue_mask)
    if tissue_mask.sum() > 0:
        brightness_mean = float(V[tissue_mask].mean())
        saturation_mean = float(S[tissue_mask].mean())
    else:
        brightness_mean = float(V.mean())
        saturation_mean = float(S.mean())
    return {
        "tissue_pct": tissue_pct, "white_pct": white_pct, "pen_pct": pen_pct,
        "blur_var": blur_val, "brightness_mean": brightness_mean, "saturation_mean": saturation_mean,
    }, tissue_mask

def qc_reason_flags(m, thresholds):
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
    thumb_path = QC_THUMBS_DIR / f"{slide_id}_thumb.jpg"
    img.save(str(thumb_path), "JPEG", quality=90)
    overlay = np.array(img).copy()
    red = np.zeros_like(overlay); red[..., 0] = 255
    alpha = 0.35
    mask3 = np.stack([tissue_mask]*3, axis=-1)
    overlay = (overlay * (~mask3) + (alpha * overlay + (1 - alpha) * red) * mask3).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    overlay_path = QC_THUMBS_DIR / f"{slide_id}_overlay.jpg"
    overlay_img.save(str(overlay_path), "JPEG", quality=90)
    return str(thumb_path), str(overlay_path)

print(f"== OP_FM Script 3: QC & Tissue Mask ==\n[{now_iso()}] Loading manifest:", MANIFEST_PARQUET)
df_manifest = pd.read_parquet(MANIFEST_PARQUET).copy()
if QC_MAX_SLIDES is not None:
    df_manifest = df_manifest.head(QC_MAX_SLIDES).copy()
print(f"[INFO] Slides to QC: {len(df_manifest)}")

thresholds = {
    "min_tissue_pct": MIN_TISSUE_PCT, "max_white_pct": MAX_WHITE_PCT,
    "min_blur_var": MIN_BLUR_VAR, "max_pen_pct": MAX_PEN_PCT,
}

results = []; failures = []; t_start = time.time()

def worker(row):
    slide_path = Path(row["path"]); slide_id = str(row["slide_id"]); cancer = str(row.get("cancer_code", "UNKNOWN"))
    try:
        img = load_thumbnail(slide_path, THUMB_MAX_SIDE)
        metrics, tissue_mask = qc_on_thumbnail(img)
        reasons = qc_reason_flags(metrics, thresholds)
        thumb_p, overlay_p = save_thumb_and_mask(slide_id, img, tissue_mask)
        rec = {
            "slide_id": slide_id, "cancer_code": cancer, "path": str(slide_path),
            "tissue_pct": metrics["tissue_pct"], "white_pct": metrics["white_pct"],
            "pen_pct": metrics["pen_pct"], "blur_var": metrics["blur_var"],
            "brightness_mean": metrics["brightness_mean"], "saturation_mean": metrics["saturation_mean"],
            "excluded": int(len(reasons) > 0), "reasons": ";".join(reasons) if reasons else "",
            "thumb": thumb_p, "overlay": overlay_p,
        }
        return True, rec
    except Exception as e:
        return False, {"slide_id": slide_id, "path": str(slide_path), "error": f"{e.__class__.__name__}: {e}"}

done = 0; last_print = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = [ex.submit(worker, row) for _, row in df_manifest.iterrows()]
    for fut in as_completed(futs):
        ok, payload = fut.result()
        if ok: results.append(payload)
        else: failures.append(payload)
        done += 1
        now = time.time()
        if now - last_print > 2 or done == len(df_manifest):
            rate = done / (now - t_start + 1e-9)
            print(f"[QC] {done}/{len(df_manifest)} ({rate:.1f} slides/s)")
            last_print = now

elapsed = time.time() - t_start
print(f"[OK] QC completed in {elapsed/60:.1f} min.")

df_qc = pd.DataFrame.from_records(results).sort_values("slide_id").reset_index(drop=True)
df_qc.to_parquet(QC_METRICS_PARQUET, index=False)
df_qc.to_csv(QC_METRICS_CSV, index=False, encoding="utf-8-sig")

if failures:
    df_fail = pd.DataFrame(failures)
    fail_path = SUBDIRS["qc"] / "qc_failures.csv"
    df_fail.to_csv(fail_path, index=False, encoding="utf-8-sig")

df_excl = df_qc[df_qc["excluded"] == 1].copy()
df_excl.to_csv(QC_EXCLUSIONS_CSV, index=False, encoding="utf-8-sig")

n_total = len(df_qc); n_excl = len(df_excl)
print(f"\n== QC Summary ==\n Total slides QC'd : {n_total:,}\n Excluded          : {n_excl:,} ({100.0*n_excl/max(1,n_total):.1f}%)")

# [QC figures and compute-passport update omitted for brevity]
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

assert 'WSI_ROOT' in globals() and 'WORKSPACE' in globals() and 'SUBDIRS' in globals(), \
    "Please run Script 1 first to define WSI_ROOT/WORKSPACE/SUBDIRS."

MANIFEST_PARQUET = SUBDIRS["manifests"] / "manifest_tcga.parquet"
QC_METRICS_PARQUET = SUBDIRS["qc"] / "qc_metrics_tcga.parquet"
TILES_DIR = SUBDIRS["tiles"] / "manifests"
TILES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = SUBDIRS["figures"]

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

QC_POLICY = "medium"
TARGET_SCALES = [0.5, 2.0]
TILE_SIZE = 256
OVERLAP   = 32
STRIDE    = TILE_SIZE - OVERLAP
MAX_TOKENS = {0.5: 1200, 2.0: 400}
MIN_TILE_TISSUE_COVERAGE = 0.30
MASK_MAX_SIDE = 2048
HSV_S_TISSUE_MIN = 20
HSV_V_WHITE_MIN  = 230

# Token selection modes
# "l2_norm": rank tiles by feature vector L2 norm, retain top-k (requires Script 5 features)
# "uniform": uniform random sampling (default for initial pass before features exist)
SAMPLING_METHOD = "l2_norm"

MAX_WORKERS = min(6, (os.cpu_count() or 8))
FORCE_REDO = False
QUICK_HEATMAPS = True
N_HEATMAPS = 2

# Feature directories for L2-norm selection (only used when SAMPLING_METHOD="l2_norm")
FEAT05_DIR = WORKSPACE / "features" / "scale0p5"
FEAT20_DIR = WORKSPACE / "features" / "scale2p0"

def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def choose_level_for_target_mpp(slide, target_mpp, fallback_base_mpp=0.25):
    props = slide.properties
    base_mpp = None
    for k in ("openslide.mpp-x", "aperio.MPP"):
        if k in props:
            try: base_mpp = float(props.get(k)); break
            except Exception: pass
    if base_mpp is None: base_mpp = fallback_base_mpp
    best_level = 0; best_mpp = base_mpp
    for lvl in range(slide.level_count):
        mpp = base_mpp * slide.level_downsamples[lvl]
        if abs(mpp - target_mpp) < abs(best_mpp - target_mpp):
            best_mpp = mpp; best_level = lvl
    return best_level, float(best_mpp)

def make_tissue_mask(slide, max_side=MASK_MAX_SIDE):
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
    sx, sy = level_to_mask_scale
    mx0, my0 = int(x * sx), int(y * sy)
    mx1, my1 = int((x + tile) * sx), int((y + tile) * sy)
    mx0, my0 = max(mx0, 0), max(my0, 0)
    mx1, my1 = min(mx1, mask.shape[1]-1), min(my1, mask.shape[0]-1)
    if mx1 <= mx0 or my1 <= my0: return 0.0
    roi = mask[my0:my1, mx0:mx1]
    return float(roi.mean())

def sample_tiles_uniform(coords, k, rng):
    if len(coords) <= k: return coords
    idx = rng.choice(len(coords), size=k, replace=False)
    return [coords[i] for i in idx]

# L2-norm token selection
def sample_tiles_l2_norm(coords, k, slide_id, scale, rng):
    """Select top-k tiles by feature vector L2 norm. Falls back to uniform if features unavailable."""
    if len(coords) <= k:
        return coords
    feat_dir = FEAT05_DIR if math.isclose(scale, 0.5, abs_tol=1e-6) else FEAT20_DIR
    feat_path = feat_dir / f"{slide_id}.npy"
    if not feat_path.exists():
        print(f"[WARN] L2-norm selection requested but features not found for {slide_id}@{scale}; falling back to uniform")
        return sample_tiles_uniform(coords, k, rng)
    feats = np.load(feat_path, mmap_mode='r')  # [N, 768]
    n_feats = feats.shape[0]
    n_coords = len(coords)
    if n_feats != n_coords:
        print(f"[WARN] Feature count ({n_feats}) != candidate count ({n_coords}) for {slide_id}@{scale}; falling back to uniform")
        return sample_tiles_uniform(coords, k, rng)
    norms = np.linalg.norm(feats, axis=1)  # [N]
    top_idx = np.argsort(norms)[-k:]  # top-k by L2 norm
    return [coords[i] for i in top_idx]

def write_parquet(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

print(f"== OP_FM Script 4: Two-scale Tiling & Token Budget ==\n[{now_iso()}] Loading:", MANIFEST_PARQUET)
df_manifest = pd.read_parquet(MANIFEST_PARQUET)

df_qc = None
if QC_POLICY in ("strict", "medium") and QC_METRICS_PARQUET.exists():
    df_qc = pd.read_parquet(QC_METRICS_PARQUET)
    df_qc = df_qc[["slide_id", "tissue_pct", "white_pct", "pen_pct", "reasons"]].copy()

if QC_POLICY == "strict" and df_qc is not None:
    keep = df_manifest.merge(df_qc[["slide_id", "reasons"]], on="slide_id", how="left")
    keep = keep[keep["reasons"].isna() | (keep["reasons"] == "")]
elif QC_POLICY == "medium" and df_qc is not None:
    keep = df_manifest.merge(df_qc, on="slide_id", how="left")
    keep = keep[(keep["tissue_pct"].fillna(1.0) >= 0.05) & (keep["white_pct"].fillna(0.0) <= 0.95)].copy()
else:
    keep = df_manifest.copy()
keep = keep.reset_index(drop=True)
print(f"[INFO] Slides selected under QC policy '{QC_POLICY}': {len(keep):,} out of {len(df_manifest):,}")
print(f"[INFO] Token selection method: {SAMPLING_METHOD}")

def process_slide(row):
    slide_path = Path(row["path"]); slide_id = str(row["slide_id"]); cancer = str(row.get("cancer_code", "UNKNOWN"))
    outputs = []; errors = []
    try:
        slide = openslide.OpenSlide(str(slide_path))
    except Exception as e:
        return slide_id, cancer, None, [f"OpenSlideError: {e}"]
    try:
        thumb_rgb, mask = make_tissue_mask(slide, MASK_MAX_SIDE)
    except Exception as e:
        slide.close(); return slide_id, cancer, None, [f"MaskBuildError: {e}"]
    level_dims = [slide.level_dimensions[i] for i in range(slide.level_count)]
    base_w, base_h = level_dims[0]

    for target in TARGET_SCALES:
        out_path = TILES_DIR / f"{slide_id}_scale{str(target).replace('.','p')}.parquet"
        if out_path.exists() and not FORCE_REDO:
            outputs.append({"scale": target, "manifest": str(out_path), "n_tiles": None, "skipped": True})
            continue
        try:
            level, approx_mpp = choose_level_for_target_mpp(slide, target)
            level_w, level_h = level_dims[level]
            tw, th = thumb_rgb.size
            sx = tw / base_w; sy = th / base_h
            ds = slide.level_downsamples[level]
            level_to_mask_scale = (sx * ds, sy * ds)
            xs, ys = grid_positions(level_w, level_h, TILE_SIZE, STRIDE)
            cand = []
            for y in ys:
                for x in xs:
                    cov = coverage_from_mask(mask, level, level_to_mask_scale, x, y, TILE_SIZE)
                    if cov >= MIN_TILE_TISSUE_COVERAGE:
                        cand.append((x, y))
            n_cand = len(cand)
            budget = MAX_TOKENS.get(target, 0)
            rng = np.random.default_rng(SEED + hash(slide_id) % (2**16) + int(target*100))

#             Use L2-norm or uniform based on SAMPLING_METHOD
            if SAMPLING_METHOD == "l2_norm":
                chosen = sample_tiles_l2_norm(cand, budget, slide_id, target, rng)
            else:
                chosen = sample_tiles_uniform(cand, budget, rng)

            data = []
            for idx, (x, y) in enumerate(chosen):
                data.append({
                    "slide_id": slide_id, "cancer_code": cancer, "scale_um_per_px": float(target),
                    "level": int(level), "x": int(x), "y": int(y), "tile_size": TILE_SIZE,
                    "overlap": OVERLAP, "approx_mpp": approx_mpp, "tile_idx": int(idx), "seed": int(SEED),
                })
            df_tiles = pd.DataFrame.from_records(data)
            write_parquet(df_tiles, out_path)
            outputs.append({"scale": target, "manifest": str(out_path), "n_tiles": len(df_tiles), "skipped": False})
        except Exception as e:
            errors.append(f"TilingError(scale={target}): {e}")
    slide.close()
    return slide_id, cancer, outputs, errors

t0 = time.time(); done = 0; errors_all = []; per_slide_counts = []
print(f"[{now_iso()}] Starting tiling on {len(keep)} slides with {MAX_WORKERS} workers...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = [ex.submit(process_slide, row) for _, row in keep.iterrows()]
    for fut in as_completed(futs):
        slide_id, cancer, outputs, errs = fut.result()
        done += 1
        if errs:
            for e in errs: errors_all.append({"slide_id": slide_id, "error": e})
        if outputs:
            for rec in outputs:
                if rec is None: continue
                if not rec.get("skipped", False):
                    per_slide_counts.append({
                        "slide_id": slide_id, "cancer_code": cancer,
                        "scale_um_per_px": rec["scale"], "n_tiles": rec["n_tiles"], "manifest": rec["manifest"],
                    })
        if done % 25 == 0 or done == len(keep):
            rate = done / (time.time() - t0 + 1e-9)
            print(f"[TILING] {done}/{len(keep)} ({rate:.2f} slides/s)")

elapsed = time.time() - t0
print(f"[OK] Tiling finished in {elapsed/60:.1f} min.")

df_sum = pd.DataFrame.from_records(per_slide_counts)
sum_path = SUBDIRS["tiles"] / "tiling_summary_tcga.parquet"
df_sum.to_parquet(sum_path, index=False)

if errors_all:
    df_err = pd.DataFrame.from_records(errors_all)
    err_path = SUBDIRS["tiles"] / "tiling_errors_tcga.csv"
    df_err.to_csv(err_path, index=False, encoding="utf-8-sig")

for scale in TARGET_SCALES:
    df_sc = df_sum[df_sum["scale_um_per_px"] == scale]
    if len(df_sc) == 0: continue
    plt.figure(figsize=(8,5))
    plt.hist(df_sc["n_tiles"].dropna().values, bins=40)
    plt.xlabel(f"Tokens per slide @ {scale} um/px"); plt.ylabel("Slides")
    plt.title(f"Token Distribution @ {scale} um/px"); plt.tight_layout()
    outp = FIG_DIR / f"tiling_tokens_dist_scale{str(scale).replace('.','p')}.png"
    plt.savefig(outp); plt.close()

print("\nScript 4 complete. Next: Script 5 (Frozen-backbone feature extraction to 768-D).")


# SCRIPT 05: FROZEN-BACKBONE FEATURE EXTRACTION (CONVNEXT-TINY)

# Script 5 — OpenSlide extractor
import os, sys, json, time, math, random, shutil, subprocess, platform, gc
from pathlib import Path
from datetime import datetime
from time import perf_counter

SUBDIRS = {
    "features": WORKSPACE / "features",
    "tiles":    WORKSPACE / "tiles",
    "logs":     WORKSPACE / "logs",
    "figures":  WORKSPACE / "figures",
}
for p in SUBDIRS.values(): p.mkdir(parents=True, exist_ok=True)

TSUM = SUBDIRS["tiles"] / "tiling_summary_tcga.parquet"
assert TSUM.exists(), f"Missing tiling summary: {TSUM}"

def ensure(pkg):
    try: __import__(pkg.split('[')[0].replace('-','_').split('==')[0])
    except Exception: subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

ensure("openslide_python"); ensure("openslide_bin"); ensure("torch>=2.1")
ensure("torchvision"); ensure("pandas"); ensure("pyarrow"); ensure("Pillow")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
import openslide

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16 if DEVICE=="cuda" else torch.bfloat16
TILE_SIZE = 256; MODEL_IN = 224; SELFTEST_SECONDS = 60
TARGET_TILES_PER_SEC = 50.0; RANDOM_SEED = 13; SAVE_DTYPE = np.float16

random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if hasattr(torch.backends,"cudnn"):
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.allow_tf32 = True
if hasattr(torch,"set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
_to_tensor = T.ToTensor()
_resize    = T.Resize((MODEL_IN, MODEL_IN), interpolation=T.InterpolationMode.BILINEAR)
_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
def to_model_tensor(img: Image.Image) -> torch.Tensor:
    if img.size != (MODEL_IN, MODEL_IN): img = _resize(img)
    t = _to_tensor(img); t = _normalize(t); return t

class ConvNeXtTinyFeats(nn.Module):
    def __init__(self):
        super().__init__()
        w = tvm.ConvNeXt_Tiny_Weights.DEFAULT
        m = tvm.convnext_tiny(weights=w)
        self.features = m.features; self.gap = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters(): p.requires_grad=False
        self.eval()
    @torch.no_grad()
    def forward(self, x):
        x = self.features(x); x = self.gap(x).flatten(1); return x

def build_model():
    m = ConvNeXtTinyFeats().to(DEVICE)
    if DEVICE=="cuda":
        m = m.to(memory_format=torch.channels_last)
        d = torch.randn(256,3,MODEL_IN,MODEL_IN, device=DEVICE).to(memory_format=torch.channels_last)
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=True): _ = m(d)
        torch.cuda.synchronize()
    return m

MODEL = build_model()

df_sum = pd.read_parquet(TSUM)
assert "manifest" in df_sum.columns and "slide_id" in df_sum.columns

SLIDE_INDEX_PATH = SUBDIRS["logs"] / "slide_path_index.json"
def index_slide_paths(root: Path) -> dict:
    print("[INDEX] Building slide path map (once) ...")
    mp={}
    for ext in ("*.svs","*.ndpi","*.tif","*.mrxs","*.scn"):
        for p in root.rglob(ext): mp[p.stem] = str(p)
    return mp
if SLIDE_INDEX_PATH.exists():
    slide_map = json.loads(SLIDE_INDEX_PATH.read_text(encoding="utf-8"))
else:
    slide_map = index_slide_paths(WSI_ROOT)
    SLIDE_INDEX_PATH.write_text(json.dumps(slide_map, indent=2), encoding="utf-8")

def slide_path_from_id(slide_id, manifest_df=None):
    if manifest_df is not None:
        for cand in ("path","source_path","slide_path","wsi_path"):
            if cand in manifest_df.columns:
                p = manifest_df[cand].iloc[0]
                if isinstance(p,str) and Path(p).exists(): return p
    if slide_id in slide_map: return slide_map[slide_id]
    base = slide_id.split(".")[0]
    return slide_map.get(base, None)

def load_manifest(man_path):
    m = pd.read_parquet(man_path)
    lower = {c.lower():c for c in m.columns}
    def pick(*names):
        for n in names:
            if n in m.columns: return n
            if n.lower() in lower: return lower[n.lower()]
        raise KeyError(f"Missing columns {names} in {man_path.name}")
    xcol = pick("x","px_x","x_level"); ycol = pick("y","px_y","y_level"); lvlcol = pick("level","lvl")
    tsize = TILE_SIZE
    for n in ("tile_size","tile_px","size"):
        if n in m.columns:
            try: tsize = int(m[n].iloc[0])
            except: pass; break
    return m, xcol, ycol, lvlcol, tsize

class SlideReader:
    def __init__(self, path):
        self.path = path; self.osr = openslide.OpenSlide(path)
        self.down = list(self.osr.level_downsamples)
    def read_tile(self, level, x_level, y_level, size):
        ds = self.down[level]; bx = int(round(x_level * ds)); by = int(round(y_level * ds))
        img = self.osr.read_region((bx,by), level, (size,size)).convert("RGB"); return img
    def close(self):
        try: self.osr.close()
        except: pass

def iter_batches_from_manifest(reader, man_df, xcol, ycol, lvlcol, tile_px, max_batch=4096):
    buf=[]
    for r in man_df[[xcol,ycol,lvlcol]].itertuples(index=False, name=None):
        x,y,lvl = map(int, r)
        img = reader.read_tile(lvl, x, y, tile_px)
        t = to_model_tensor(img); buf.append(t)
        if len(buf) >= max_batch:
            batch = torch.stack(buf,0).to(memory_format=torch.channels_last); yield batch; buf.clear()
    if buf:
        batch = torch.stack(buf,0).to(memory_format=torch.channels_last); yield batch

def forward_batches(model, batches_iter):
    outs=[]
    for cpu_batch in batches_iter:
        with torch.no_grad():
            chunk = cpu_batch.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE=="cuda")):
                out = model(chunk)
            outs.append(out.detach().cpu())
        del cpu_batch
    feats = torch.cat(outs,0).contiguous().numpy(); return feats

OUT05 = SUBDIRS["features"] / "scale0p5"; OUT20 = SUBDIRS["features"] / "scale2p0"
OUT05.mkdir(parents=True, exist_ok=True); OUT20.mkdir(parents=True, exist_ok=True)
def out_paths(slide_id, scale, ext="npy"):
    d = OUT05 if math.isclose(scale,0.5,abs_tol=1e-6) else OUT20
    return d / f"{slide_id}.{ext}", d / f"{slide_id}_meta.parquet"

# Environment log
env_s5 = {
    "time": datetime.now().isoformat(timespec="seconds"),
    "python": sys.version.split()[0], "platform": platform.platform(),
    "device": DEVICE, "torch": torch.__version__, "amp_dtype": str(AMP_DTYPE),
}
(SUBDIRS["logs"] / "script5_env.json").write_text(json.dumps(env_s5, indent=2), encoding="utf-8")
print("[ENV]\n" + json.dumps(env_s5, indent=2))

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
print(f"[INFO] Slides pending (>=1 scale): {len(groups)}")

# Self-test (60 s throughput gate)
def selftest(seconds=SELFTEST_SECONDS, target=TARGET_TILES_PER_SEC):
    cand=[]
    for sid, g in df_sum.groupby("slide_id"):
        n=int(g["n_tiles"].sum()); man=Path(g.sort_values("n_tiles").iloc[-1]["manifest"])
        cand.append((n, sid, man))
    cand.sort(key=lambda x:x[0])
    pick = cand[:min(12, len(cand))]
    readers={}
    for _, sid, manp in pick:
        m, xcol, ycol, lvlcol, tpx = load_manifest(manp)
        fn = slide_path_from_id(sid, m)
        if not fn or not Path(fn).exists(): continue
        readers[sid] = (SlideReader(fn), m[[xcol,ycol,lvlcol]].copy(), xcol, ycol, lvlcol, tpx)
    tiles_done=0; t0=perf_counter(); stop=t0+seconds
    while perf_counter()<stop and readers:
        for sid,(sr, m, xcol,ycol,lvlcol,tpx) in list(readers.items()):
            take = m.iloc[:512]
            if take.empty: del readers[sid]; sr.close(); continue
            batches = iter_batches_from_manifest(sr, take, xcol,ycol,lvlcol, tpx, max_batch=2048)
            with torch.no_grad():
                for cpu_batch in batches:
                    chunk = cpu_batch.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE=="cuda")):
                        _ = MODEL(chunk)
                    tiles_done += chunk.size(0); del cpu_batch, chunk
                    if perf_counter()>=stop: break
            m = m.iloc[len(take):]
            readers[sid]=(sr, m, xcol,ycol,lvlcol,tpx)
            if perf_counter()>=stop: break
    dt = perf_counter()-t0; rate = tiles_done / max(dt,1e-6)
    print(f"[SELFTEST] tiles={tiles_done}  time={dt:.1f}s  tiles/s={rate:.1f}")
    print(("[PASS] " if rate>=target else "[FAIL] ")+f"{rate:.1f} tiles/s (target >= {target:.0f})")
    (SUBDIRS["logs"]/ "script5_selftest.json").write_text(json.dumps({
        "tiles": tiles_done, "seconds": round(dt,2), "tiles_per_s": round(rate,2),
        "target": target, "pass": rate>=target
    }, indent=2), encoding="utf-8")
    for sid,(sr, *_rest) in readers.items(): sr.close()
    return rate

rate = selftest()
if rate < TARGET_TILES_PER_SEC:
    print("[ABORT] Below target. This cell stops here (no full run).")
    raise SystemExit(0)

# Full extraction run
PROG = SUBDIRS["logs"] / "script5_progress.jsonl"
def log_progress(**kw):
    kw["ts"]=datetime.now().isoformat(timespec="seconds")
    with open(PROG,"a",encoding="utf-8") as f: f.write(json.dumps(kw,ensure_ascii=False)+"\n")

for i, grp in enumerate(groups, 1):
    sid = grp["slide_id"]
    man_pref = min(grp["entries"], key=lambda e: abs(e["scale"]-0.5))
    m_probe, xcol, ycol, lvlcol, tpx = load_manifest(man_pref["manifest"])
    fn = slide_path_from_id(sid, m_probe)
    if not fn or not Path(fn).exists():
        print(f"[WARN] slide path not found: {sid} — skipped"); continue
    reader = SlideReader(fn)

    for e in grp["entries"]:
        sc = float(e["scale"])
        npy_path, meta_path = out_paths(sid, sc)
        if npy_path.exists() and meta_path.exists(): continue

        man_df, xcol, ycol, lvlcol, tpx = load_manifest(e["manifest"])
        if man_df.empty:
            print(f"[WARN] empty manifest: {e['manifest']} — skip"); continue

        t0 = perf_counter()
        batches = iter_batches_from_manifest(reader, man_df, xcol,ycol,lvlcol, tpx, max_batch=4096)
        feats = forward_batches(MODEL, batches)
        if DEVICE=="cuda": torch.cuda.synchronize()
        dt = perf_counter()-t0

        np.save(npy_path, feats.astype(SAVE_DTYPE))
        md = man_df.copy(); md["slide_id"]=sid; md["scale_um_per_px"]=sc
        md.to_parquet(meta_path, index=False)

        N = int(feats.shape[0]); tiles_per_s = N / max(dt,1e-6)
        vram = (torch.cuda.max_memory_allocated()/(1024**3)) if DEVICE=="cuda" else 0.0
        print(f"[OK] {i}/{len(groups)} | {sid} @{sc:.1f} um/px -> ({N},768) | {tiles_per_s:.1f} tiles/s | VRAM~{vram:.2f} GB")
        log_progress(slide_id=sid, scale=sc, tiles=N, seconds=round(dt,2), tps=round(tiles_per_s,2), vram_gb=round(vram,2))

        del feats; gc.collect()
        if DEVICE=="cuda": torch.cuda.empty_cache()

    reader.close()

print("[DONE] Script 5 — All pending entries processed.")


# SCRIPT 06: SELF-SUPERVISED PRE-TRAINING (BYOL + MFR)
import os, sys, json, math, random, gc, subprocess, platform
from pathlib import Path
from time import perf_counter
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

FEATURES05 = WORKSPACE / "features" / "scale0p5"
FEATURES20 = WORKSPACE / "features" / "scale2p0"
LOGS       = WORKSPACE / "logs"
WEIGHTS    = WORKSPACE / "weights"
FIGS       = WORKSPACE / "figures"
EMBED      = WORKSPACE / "embeddings" / "student_final"
for p in [LOGS, WEIGHTS, FIGS, EMBED]:
    p.mkdir(parents=True, exist_ok=True)

def _pip(*pkgs):
    try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    except Exception as e: print(f"[WARN] pip install failed for {pkgs}: {e}")

try: import numpy as np; import pandas as pd
except Exception: _pip("numpy>=1.24","pandas>=2.0"); import numpy as np, pandas as pd

try: import torch, torch.nn as nn, torch.nn.functional as F
except Exception: _pip("torch>=2.1"); import torch, torch.nn as nn, torch.nn.functional as F

try: from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
except Exception: _pip("safetensors>=0.4.0"); from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; HAS_MPL = True
except Exception: HAS_MPL = False

CONFIG = {
    "seed": 13,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype_amp": "float16",
    "token_budget_0p5": 1200,
    "token_budget_2p0":  400,
    "mask_frac": 0.25,
    "lambda_mfr": 0.5,
    "d_model": 768,
    "n_heads": 8,
    "n_layers": 6,
    "ff_mult": 4,
    "dropout": 0.1,
    "batch_slides": 3,
    "grad_accum": 2,
    "epochs": 4,
    "steps_per_epoch_cap": None,
    "lr": 1.5e-4,
    "weight_decay": 1e-4,
    "ema_tau": 0.996,
    "warmup_steps": 500,
    "save_every_steps": 1000,
    "log_every_steps": 50,
    "resume_if_available": True,
    "export_embeddings_after_train": True,
    "export_use_budget": True,
#     Feature-space augmentation parameters
    "aug_dropout_p": 0.1,
    "aug_noise_std": 0.02,
}

SEED = CONFIG["seed"]
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if hasattr(torch.backends,"cudnn"):
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DEVICE = CONFIG["device"]
AMP_DTYPE = (torch.float16 if (DEVICE=="cuda" and CONFIG["dtype_amp"]=="float16") else
             torch.bfloat16 if (DEVICE=="cuda" and CONFIG["dtype_amp"]=="bfloat16") else
             torch.float32)

def _collect(dir_path: Path) -> Dict[str, Path]:
    return {p.stem: p for p in dir_path.glob("*.npy")}

mp05 = _collect(FEATURES05); mp20 = _collect(FEATURES20)
common_ids = sorted(set(mp05.keys()) & set(mp20.keys()))
assert len(common_ids)>0, "No slides found that have both 0.5 and 2.0 um features."

@dataclass
class SlideRec:
    slide_id: str; npy05: Path; meta05: Path; npy20: Path; meta20: Path

def meta_path(npy_path: Path) -> Path:
    return npy_path.with_name(npy_path.stem + "_meta.parquet")

slides: List[SlideRec] = []
for sid in common_ids:
    p05 = mp05[sid]; p20 = mp20[sid]
    m05 = meta_path(p05); m20 = meta_path(p20)
    if m05.exists() and m20.exists():
        slides.append(SlideRec(sid, p05, m05, p20, m20))

print(json.dumps({
    "time": datetime.now().isoformat(timespec="seconds"),
    "torch": torch.__version__, "device": DEVICE,
    "amp_dtype": str(AMP_DTYPE).split(".")[-1], "slides_2scale": len(slides)
}, indent=2))

_META_CACHE: Dict[Path, pd.DataFrame] = {}
def load_meta(p: Path) -> pd.DataFrame:
    if p in _META_CACHE: return _META_CACHE[p]
    df = pd.read_parquet(p)
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in cols_lower: return cols_lower[n.lower()]
        raise KeyError(f"Missing one of {names} in {p.name}")
    xcol = pick("x"); ycol = pick("y"); lvlcol = pick("level","lvl"); sccol = pick("scale_um_per_px")
    tsize = 256
    for n in ("tile_size","tile_px","size"):
        if n in df.columns:
            try: tsize = int(df[n].iloc[0])
            except: pass; break
    out = df[[xcol,ycol,lvlcol,sccol]].copy()
    out.columns = ["x","y","level","scale_um_per_px"]; out["tile_px"] = tsize
    _META_CACHE[p] = out; return out

def compute_mm_xy(df: pd.DataFrame) -> np.ndarray:
    um_per_px = df["scale_um_per_px"].astype(float).to_numpy()
    mm_per_px = um_per_px / 1000.0
    cx = (df["x"].to_numpy() + df["tile_px"].to_numpy()/2.0) * mm_per_px
    cy = (df["y"].to_numpy() + df["tile_px"].to_numpy()/2.0) * mm_per_px
    return np.stack([cx, cy], axis=1).astype(np.float32)

# Model uses MAX POOLING, no CLS token
class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(3, d_model//2), nn.GELU(), nn.Linear(d_model//2, d_model))
    def forward(self, mmxy, scale_um):
        x = torch.cat([mmxy, scale_um], dim=-1)
        return self.proj(x)

class MILTransformer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_layers=6, ff_mult=4, dropout=0.1):
        super().__init__()
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

    def forward(self, feats, mmxy, scale_um, pad_mask):
        B, T, _ = feats.shape
        pos = self.pos(mmxy, scale_um)
        x = feats + pos
        # No CLS token prepended — direct transformer encoding
        x = self.enc(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)  # [B, T, D]
        # Max pooling over non-padded tokens
        x_for_pool = x.masked_fill(pad_mask.unsqueeze(-1), float('-inf'))
        g = x_for_pool.max(dim=1).values  # [B, D]
        t = x  # token-level outputs for MFR
        g_proj = self.proj_global(g)
        t_proj = self.proj_token(t)
        g_pred = self.pred_global(g_proj)
        t_pred = self.pred_token(t_proj)
        return g_proj, t_proj, g_pred, t_pred

# Losses & EMA
def cosine_loss(p, z):
    p = F.normalize(p, dim=-1); z = F.normalize(z.detach(), dim=-1)
    return (1.0 - (p * z).sum(dim=-1)).mean()

@torch.no_grad()
def ema_update(teacher, student, tau):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=(1.0 - tau))

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

def _sample(n, k):
    if n <= k: return np.arange(n, dtype=np.int64)
    return np.random.choice(n, size=k, replace=False).astype(np.int64)

def load_tokens_for_slide(rec, budget05, budget20):
    f05 = np.load(rec.npy05, mmap_mode='r')
    m05 = load_meta(rec.meta05); idx05 = _sample(f05.shape[0], budget05)
    mm05 = compute_mm_xy(m05.iloc[idx05])
    sc05 = m05["scale_um_per_px"].iloc[idx05].to_numpy(np.float32).reshape(-1,1)
    f20 = np.load(rec.npy20, mmap_mode='r')
    m20 = load_meta(rec.meta20); idx20 = _sample(f20.shape[0], budget20)
    mm20 = compute_mm_xy(m20.iloc[idx20])
    sc20 = m20["scale_um_per_px"].iloc[idx20].to_numpy(np.float32).reshape(-1,1)
    feats = np.concatenate([f05[idx05], f20[idx20]], axis=0).astype(np.float32)
    mmxy  = np.concatenate([mm05, mm20], axis=0).astype(np.float32)
    scl   = np.concatenate([sc05, sc20], axis=0).astype(np.float32)
    return feats, mmxy, scl

# Feature-space augmentation for BYOL view generation
def create_augmented_view(feats_np, drop_p=0.1, noise_std=0.02):
    """Feature-space augmentation: stochastic dropout + Gaussian noise."""
    mask = (np.random.rand(*feats_np.shape) > drop_p).astype(np.float32)
    aug = feats_np * mask
    aug = aug + np.random.normal(0, noise_std, size=feats_np.shape).astype(np.float32)
    return aug

def make_batch(batch_recs, budget05, budget20, mask_frac):
    # Create two augmented views for student; teacher will also process
    # both augmented views for cross-view BYOL
    feats_view1 = []; feats_view2 = []
    mmxy_list=[]; sc_list=[]; mask_tiles=[]
    for rec in batch_recs:
        f, mm, sc = load_tokens_for_slide(rec, budget05, budget20)
        Tn = f.shape[0]
        feats_view1.append(torch.from_numpy(create_augmented_view(f, CONFIG["aug_dropout_p"], CONFIG["aug_noise_std"])))
        feats_view2.append(torch.from_numpy(create_augmented_view(f, CONFIG["aug_dropout_p"], CONFIG["aug_noise_std"])))
        mmxy_list.append(torch.from_numpy(mm))
        sc_list.append(torch.from_numpy(sc))
        mcount = max(1, int(round(mask_frac*Tn)))
        mask_idx = np.random.choice(Tn, size=mcount, replace=False).astype(np.int64)
        mask_tiles.append(torch.from_numpy(mask_idx))

    T = max(t.shape[0] for t in feats_view1)
    B = len(batch_recs); D = feats_view1[0].shape[1]
    f_v1 = torch.zeros(B, T, D); f_v2 = torch.zeros(B, T, D)
    mmxy = torch.zeros(B, T, 2); scl = torch.zeros(B, T, 1); pad = torch.ones(B, T, dtype=torch.bool)
    for i in range(B):
        n = feats_view1[i].shape[0]
        f_v1[i,:n] = feats_view1[i]; f_v2[i,:n] = feats_view2[i]
        mmxy[i,:n] = mmxy_list[i]; scl[i,:n] = sc_list[i]; pad[i,:n] = False

    mfr_index = []
    for b, idx in enumerate(mask_tiles):
        mfr_index.append(torch.stack([torch.full_like(idx, b), idx], dim=1))
    mfr_index = torch.cat(mfr_index, dim=0)

    # MFR: zero out masked token positions in student input so the
    # transformer must reconstruct them from surrounding context
    for b_idx, m_idx in enumerate(mask_tiles):
        f_v1[b_idx, m_idx, :] = 0.0
        f_v2[b_idx, m_idx, :] = 0.0

    return {
        "feats_v1": f_v1.to(DEVICE, non_blocking=True),
        "feats_v2": f_v2.to(DEVICE, non_blocking=True),
        "mmxy": mmxy.to(DEVICE, non_blocking=True),
        "scl": scl.to(DEVICE, non_blocking=True),
        "pad": pad.to(DEVICE, non_blocking=True),
        "mfr_index": mfr_index.to(DEVICE, non_blocking=True)
    }

class CosineWarmup:
    def __init__(self, optimizer, warmup, max_steps, base_lr):
        self.opt=optimizer; self.warmup=warmup; self.max=max_steps; self.base=base_lr; self.t=0
    def step(self):
        self.t += 1
        if self.t <= self.warmup: lr = self.base * self.t / max(1,self.warmup)
        else:
            p = (self.t - self.warmup) / max(1, self.max - self.warmup)
            lr = self.base * 0.5*(1+math.cos(math.pi*p))
        for g in self.opt.param_groups: g["lr"]=lr
        return lr

LOG_CSV = LOGS / "script6_train_log.csv"
if not LOG_CSV.exists():
    LOG_CSV.write_text("ts,epoch,step,lr,loss,loss_byol,loss_mfr,tokens_per_s,vram_gb\n", encoding="utf-8")
LOG_JL = LOGS / "script6_train_log.jsonl"

def log_row(d):
    d2 = d.copy(); d2["ts"]=datetime.now().isoformat(timespec="seconds")
    with open(LOG_JL,"a",encoding="utf-8") as f: f.write(json.dumps(d2,ensure_ascii=False)+"\n")

def save_ckpt(tag):
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
    sd = load_safetensors(str(ck)); student.load_state_dict(sd, strict=True)
    teacher.load_state_dict(sd, strict=False); return True

total_steps = CONFIG["epochs"] * (len(slides)//CONFIG["batch_slides"] + 1)
sched = CosineWarmup(opt, warmup=CONFIG["warmup_steps"], max_steps=total_steps, base_lr=CONFIG["lr"])
resumed = try_resume()

global_step=0
for epoch in range(1, CONFIG["epochs"]+1):
    random.shuffle(slides)
    steps_this_epoch = 0
    max_steps_epoch = (CONFIG["steps_per_epoch_cap"] or (len(slides)//CONFIG["batch_slides"] + 1))
    i = 0
    while steps_this_epoch < max_steps_epoch and i < len(slides):
        batch_recs = slides[i : i+CONFIG["batch_slides"]]; i += CONFIG["batch_slides"]
        try:
            b = make_batch(batch_recs, CONFIG["token_budget_0p5"], CONFIG["token_budget_2p0"], CONFIG["mask_frac"])
        except Exception as e:
            print(f"[SKIP] Batch error: {e}"); continue

        # Two-view BYOL with augmented student, clean teacher
        mmxy, scl, pad, mfr_index = b["mmxy"], b["scl"], b["pad"], b["mfr_index"]
        tokens_total = int((~pad).sum().item())
        opt.zero_grad(set_to_none=True)
        t0 = perf_counter()

        # Teacher forward on both augmented views (stop-gradient)
        with torch.no_grad():
            g_t1, t_t1, _, _ = teacher(b["feats_v1"], mmxy, scl, pad)
            g_t2, t_t2, _, _ = teacher(b["feats_v2"], mmxy, scl, pad)

        # Student forward on both augmented views
        with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(DEVICE=="cuda" and AMP_DTYPE!=torch.float32)):
            g_s1, t_s1, g_sp1, t_sp1 = student(b["feats_v1"], mmxy, scl, pad)
            g_s2, t_s2, g_sp2, t_sp2 = student(b["feats_v2"], mmxy, scl, pad)

            # Cross-view BYOL: student(view1) predicts teacher(view2) and vice versa
            loss_byol = 0.5 * cosine_loss(g_sp1, g_t2) + 0.5 * cosine_loss(g_sp2, g_t1)

            # MFR loss: cross-view token reconstruction at masked positions
            bi = mfr_index
            t_s1_mask = t_sp1[bi[:,0], bi[:,1], :]
            t_s2_mask = t_sp2[bi[:,0], bi[:,1], :]
            t_t2_mask = t_t2[bi[:,0], bi[:,1], :]
            t_t1_mask = t_t1[bi[:,0], bi[:,1], :]
            loss_mfr = 0.5 * cosine_loss(t_s1_mask, t_t2_mask) + 0.5 * cosine_loss(t_s2_mask, t_t1_mask)

            loss = loss_byol + CONFIG["lambda_mfr"] * loss_mfr

        scaler.scale(loss / CONFIG["grad_accum"]).backward()

        if ((steps_this_epoch+1) % CONFIG["grad_accum"] == 0):
            scaler.step(opt); scaler.update()
            ema_update(teacher, student, tau=CONFIG["ema_tau"])
            lr = sched.step()
        else:
            lr = sched.opt.param_groups[0]["lr"]

        if DEVICE=="cuda":
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated()/(1024**3)
            torch.cuda.reset_peak_memory_stats()
        else: vram = 0.0

        dt = perf_counter()-t0; tps = tokens_total/max(dt,1e-6)
        global_step += 1; steps_this_epoch += 1

        if global_step % CONFIG["log_every_steps"] == 0:
            print(f"[E{epoch} S{global_step}] loss={loss.item():.4f} (byol {loss_byol.item():.4f} | mfr {loss_mfr.item():.4f}) "
                  f"| tokens={tokens_total} | {tps:.1f} tok/s | lr={lr:.2e} | VRAM~{vram:.2f} GB")
            log_row({"epoch":epoch, "step":global_step, "lr":lr,
                     "loss":float(loss.item()), "loss_byol":float(loss_byol.item()),
                     "loss_mfr":float(loss_mfr.item()), "tps":float(tps), "vram_gb":float(vram)})

        if global_step % CONFIG["save_every_steps"] == 0:
            save_ckpt(f"e{epoch}_s{global_step}")

        if (global_step % 200) == 0:
            del mmxy, scl, pad, mfr_index, g_t1, t_t1, g_t2, t_t2
            gc.collect()
            if DEVICE=="cuda": torch.cuda.empty_cache()

    save_ckpt(f"e{epoch}")

print("[TRAIN] Finished Script 6 pretraining.")


# Export slide embeddings using trained transformer + max pooling
def export_embeddings(ckpt_name=None, use_budget=True):
    if ckpt_name is None:
        txt = (WEIGHTS / "latest.txt")
        assert txt.exists(), "Missing weights/latest.txt"
        ckpt_name = txt.read_text(encoding="utf-8").strip()
    ckpt_path = WEIGHTS / ckpt_name
    sd = load_safetensors(str(ckpt_path))
    student.load_state_dict(sd, strict=True); student.eval()
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
        np.save(outn, emb); count += 1
        if count % 200 == 0: print(f"[EMB] {count}/{len(slides)} saved...")
    print(f"[EMB] Done: {count} slides in {(perf_counter()-t0)/60:.1f} min")

if CONFIG["export_embeddings_after_train"]:
    export_embeddings(ckpt_name=None, use_budget=CONFIG["export_use_budget"])

print("[DONE] Script 6 complete.")


# SCRIPT 06C: POST-PRETRAINING DIAGNOSTICS
# Reads training logs, checks checkpoint integrity, computes pass/warn/fail gates.
import os, sys, json, math, hashlib, shutil, subprocess, warnings, tempfile
from pathlib import Path
from datetime import datetime, timedelta

LOGS_DIR    = WORKSPACE / "logs"
MODELS_DIR  = WORKSPACE / "models"
FEATURES_05_DIR  = WORKSPACE / "features" / "scale0p5"
FEATURES_20_DIR  = WORKSPACE / "features" / "scale2p0"
DIAG_DIR    = WORKSPACE / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LOG_CSV = LOGS_DIR / "script6_train_log.csv"

import pandas as pd, numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
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
    s05 = {p.stem for p in FEATURES_05_DIR.glob("*.npy")}
    s20 = {p.stem for p in FEATURES_20_DIR.glob("*.npy")}
    return len(s05 & s20), len(s05), len(s20)

df_log = safe_read_csv(TRAIN_LOG_CSV)
diag = {"time": datetime.now().isoformat(timespec="seconds"),
        "workspace": str(WORKSPACE),
        "log_csv_exists": TRAIN_LOG_CSV.exists(),
        "log_rows": int(len(df_log))}

for c in ["epoch","step","loss","loss_byol","loss_mfr","tps","vram_gb","ts"]:
    if c not in df_log.columns:
        df_log[c] = datetime.now().isoformat(timespec="seconds") if c == "ts" else np.nan

for c in ["epoch","step","loss","loss_byol","loss_mfr","tps","vram_gb"]:
    df_log[c] = pd.to_numeric(df_log[c], errors="coerce")

steps_logged = int(df_log["step"].max()) if len(df_log) else 0
diag["steps_logged"] = steps_logged

if len(df_log) > 3 and df_log["loss"].notna().any():
    n = len(df_log)
    head = df_log["loss"].dropna().iloc[:max(3, n//10)]
    tail = df_log["loss"].dropna().iloc[-max(3, n//10):]
    start_med = float(np.median(head)) if len(head) else np.nan
    end_med   = float(np.median(tail)) if len(tail) else np.nan
    rel_impr  = float((start_med - end_med) / start_med) if (start_med and start_med==start_med) else np.nan
else:
    start_med = end_med = rel_impr = np.nan

diag["loss_start_median"] = start_med
diag["loss_end_median"]   = end_med
diag["loss_rel_improvement"] = rel_impr

n_both, n05, n20 = count_2scale_slides()
diag["features_2scale_intersection"] = int(n_both)

ckpts = list_ckpts(MODELS_DIR)
diag["checkpoint_count"] = int(len(ckpts))
ckpt_info = []
for p in ckpts[-6:]:
    ok, meta = try_torch_load(p)
    ckpt_info.append({
        "file": str(p), "size_mb": round(p.stat().st_size/(1024**2),2),
        "sha256_12": sha256_12(p), "load_ok": bool(ok), "meta": meta
    })
diag["checkpoints_recent"] = ckpt_info
diag["suggest_checkpoint"] = (str(ckpts[-1]) if len(ckpts) else None)

gates = []
if n_both >= 18000:
    gates.append(("G1_2scale_coverage", "PASS", f"{n_both} slides with both scales"))
elif n_both >= 15000:
    gates.append(("G1_2scale_coverage", "WARN", f"{n_both} < expected"))
else:
    gates.append(("G1_2scale_coverage", "FAIL", f"{n_both} very low"))

if rel_impr == rel_impr:
    if rel_impr >= 0.60:   gates.append(("G2_loss_improvement", "PASS", f"relative drop {rel_impr:.2f}"))
    elif rel_impr >= 0.30: gates.append(("G2_loss_improvement", "WARN", f"modest drop {rel_impr:.2f}"))
    else:                  gates.append(("G2_loss_improvement", "FAIL", f"weak drop {rel_impr:.2f}"))
else:
    gates.append(("G2_loss_improvement", "WARN", "loss trend unavailable"))

if len(ckpts) == 0:
    gates.append(("G4_checkpoints", "FAIL", "no model files"))
elif any(not c["load_ok"] for c in ckpt_info):
    gates.append(("G4_checkpoints", "WARN", f"{sum(1 for c in ckpt_info if not c['load_ok'])} failed to load"))
else:
    gates.append(("G4_checkpoints", "PASS", f"{len(ckpts)} file(s), latest loads OK"))

diag["gates"] = [{"name": n, "status": s, "detail": d} for (n,s,d) in gates]

DIAG_JSON = DIAG_DIR / "script6c_posttrain_diagnostics.json"
with open(DIAG_JSON, "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2, ensure_ascii=False)

print("\nGATES:")
for (n,s,d) in gates:
    tag = {"PASS":"[ OK ]", "WARN":"[WARN]", "FAIL":"[FAIL]"}[s]
    print(f" {tag} {n}: {d}")
print(f"[OK] Diagnostics saved: {DIAG_JSON}")


# SCRIPT 07: TCGA 31-CLASS PAN-CANCER EVALUATION
import os, sys, json, warnings, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Config:
    OPENSLIDE = WORKSPACE
    # Use OpenSlideFM embeddings from Script 08
    EMBEDDINGS_DIR = WORKSPACE / "embeddings"
    LABELS = WSI_ROOT.parent / "artifacts" / "labels" / "labels.csv"
    MANIFEST = WORKSPACE / "manifests" / "manifest_tcga.csv"
    OUTPUT = WORKSPACE / "results" / "tcga_baseline_evaluation"
    # 3 seeds
    SEEDS = [42, 123, 456]
    N_FOLDS = 5
    BOOTSTRAP_N = 1000

CFG = Config()
CFG.OUTPUT.mkdir(parents=True, exist_ok=True)

# Load OpenSlideFM embeddings (per-slide .npy from Script 08)
def load_openslidefm_embeddings():
    """Load per-slide embeddings and aggregate to patient-level by mean pooling."""
    print("[LOAD] Reading OpenSlideFM embeddings from Script 08 outputs...")
    emb_dirs = [d for d in CFG.EMBEDDINGS_DIR.iterdir() if d.is_dir()]
    records = []
    for emb_dir in emb_dirs:
        for npy_path in emb_dir.glob("*.npy"):
            slide_id = npy_path.stem
            emb = np.load(npy_path).astype(np.float32)
            if emb.ndim == 1 and emb.shape[0] == 768:
                records.append({"slide_id": slide_id, "emb": emb})
    # Also check top-level embeddings dir
    for npy_path in CFG.EMBEDDINGS_DIR.glob("*.npy"):
        slide_id = npy_path.stem
        emb = np.load(npy_path).astype(np.float32)
        if emb.ndim == 1 and emb.shape[0] == 768:
            records.append({"slide_id": slide_id, "emb": emb})
    # Also check student_final subdirectory
    sf_dir = CFG.EMBEDDINGS_DIR / "student_final"
    if sf_dir.exists():
        for npy_path in sf_dir.glob("*.npy"):
            slide_id = npy_path.stem
            emb = np.load(npy_path).astype(np.float32)
            if emb.ndim == 1 and emb.shape[0] == 768:
                records.append({"slide_id": slide_id, "emb": emb})

    print(f"[LOAD] Found {len(records)} slide embeddings")
    # Extract patient ID from TCGA barcode
    patient_embs = {}
    for rec in records:
        import re
        m = re.search(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', rec["slide_id"])
        if m:
            pid = m.group(1)
            patient_embs.setdefault(pid, []).append(rec["emb"])
    # Patient-level mean pooling
    rows = []
    for pid, embs in patient_embs.items():
        mean_emb = np.mean(embs, axis=0)
        row = {"patient_id": pid}
        for i in range(768):
            row[f"f{i}"] = float(mean_emb[i])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("patient_id")
    print(f"[LOAD] Aggregated to {len(df)} patients")
    return df

def load_data():
    print("\n== 1. LOADING DATA ==")
    df_emb = load_openslidefm_embeddings()
    df_manifest = pd.read_csv(CFG.MANIFEST)
    return df_emb, df_manifest

def prepare_dataset(df_emb, df_manifest):
    print("\n== 2. PREPARING DATASET ==")
    # Patient-to-cancer mapping from manifest
    df_manifest['patient_id'] = df_manifest['slide_id'].str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', expand=False)
    patient_cancer_map = df_manifest.groupby('patient_id')['cancer_code'].first().to_dict()

    # Extract TSS code for grouped CV
    df_manifest['tss'] = df_manifest['patient_id'].str.extract(r'TCGA-([A-Z0-9]{2})-', expand=False)
    patient_tss_map = df_manifest.groupby('patient_id')['tss'].first().to_dict()

    df_emb = df_emb.copy()
    df_emb['patient_id'] = df_emb.index
    df_emb['cancer_type'] = df_emb['patient_id'].map(patient_cancer_map)
    df_emb['tss'] = df_emb['patient_id'].map(patient_tss_map)
    df_emb = df_emb[df_emb['cancer_type'].notna()].copy()
    df_emb = df_emb[df_emb['tss'].notna()].copy()
    print(f"  Patients with labels + TSS: {len(df_emb)}")

    feature_cols = [c for c in df_emb.columns if c.startswith('f')]
    X = df_emb[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df_emb['cancer_type'].values)
    tss = df_emb['tss'].values

    print(f"  Feature matrix: {X.shape}")
    print(f"  Number of classes: {len(le.classes_)}")
    return X, y, tss, le, df_emb

# Bootstrap CI (1000 replicates)
def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42, metric_fn=accuracy_score):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

# Paired t-test with Bonferroni correction for ablation comparisons
# Used when comparing fold-wise scores between two configurations (e.g., multi-scale vs single-scale)
from scipy.stats import ttest_rel

def paired_test_bonferroni(scores_a, scores_b, n_comparisons=1):
    """Compare two sets of fold-wise scores with paired t-test and Bonferroni correction.
    scores_a, scores_b: arrays of per-fold metrics from two configurations
    n_comparisons: number of pairwise comparisons for Bonferroni adjustment
    Returns: (t_statistic, raw_p, corrected_p, mean_diff)
    """
    scores_a = np.array(scores_a); scores_b = np.array(scores_b)
    assert len(scores_a) == len(scores_b), "Fold counts must match for paired test"
    t_stat, p_val = ttest_rel(scores_a, scores_b)
    p_corrected = min(p_val * n_comparisons, 1.0)
    mean_diff = float(np.mean(scores_a) - np.mean(scores_b))
    return float(t_stat), float(p_val), float(p_corrected), mean_diff

def main():
    print("="*80)
    print(" TCGA EVALUATION — LogisticRegression, TSS-Grouped CV, 3 Seeds")
    print("="*80)

    df_emb, df_manifest = load_data()
    X, y, tss, le, df_labeled = prepare_dataset(df_emb, df_manifest)

    all_fold_results = []
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), len(le.classes_)), dtype=np.float32)
    oof_counts = np.zeros(len(y), dtype=int)

    # 3-seed repetition with StratifiedGroupKFold
    for seed in CFG.SEEDS:
        print(f"\n--- Seed {seed} ---")
        sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=tss)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # LogisticRegression with L2, C=1.0
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(
                C=1.0, penalty="l2", multi_class="multinomial",
                solver="saga", max_iter=5000, random_state=seed, n_jobs=-1
            )
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            y_probs = clf.predict_proba(X_test_s)

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            try:
                auc = roc_auc_score(y_test, y_probs, multi_class='ovr', average='macro')
            except Exception:
                auc = None

            all_fold_results.append({
                'seed': seed, 'fold': fold_idx, 'accuracy': acc,
                'balanced_accuracy': bal_acc, 'macro_f1': macro_f1, 'macro_auc': auc,
            })

            # Accumulate OOF predictions (last seed wins for OOF bootstrap)
            oof_preds[test_idx] = y_pred
            oof_probs[test_idx] = y_probs
            oof_counts[test_idx] += 1

            print(f"  Fold {fold_idx+1}/{CFG.N_FOLDS}: Acc={acc:.4f} | Bal-Acc={bal_acc:.4f} | F1={macro_f1:.4f}")

    # Aggregate across 15 evaluations (5 folds x 3 seeds)
    df_results = pd.DataFrame(all_fold_results)
    mean_acc = df_results['accuracy'].mean()
    std_acc = df_results['accuracy'].std()
    mean_bal = df_results['balanced_accuracy'].mean()
    std_bal = df_results['balanced_accuracy'].std()
    mean_f1 = df_results['macro_f1'].mean()
    std_f1 = df_results['macro_f1'].std()
    auc_vals = df_results['macro_auc'].dropna()
    mean_auc = auc_vals.mean() if len(auc_vals) else None
    std_auc = auc_vals.std() if len(auc_vals) else None

    # Bootstrap CIs on OOF predictions
    ci_acc = bootstrap_ci(y, oof_preds, n_boot=CFG.BOOTSTRAP_N, metric_fn=accuracy_score)

    print(f"\n{'='*80}")
    print(f" RESULTS (15 evaluations: 5-fold x 3 seeds)")
    print(f"{'='*80}")
    print(f"  Accuracy:          {mean_acc:.4f} +/- {std_acc:.4f}  95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]")
    print(f"  Balanced Accuracy: {mean_bal:.4f} +/- {std_bal:.4f}")
    print(f"  Macro F1:          {mean_f1:.4f} +/- {std_f1:.4f}")
    if mean_auc is not None:
        print(f"  Macro AUROC (OvR): {mean_auc:.4f} +/- {std_auc:.4f}")

    # Save results
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'method': 'LogisticRegression_C1.0_L2',
        'cv': 'StratifiedGroupKFold_TSS',
        'seeds': CFG.SEEDS,
        'n_evaluations': len(df_results),
        'accuracy': {'mean': float(mean_acc), 'std': float(std_acc), 'ci95': list(ci_acc)},
        'balanced_accuracy': {'mean': float(mean_bal), 'std': float(std_bal)},
        'macro_f1': {'mean': float(mean_f1), 'std': float(std_f1)},
        'macro_auc_ovr': {'mean': float(mean_auc), 'std': float(std_auc)} if mean_auc else None,
    }
    with open(str(CFG.OUTPUT / 'test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    df_results.to_csv(str(CFG.OUTPUT / 'fold_results.csv'), index=False)

    # Per-class report on last seed's OOF
    report = classification_report(y, oof_preds, target_names=le.classes_, output_dict=True)
    pd.DataFrame(report).T.to_csv(str(CFG.OUTPUT / 'per_class_metrics.csv'))

    cm = confusion_matrix(y, oof_preds)
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(str(CFG.OUTPUT / 'confusion_matrix.csv'))

    print(f"\n  Results saved to: {CFG.OUTPUT}")
    print("  Script 7 complete.")

if __name__ == "__main__":
    main()


# SCRIPT 08: SLIDE EMBEDDINGS EXPORT
# Exports slide-level embeddings using the trained MILTransformer + max pooling.
import os, sys, json, time, gc
from pathlib import Path
from datetime import datetime
import subprocess, warnings
warnings.filterwarnings("ignore")

EMB_DIR = WORKSPACE / "embeddings"
MANIFESTS = WORKSPACE / "manifests"
DIAG = WORKSPACE / "diagnostics"
FEAT05 = WORKSPACE / "features" / "scale0p5"
FEAT20 = WORKSPACE / "features" / "scale2p0"
WEIGHTS_DIR = WORKSPACE / "weights"
for p in [EMB_DIR, DIAG]: p.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd

def load_slide_sets():
    sets = {}
    def _load_csv(name):
        p = MANIFESTS / f"manifest_{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            if "slide_id" in df.columns:
                return set(df["slide_id"])
            elif "filename" in df.columns:
                return set(Path(x).stem for x in df["filename"])
        return set()
    sets["tcga"] = _load_csv("tcga")
    sets["camelyon16"] = _load_csv("camelyon16")
    sets["camelyon17"] = _load_csv("camelyon17")
    return sets

SLIDESETS = load_slide_sets()

def available_two_scale_ids():
    s05 = set(p.stem for p in FEAT05.glob("*.npy"))
    s20 = set(p.stem for p in FEAT20.glob("*.npy"))
    return sorted(list(s05 & s20))

TWO_SCALE_IDS = available_two_scale_ids()

def dataset_of(slide_id):
    if slide_id in SLIDESETS.get("camelyon16", set()): return "CAMELYON16"
    if slide_id in SLIDESETS.get("camelyon17", set()): return "CAMELYON17"
    if slide_id in SLIDESETS.get("tcga", set()):       return "TCGA"
    return "OTHER"

# Load trained MILTransformer and use it for embedding export
def load_trained_model():
    """Load the trained MILTransformer from Script 6 checkpoint."""
    import torch
    from safetensors.torch import load_file as load_safetensors

    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt = WEIGHTS_DIR / "latest.txt"
    if not txt.exists():
        print("[WARN] No trained model checkpoint found (weights/latest.txt missing).")
        print("[WARN] Falling back to raw feature mean pooling (fallback).")
        return None, device

    ckpt_name = txt.read_text(encoding="utf-8").strip()
    ckpt_path = WEIGHTS_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint {ckpt_path} not found. Falling back to raw mean.")
        return None, device

    # Reconstruct MILTransformer (must match Script 6 architecture)
    model = MILTransformer(d_model=768, n_heads=8, n_layers=6, ff_mult=4, dropout=0.1).to(device)
    sd = load_safetensors(str(ckpt_path))
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"[OK] Loaded trained MILTransformer from {ckpt_path.name}")
    return model, device

def embed_one_with_model(slide_id, model, device):
    """Produce slide embedding using trained transformer + max pooling."""
    import torch

    f05_path = FEAT05 / f"{slide_id}.npy"
    f20_path = FEAT20 / f"{slide_id}.npy"
    if not (f05_path.exists() and f20_path.exists()):
        return {"slide_id": slide_id, "ok": False, "reason": "missing_feature_file"}

    try:
        a = np.load(f05_path, mmap_mode="r").astype(np.float32)
        b = np.load(f20_path, mmap_mode="r").astype(np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 768 or b.shape[1] != 768:
            return {"slide_id": slide_id, "ok": False, "reason": f"bad_shape a{tuple(a.shape)} b{tuple(b.shape)}"}

        # Budget-cap to 1200 + 400 tokens (deterministic per slide)
        rng = np.random.default_rng(1337 + hash(slide_id) % (2**16))
        if a.shape[0] > 1200:
            idx = rng.choice(a.shape[0], 1200, replace=False)
            a = a[idx]
        if b.shape[0] > 400:
            idx = rng.choice(b.shape[0], 400, replace=False)
            b = b[idx]

        feats = np.concatenate([a, b], axis=0)  # [T, 768]

        if model is not None:
            # Compute positional encoding from tile metadata
            T = feats.shape[0]
            meta05_path = FEAT05 / f"{slide_id}_meta.parquet"
            meta20_path = FEAT20 / f"{slide_id}_meta.parquet"
            if meta05_path.exists() and meta20_path.exists():
                # Reuse load_meta and compute_mm_xy from Script 6 (already in scope)
                m05 = load_meta(meta05_path)
                m20 = load_meta(meta20_path)
                m05 = m05.iloc[:a.shape[0]]
                m20 = m20.iloc[:b.shape[0]]
                mmxy = np.concatenate([compute_mm_xy(m05), compute_mm_xy(m20)], axis=0)
            else:
                mmxy = np.zeros((T, 2), dtype=np.float32)
            scl = np.concatenate([
                np.full((a.shape[0], 1), 0.5, dtype=np.float32),
                np.full((b.shape[0], 1), 2.0, dtype=np.float32)
            ], axis=0)

            feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            mmxy_t = torch.from_numpy(mmxy).unsqueeze(0).to(device)
            scl_t = torch.from_numpy(scl).unsqueeze(0).to(device)
            pad_t = torch.zeros(1, T, dtype=torch.bool, device=device)

            with torch.no_grad():
                g_proj, _, _, _ = model(feats_t, mmxy_t, scl_t, pad_t)
            emb = g_proj.squeeze(0).cpu().numpy().astype(np.float32)
        else:
            # Fallback: simple mean (fallback)
            emb = feats.mean(axis=0).astype(np.float32)

        ds = dataset_of(slide_id)
        out_dir = EMB_DIR / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slide_id}.npy"
        np.save(out_path, emb)

        return {
            "slide_id": slide_id, "dataset": ds, "ok": True, "path_emb": str(out_path),
            "t05": int(a.shape[0]), "t20": int(b.shape[0]), "norm": float(np.linalg.norm(emb)),
        }
    except Exception as e:
        return {"slide_id": slide_id, "ok": False, "reason": f"{type(e).__name__}:{e}"}

CONFIG_S8 = {
    "only_datasets": None,  # Process ALL datasets (TCGA + external)
    "workers": 16,
    "print_every": 200
}

def main_s8():
    print("== Script 8 — Slide Embeddings Export (using trained transformer) ==")

    model, device = load_trained_model()

    target_ids = TWO_SCALE_IDS
    print(f"[PLAN] Slides with 2-scale features: {len(target_ids)}")
    if len(target_ids) == 0:
        print("[EXIT] Nothing to export."); return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0=time.time(); done=0; ok=0; bad=0; rows=[]
    # Note: if using GPU model, serialize to avoid CUDA contention
    if model is not None:
        for sid in target_ids:
            r = embed_one_with_model(sid, model, device)
            rows.append(r); done += 1
            ok += int(r.get("ok", False)); bad += int(not r.get("ok", False))
            if done % CONFIG_S8["print_every"] == 0:
                print(f"[{done:6d}/{len(target_ids)}] ok={ok} bad={bad}")
    else:
        # CPU fallback — can parallelize
        with ThreadPoolExecutor(max_workers=CONFIG_S8["workers"]) as ex:
            futs = {ex.submit(embed_one_with_model, sid, None, "cpu"): sid for sid in target_ids}
            for fut in as_completed(futs):
                r = fut.result(); rows.append(r); done += 1
                ok += int(r.get("ok", False)); bad += int(not r.get("ok", False))

    df = pd.DataFrame(rows)
    for ds, g in df[df["ok"]==True].groupby("dataset"):
        out_csv = EMB_DIR / f"{ds.lower()}_index.csv"
        g[["slide_id","path_emb","t05","t20","norm"]].to_csv(out_csv, index=False)
        print(f"[OK] Index for {ds}: {len(g)} -> {out_csv}")

    print(f"\n[DONE] Script 8 complete. ok={ok} bad={bad}")

main_s8()




# SCRIPT 09: CAMELYON17 PN STAGING (LOCO-CV)
# Bootstrap CI: 1000 replicates
import os, sys, re, json, math, time, subprocess, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

WS = WORKSPACE
RAW = WS / "Raw Data" / "CAMELYON17"
EMB_INDEX = WS / "embeddings" / "camelyon17_index.csv"
MANIFEST_CAM17 = WS / "manifests" / "manifest_camelyon17.csv"
OUTDIR = WS / "results" / "cam17_pn_eval" / "ablations"
OUTDIR.mkdir(parents=True, exist_ok=True)

import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

try:
    import mord
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "mord"])
    import mord

CFG_S9 = {
  "Cs": [0.1, 0.3, 1.0, 3.0, 10.0],
  "random_state": 17,
  "max_iter": 4000,
  "n_jobs": 4,
  "fallback_k": 5,
  "boots": 1000  # 1000 bootstrap replicates
}

def _now(): return datetime.now().isoformat(timespec="seconds")

def guess_label_csv(root: Path):
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
        n=int(s); return n if 0<=n<=9 else None
    return None

def load_labels():
    cand = guess_label_csv(RAW)
    if cand is None:
        raise FileNotFoundError("Place a CAMELYON17 labels CSV under Raw Data/CAMELYON17/")
    df = pd.read_csv(cand)
    df.columns = [c.lower() for c in df.columns]
    if "case" in df.columns and "patient" not in df.columns: df["patient"] = df["case"]
    if "pn" not in df.columns and "stage" in df.columns: df["pn"] = df["stage"]
    assert "patient" in df.columns and "pn" in df.columns
    df["patient"] = df["patient"].astype(str)
    df["patient"] = df["patient"].str.extract(r"(patient[_\-]?\d+)", expand=False).fillna(df["patient"])
    df["pn_int"] = df["pn"].apply(_pn_to_int)
    df = df.dropna(subset=["pn_int"]).copy()
    if "center" not in df.columns and "centerid" in df.columns: df["center"] = df["centerid"]
    if "center" in df.columns:
        df["center"] = df["center"].apply(_center_from_any)
    else:
        df["center"] = None
    if MANIFEST_CAM17.exists() and df["center"].isna().mean() > 0.1:
        man = pd.read_csv(MANIFEST_CAM17)
        pcol = None
        for c in ["path","filepath","fullpath","filename"]:
            if c in man.columns.str.lower().tolist():
                pcol = man.columns[[cc.lower()==c for cc in man.columns]].tolist()[0]; break
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
    return pd.read_csv(EMB_INDEX)

def patient_from_slide(sid):
    s = sid.lower()
    m = re.search(r"(patient[_\-]?\d+)", s)
    return m.group(1) if m else s.split("_node")[0]

def load_embeddings(df_idx):
    X=[]; S=[]; P=[]
    for sid, p in zip(df_idx["slide_id"], df_idx["path_emb"]):
        try:
            v = np.load(p).astype(np.float32)
            if v.ndim!=1 or v.shape[0]!=768: continue
        except: continue
        X.append(v); S.append(str(sid)); P.append(patient_from_slide(str(sid)))
    X = np.stack(X, axis=0) if X else np.zeros((0,768), dtype=np.float32)
    return X, np.array(S), np.array(P)

def aggregate_patient(X, pats):
    uniq = pd.unique(pats); P = []; order=[]
    for u in uniq:
        idx = np.where(pats==u)[0]
        P.append(X[idx].max(axis=0)); order.append(u)
    return np.stack(P,axis=0) if P else np.zeros((0,768),np.float32), np.array(order)

def fit_predict(model_key, C, Xtr, ytr, Xte):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr); Xte = scaler.transform(Xte)
    if model_key=="logreg_l2":
        clf = LogisticRegression(multi_class="multinomial", solver="saga", penalty="l2", C=C,
                                 max_iter=CFG_S9["max_iter"], n_jobs=CFG_S9["n_jobs"],
                                 class_weight="balanced", random_state=CFG_S9["random_state"])
    elif model_key=="logreg_l1":
        clf = LogisticRegression(multi_class="multinomial", solver="saga", penalty="l1", C=C,
                                 max_iter=CFG_S9["max_iter"], n_jobs=CFG_S9["n_jobs"],
                                 class_weight="balanced", random_state=CFG_S9["random_state"])
    elif model_key=="ridge":
        clf = RidgeClassifier(alpha=1.0/max(C,1e-6), class_weight="balanced", random_state=CFG_S9["random_state"])
    elif model_key=="ordinal":
        clf = mord.LogisticAT(alpha=1.0/max(C,1e-6))
    else: raise ValueError("unknown model_key")
    clf.fit(Xtr, ytr); return clf.predict(Xte)

def qw_kappa(y, yhat): return cohen_kappa_score(y, yhat, weights="quadratic")

def bootstrap_ci_s9(y, yhat, groups, B=1000, seed=123):
    rng = np.random.default_rng(seed)
    pts = pd.unique(groups)
    if len(pts)==0: return (float("nan"), float("nan"))
    mapping = {p: np.where(groups==p)[0] for p in pts}
    vals=[]
    for _ in range(B):
        idx=[]
        for _ in range(len(pts)):
            pick = rng.choice(pts); idx.extend(mapping[pick])
        idx = np.array(idx, dtype=int)
        vals.append(qw_kappa(y[idx], yhat[idx]))
    return float(np.nanpercentile(vals,2.5)), float(np.nanpercentile(vals,97.5))

print("== Script 9 — CAMELYON17 pN LOCO ablation ==")
df_idx = load_emb_index()
X_s, slide_ids, pats_s = load_embeddings(df_idx)
Xp, patients = aggregate_patient(X_s, pats_s)
df_lbl = load_labels()

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
print(f"[MODE] {'LOCO' if use_loco else str(CFG_S9['fallback_k'])+'-fold CV'}  centers={sorted(centers)}")
print(f"[DATA] patients={len(patients)}  class_counts={pd.Series(y).value_counts().sort_index().to_dict()}")

models = [("ordinal",), ("logreg_l2",), ("logreg_l1",), ("ridge",)]
grid = [(m[0], float(C)) for m in models for C in CFG_S9["Cs"]]

rows_s9=[]; all_preds = {}
for mk, C in grid:
    preds=[]; truths=[]; groups=[]; per_fold=[]
    if use_loco:
        for cc in sorted(pd.unique(c)):
            if cc==-1: continue
            te = np.where(c==cc)[0]; tr = np.where(c!=cc)[0]
            if len(te)==0 or len(tr)==0: continue
            yhat = fit_predict(mk, C, Xp[tr], y[tr], Xp[te])
            k = qw_kappa(y[te], yhat)
            per_fold.append(("CEN"+str(int(cc)), int(len(te)), float(k)))
            preds.extend(yhat.tolist()); truths.extend(y[te].tolist()); groups.extend(patients[te].tolist())
    else:
        skf = StratifiedKFold(n_splits=CFG_S9["fallback_k"], shuffle=True, random_state=CFG_S9["random_state"])
        fold=0
        for tr, te in skf.split(Xp, y):
            fold+=1; yhat = fit_predict(mk, C, Xp[tr], y[tr], Xp[te])
            k = qw_kappa(y[te], yhat)
            per_fold.append(("FOLD"+str(fold), int(len(te)), float(k)))
            preds.extend(yhat.tolist()); truths.extend(y[te].tolist()); groups.extend(patients[te].tolist())
    preds = np.array(preds, dtype=int); truths = np.array(truths, dtype=int); groups = np.array(groups)
    mean_k = float(np.mean([r[2] for r in per_fold])) if per_fold else float("nan")
    rows_s9.append({"model": mk, "C": C, "kappa_qw_mean": mean_k, "folds": len(per_fold),
                     "detail": "; ".join([f"{lab}:n={n}|k={k:.3f}" for lab,n,k in per_fold])})
    all_preds[(mk,C)] = (truths, preds, groups)

df_s9 = pd.DataFrame(rows_s9).sort_values("kappa_qw_mean", ascending=False)
df_s9.to_csv(OUTDIR / "ablations_summary.csv", index=False)

best = df_s9.iloc[0].to_dict()
bkey = (best["model"], float(best["C"]))
y_true_s9, y_pred_s9, pgroup = all_preds[bkey]
ci_lo, ci_hi = bootstrap_ci_s9(y_true_s9, y_pred_s9, pgroup, B=CFG_S9["boots"])
overall_k = qw_kappa(y_true_s9, y_pred_s9)

pd.DataFrame({"patient": pgroup, "y_true": y_true_s9, "y_pred": y_pred_s9}).to_csv(OUTDIR/"best_patient_predictions.csv", index=False)
meta_s9 = {
  "time": _now(), "mode": "LOCO" if use_loco else f"{CFG_S9['fallback_k']}-fold-CV",
  "best_model": best["model"], "best_C": float(best["C"]),
  "kappa_qw_mean_cv": float(best["kappa_qw_mean"]) if not math.isnan(best["kappa_qw_mean"]) else None,
  "overall_kappa_qw": float(overall_k),
  "kappa_ci95": [float(ci_lo), float(ci_hi)],
}
(Path(OUTDIR/"best_config.json")).write_text(json.dumps(meta_s9, indent=2), encoding="utf-8")
print(json.dumps(meta_s9, indent=2))
print(f"[OK] Script 9 complete.")


# SCRIPT 09A: CAMELYON16 METASTASIS DETECTION (5-FOLD CV)
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
import json

OUTDIR_CAM16 = WORKSPACE / "results" / "cam16_eval"
OUTDIR_CAM16.mkdir(parents=True, exist_ok=True)
EMB_INDEX_CAM16 = WORKSPACE / "embeddings" / "camelyon16_index.csv"

def main_cam16():
    print("== Script 9A — CAMELYON16 Metastasis Detection (5-fold CV) ==")

    if not EMB_INDEX_CAM16.exists():
        print(f"[SKIP] {EMB_INDEX_CAM16} not found. Run Script 08 first with CAMELYON16 data.")
        return

    df_idx = pd.read_csv(EMB_INDEX_CAM16)
    # Load embeddings
    X = []; slide_ids = []
    for _, row in df_idx.iterrows():
        try:
            v = np.load(row["path_emb"]).astype(np.float32)
            if v.ndim == 1 and v.shape[0] == 768:
                X.append(v); slide_ids.append(row["slide_id"])
        except Exception:
            pass
    X = np.stack(X, axis=0) if X else np.zeros((0,768), dtype=np.float32)

    # Load labels: slides with "tumor" in name or from a labels file
    # CAMELYON16 convention: tumor_XXX = positive, normal_XXX = negative
    y = np.array([1 if "tumor" in str(sid).lower() else 0 for sid in slide_ids], dtype=int)
    print(f"[DATA] {len(X)} slides, {y.sum()} tumor, {(1-y).sum()} normal")

    if len(X) < 10:
        print("[SKIP] Too few slides for evaluation."); return

    # 5-fold stratified CV with LogisticRegression(C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y), dtype=np.float64)
    oof_preds = np.zeros(len(y), dtype=int)

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(C=1.0, penalty='l2', max_iter=5000, random_state=42)
        clf.fit(X_tr, y[tr])
        oof_probs[te] = clf.predict_proba(X_te)[:, 1]
        oof_preds[te] = clf.predict(X_te)

    auroc = roc_auc_score(y, oof_probs)
    acc = accuracy_score(y, oof_preds)
    f1 = f1_score(y, oof_preds)
    avg_prec = average_precision_score(y, oof_probs)

    # Bootstrap CI (1000 replicates)
    rng = np.random.default_rng(42)
    auc_boots = []
    for _ in range(1000):
        idx = rng.choice(len(y), size=len(y), replace=True)
        try: auc_boots.append(roc_auc_score(y[idx], oof_probs[idx]))
        except: pass
    ci_lo = float(np.percentile(auc_boots, 2.5))
    ci_hi = float(np.percentile(auc_boots, 97.5))

    results = {
        "auroc": float(auroc), "auroc_ci95": [ci_lo, ci_hi],
        "accuracy": float(acc), "f1": float(f1), "avg_precision": float(avg_prec),
        "n_slides": int(len(y)), "n_tumor": int(y.sum()), "n_normal": int((1-y).sum()),
    }
    (OUTDIR_CAM16 / "cam16_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  AUROC: {auroc:.3f}  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"  Accuracy: {acc:.3f}  F1: {f1:.3f}  Avg Precision: {avg_prec:.3f}")
    print(f"  Results saved to: {OUTDIR_CAM16}")

main_cam16()




# SCRIPT 10: PANDA FEATURE PROCESSING
import os, sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIRS_S10 = {
    "features_05": WORKSPACE / "features" / "panda" / "scale0p5",
    "features_20": WORKSPACE / "features" / "panda" / "scale2p0",
    "results": WORKSPACE / "results" / "panda",
    "logs": WORKSPACE / "logs" / "panda"
}
for d in OUTPUT_DIRS_S10.values(): d.mkdir(parents=True, exist_ok=True)

N_WORKERS_S10 = min(cpu_count() - 1, 8)
BATCH_SIZE_S10 = 128
USE_MIXED_PRECISION_S10 = True

print(f"[INFO] Script 10 — PANDA feature processing, {cpu_count()} CPUs, using {N_WORKERS_S10} workers")

def check_already_processed(image_id):
    feat_05 = OUTPUT_DIRS_S10["features_05"] / f"{image_id}.npy"
    feat_20 = OUTPUT_DIRS_S10["features_20"] / f"{image_id}.npy"
    if feat_05.exists() and feat_20.exists():
        try:
            f05 = np.load(feat_05, mmap_mode='r')
            f20 = np.load(feat_20, mmap_mode='r')
            if f05.shape[1] == 768 and f20.shape[1] == 768: return True
        except: pass
    return False

def process_batch_s10(tile_batch, model, device):
    import torch
    tensors = []
    for tile in tile_batch:
        tile_array = np.array(tile).astype(np.float32) / 255.0
        tensor = torch.from_numpy(tile_array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensors.append(tensor)
    batch_tensor = torch.stack(tensors).to(device, non_blocking=True)
    if device != "cpu": batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
    with torch.no_grad():
        if USE_MIXED_PRECISION_S10 and device != "cpu":
            with torch.cuda.amp.autocast(): features = model(batch_tensor)
        else: features = model(batch_tensor)
        features = features.cpu().numpy()
    return features

def process_single_slide_s10(args):
    row, device_id = args
    import torch
    import torchvision.models as tvm
    import torch.nn as nn
    from PIL import Image
    import openslide

    device = f"cuda:{device_id % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    image_id = row['image_id']; image_path = Path(row['image_path'])
    if check_already_processed(image_id): return image_id, "skipped", 0

    class ConvNeXtTinyFeats_S10(nn.Module):
        def __init__(self):
            super().__init__()
            model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
            self.features = model.features; self.gap = nn.AdaptiveAvgPool2d(1)
            self.eval()
            for p in self.parameters(): p.requires_grad = False
        @torch.no_grad()
        def forward(self, x): return self.gap(self.features(x)).flatten(1)

    try:
        model = ConvNeXtTinyFeats_S10().to(device)
        if device != "cpu": model = model.to(memory_format=torch.channels_last)
        slide = openslide.OpenSlide(str(image_path))
        TILE_SIZE = 256; STRIDE = 224; SCALES = [0.5, 2.0]; MAX_TILES = {0.5: 1200, 2.0: 400}
        tiles_extracted = 0
        for scale in SCALES:
            scale_key = f"features_{str(scale).replace('.','').replace('p','')}"
            feat_path = OUTPUT_DIRS_S10[scale_key] / f"{image_id}.npy"
            if feat_path.exists(): continue
            base_mpp = 0.5; target_downsample = scale / base_mpp
            level = slide.get_best_level_for_downsample(target_downsample)
            actual_downsample = slide.level_downsamples[level]
            level_w, level_h = slide.level_dimensions[level]
            tiles = []; tile_batch = []
            for y in range(0, level_h - TILE_SIZE + 1, STRIDE):
                for x in range(0, level_w - TILE_SIZE + 1, STRIDE):
                    if len(tiles) >= MAX_TILES[scale]: break
                    x0 = int(x * actual_downsample); y0 = int(y * actual_downsample)
                    tile = slide.read_region((x0, y0), level, (TILE_SIZE, TILE_SIZE)).convert('RGB')
                    tile_np = np.array(tile)
                    if tile_np.mean() < 235 and tile_np.std() > 15:
                        tile_224 = tile.resize((224, 224), Image.BILINEAR)
                        tile_batch.append(tile_224)
                        if len(tile_batch) >= BATCH_SIZE_S10:
                            batch_features = process_batch_s10(tile_batch, model, device)
                            tiles.extend(batch_features); tile_batch = []; tiles_extracted += len(batch_features)
                if len(tiles) >= MAX_TILES[scale]: break
            if tile_batch:
                batch_features = process_batch_s10(tile_batch, model, device)
                tiles.extend(batch_features); tiles_extracted += len(batch_features)
            if tiles: np.save(feat_path, np.vstack(tiles).astype(np.float16))
            else: np.save(feat_path, np.zeros((0, 768), dtype=np.float16))
        slide.close()
        if device != "cpu": torch.cuda.empty_cache()
        return image_id, "success", tiles_extracted
    except Exception as e:
        return image_id, f"error: {str(e)}", 0

def extract_features_panda(df, max_slides=None):
    import torch
    pending = [row for _, row in df[df['image_exists']].iterrows() if not check_already_processed(row['image_id'])]
    if max_slides: pending = pending[:max_slides]
    if not pending: print("All slides already processed!"); return
    print(f"Found {len(pending)} slides to process with {N_WORKERS_S10} workers")
    cuda_available = torch.cuda.is_available()
    worker_args = [(row, i % N_WORKERS_S10 if cuda_available else 0) for i, row in enumerate(pending)]
    successful = 0; failed = []; start_time = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS_S10) as executor:
        futures = {executor.submit(process_single_slide_s10, args): args[0]['image_id'] for args in worker_args}
        for future in as_completed(futures):
            try:
                slide_id, status, tiles = future.result()
                if status == "success": successful += 1
                elif status != "skipped": failed.append((slide_id, status))
            except Exception as e: failed.append((futures[future], str(e)))
    elapsed = time.time() - start_time
    print(f"Extraction complete in {elapsed/60:.1f} min | Successful: {successful} | Failed: {len(failed)}")
    if failed:
        pd.DataFrame(failed, columns=['image_id', 'error']).to_csv(OUTPUT_DIRS_S10["logs"] / "failed_extractions.csv", index=False)

# Load manifest and run
manifest_path_s10 = OUTPUT_DIRS_S10["logs"] / "panda_manifest.csv"
if manifest_path_s10.exists():
    df_s10 = pd.read_csv(manifest_path_s10)
else:
    train_csv = PANDA_ROOT / "train.csv"
    if train_csv.exists():
        df_s10 = pd.read_csv(train_csv)
        df_s10['image_path'] = df_s10['image_id'].apply(lambda x: str(PANDA_ROOT / "train_images" / f"{x}.tiff"))
        df_s10['image_exists'] = df_s10['image_path'].apply(lambda x: Path(x).exists())
        df_s10.to_csv(manifest_path_s10, index=False)
    else:
        print("[WARN] PANDA train.csv not found; skipping Script 10.")
        df_s10 = pd.DataFrame()

if len(df_s10):
    already = sum(1 for _, r in df_s10.iterrows() if r.get('image_exists', False) and check_already_processed(r['image_id']))
    print(f"Total: {len(df_s10)} | With images: {df_s10.get('image_exists', pd.Series()).sum()} | Already done: {already}")
    extract_features_panda(df_s10)
print("[DONE] Script 10 complete.")


# SCRIPT 11: PANDA GLEASON GRADING (MIL)
# Token budgets: 1200 @ 0.5 um/px, 400 @ 2.0 um/px (1600 total)
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

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def qwk_s11(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def now_s11(): return time.strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class PandaConfig:
    WORKSPACE: Path = WORKSPACE
    PANDA_ROOT: Path = WORKSPACE / "Validation Data" / "PANDA"
    FEAT_05: Path = WORKSPACE / "features" / "panda" / "scale0p5"
    FEAT_20: Path = WORKSPACE / "features" / "panda" / "scale2p0"
    OUTPUT:  Path = WORKSPACE / "results" / "panda_mil_088"
    INPUT_DIM: int = 768; NUM_CLASSES: int = 6; NUM_HEADS: int = 8
    POOL_DROPOUT: float = 0.1; TOPK_RATIO: float = 0.15; TOPK_MIN: int = 24
    HIDDEN_DIM: int = 512; FUSE_DROPOUT: float = 0.15
    EPOCHS: int = 30; WARMUP_EPOCHS: int = 2; BATCH_SLIDES: int = 24
    LR: float = 3e-4; WD: float = 1e-4; MAX_GRAD_NORM: float = 1.0; AMP: bool = True
    NUM_WORKERS: int = 0 if platform.system()=="Windows" else max(4, (os.cpu_count() or 8)-2)
    PIN_MEMORY: bool = torch.cuda.is_available(); PROVIDER_AWARE: bool = True; PATIENCE: int = 7
    LABEL_SMOOTH: float = 0.05; FOCAL_ALPHA: float = 0.25; FOCAL_GAMMA: float = 2.0
    ORDINAL_LAM: float = 0.25; EXP_LAM: float = 0.02
    NOISE_STD: float = 0.01; FEAT_DROPOUT_P: float = 0.05
    MAX_TILES_05: int = 1200; MAX_TILES_20: int = 400
    N_FOLDS: int = 5; SEEDS: Tuple[int,...] = (42, 777, 1337); EMA_DECAY: float = 0.999
    PRINT_EVERY: int = 25
    def __post_init__(self):
        (self.OUTPUT / "models").mkdir(parents=True, exist_ok=True)
        (self.OUTPUT / "logs").mkdir(parents=True, exist_ok=True)

CFG_S11 = PandaConfig()
DEVICE_S11 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NAME_S11 = torch.cuda.get_device_name(0) if DEVICE_S11.type=="cuda" else "CPU"

class PANDADatasetIndex:
    def __init__(self, cfg):
        df = pd.read_csv(cfg.PANDA_ROOT / "train.csv")
        df["isup_grade"] = df["isup_grade"].fillna(0).astype(int)
        has05 = df["image_id"].apply(lambda s: (cfg.FEAT_05 / f"{s}.npy").exists())
        has20 = df["image_id"].apply(lambda s: (cfg.FEAT_20 / f"{s}.npy").exists())
        df = df[has05 & has20].reset_index(drop=True)
        self.df = df
        print(f"[DATA] Slides (both scales): {len(df)}")
    def splits(self):
        n = len(self.df)
        key = self.df["isup_grade"].astype(str) + "_" + self.df["data_provider"].astype(str) if CFG_S11.PROVIDER_AWARE and "data_provider" in self.df.columns else self.df["isup_grade"]
        skf = StratifiedKFold(n_splits=CFG_S11.N_FOLDS, shuffle=True, random_state=42)
        return list(skf.split(np.arange(n), key))

class SlideBagDataset(Dataset):
    def __init__(self, df, train):
        self.df = df.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.df)
    def _load(self, root, sid, budget, train):
        arr = np.load(root / f"{sid}.npy", mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != CFG_S11.INPUT_DIM:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1, CFG_S11.INPUT_DIM)
        T = arr.shape[0]
        if T == 0: arr = np.zeros((1, CFG_S11.INPUT_DIM), dtype=np.float32); T = 1
        if T > budget:
            if train: idx = np.random.choice(T, budget, replace=False); arr = arr[idx]
            else: arr = arr[:budget]
        return np.asarray(arr, dtype=np.float32).copy(order="C")
    def __getitem__(self, i):
        r = self.df.iloc[i]; sid = r["image_id"]; y = int(r["isup_grade"])
        f05 = self._load(CFG_S11.FEAT_05, sid, CFG_S11.MAX_TILES_05, self.train)
        f20 = self._load(CFG_S11.FEAT_20, sid, CFG_S11.MAX_TILES_20, self.train)
        if self.train:
            if np.random.rand() < 0.4:
                f05 += np.random.normal(0, CFG_S11.NOISE_STD, f05.shape).astype(np.float32)
                f20 += np.random.normal(0, CFG_S11.NOISE_STD, f20.shape).astype(np.float32)
            if np.random.rand() < 0.3:
                f05 *= (np.random.rand(*f05.shape) > CFG_S11.FEAT_DROPOUT_P).astype(np.float32)
                f20 *= (np.random.rand(*f20.shape) > CFG_S11.FEAT_DROPOUT_P).astype(np.float32)
        return {"feat_05": torch.from_numpy(f05), "feat_20": torch.from_numpy(f20),
                "label": torch.tensor(y, dtype=torch.long), "id": sid, "prov": r.get("data_provider", "NA")}

def collate_bags(batch):
    B = len(batch); D = batch[0]["feat_05"].shape[1]
    n1 = [b["feat_05"].shape[0] for b in batch]; n2 = [b["feat_20"].shape[0] for b in batch]
    N1, N2 = max(n1), max(n2)
    f05 = torch.zeros(B, N1, D); f20 = torch.zeros(B, N2, D)
    m05 = torch.ones(B, N1, dtype=torch.bool); m20 = torch.ones(B, N2, dtype=torch.bool)
    for i,b in enumerate(batch):
        a = b["feat_05"]; f05[i,:a.size(0)] = a; m05[i,:a.size(0)] = False
        c = b["feat_20"]; f20[i,:c.size(0)] = c; m20[i,:c.size(0)] = False
    y = torch.stack([b["label"] for b in batch],0)
    return {"feat_05":f05, "mask_05":m05, "feat_20":f20, "mask_20":m20,
            "label":y, "ids":[b["id"] for b in batch], "prov":[b["prov"] for b in batch]}

class MultiHeadPool(nn.Module):
    def __init__(self, d_model, num_heads, pdrop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=pdrop, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*2), nn.ReLU(), nn.Dropout(pdrop), nn.Linear(d_model*2, d_model))
        self.norm2 = nn.LayerNorm(d_model); self.dropout = nn.Dropout(pdrop)
        self.scorer = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Linear(d_model//2, 1))
    def forward(self, x, mask):
        x = x.float()
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)
        h = self.norm1(x + self.dropout(attn_out)); f = self.ffn(h); h = self.norm2(h + self.dropout(f))
        scores = self.scorer(h).squeeze(-1).masked_fill(mask, -1e4)
        with torch.no_grad():
            nonpad = (~mask).sum(1)
            k = (nonpad.float() * CFG_S11.TOPK_RATIO).floor().clamp(min=CFG_S11.TOPK_MIN)
            k = torch.minimum(k, nonpad.float()).to(torch.long).clamp(min=1)
        pooled = []
        for b in range(x.size(0)):
            k_b = int(k[b].item())
            topv, topi = torch.topk(scores[b, :nonpad[b]], k_b, dim=0)
            sel = h[b, topi]; w = F.softmax(topv, dim=0).unsqueeze(1)
            pooled.append((sel * w).sum(0))
        return torch.stack(pooled, dim=0)

class MILModelPanda(nn.Module):
    def __init__(self, in_dim=CFG_S11.INPUT_DIM, num_classes=CFG_S11.NUM_CLASSES):
        super().__init__()
        self.pool05 = MultiHeadPool(in_dim, CFG_S11.NUM_HEADS, CFG_S11.POOL_DROPOUT)
        self.pool20 = MultiHeadPool(in_dim, CFG_S11.NUM_HEADS, CFG_S11.POOL_DROPOUT)
        self.fuse = nn.Sequential(nn.Linear(in_dim*2, CFG_S11.HIDDEN_DIM), nn.LayerNorm(CFG_S11.HIDDEN_DIM),
                                  nn.ReLU(), nn.Dropout(CFG_S11.FUSE_DROPOUT),
                                  nn.Linear(CFG_S11.HIDDEN_DIM, CFG_S11.HIDDEN_DIM), nn.ReLU(), nn.Dropout(CFG_S11.FUSE_DROPOUT))
        self.classifier = nn.Linear(CFG_S11.HIDDEN_DIM, num_classes)
    def forward(self, f05, m05, f20, m20):
        e05 = self.pool05(f05, m05); e20 = self.pool20(f20, m20)
        h = self.fuse(torch.cat([e05, e20], dim=1))
        return {"logits": self.classifier(h).float(), "emb": h}

def smooth_one_hot(y, num_classes, eps):
    with torch.no_grad():
        target = torch.full((y.size(0), num_classes), eps/num_classes, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0 - eps + eps/num_classes)
    return target

def focal_ce_loss(logits, targets, class_weights=None, alpha=0.25, gamma=2.0, label_smooth=0.05):
    K = logits.size(1); logp = F.log_softmax(logits, dim=1); p = logp.exp()
    T = smooth_one_hot(targets, K, label_smooth)
    pt = (p * T).sum(dim=1).clamp(min=1e-8); ce = -(T * logp).sum(dim=1)
    focal = (alpha * (1 - pt)**gamma) * ce
    if class_weights is not None: focal = focal * class_weights[targets]
    return focal.mean()

def ordinal_penalties(probs, targets):
    pred = probs.argmax(dim=1); dist = (pred - targets).abs().float()
    expc = (probs * torch.arange(CFG_S11.NUM_CLASSES, device=probs.device).float()).sum(dim=1)
    return dist.mean(), ((expc - targets.float())**2).mean()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay; self.shadow = {}; self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.data.clone()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.backup[n])
        self.backup = {}

def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    freq = counts / np.clip(counts.sum(), 1, None)
    inv = 1.0 / np.clip(freq, 1e-6, None); inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)

class TrainerPanda:
    def __init__(self, in_dim, fold, seed, class_weights):
        self.fold = fold; self.seed = seed
        self.model = MILModelPanda(in_dim, CFG_S11.NUM_CLASSES).to(DEVICE_S11)
        self.ema = EMA(self.model, decay=CFG_S11.EMA_DECAY)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=CFG_S11.LR, weight_decay=CFG_S11.WD)
        self.scaler = torch.cuda.amp.GradScaler(enabled=CFG_S11.AMP)
        self.best_kappa = -1.0; self.no_improve = 0
        self.cw = class_weights.to(DEVICE_S11) if class_weights is not None else None
        self.lr_warm = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=0.1, total_iters=CFG_S11.WARMUP_EPOCHS)
        self.lr_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=max(1, CFG_S11.EPOCHS-CFG_S11.WARMUP_EPOCHS), eta_min=1e-6)
        self.ckpt = CFG_S11.OUTPUT / "models" / f"seed{seed}_fold{fold}_best.pth"

    def run_epoch(self, loader, train):
        self.model.train(train); total_loss = 0.0; y_true=[]; y_pred=[]
        for i,b in enumerate(loader,1):
            f05 = b["feat_05"].to(DEVICE_S11); f20 = b["feat_20"].to(DEVICE_S11)
            m05 = b["mask_05"].to(DEVICE_S11); m20 = b["mask_20"].to(DEVICE_S11)
            y = b["label"].to(DEVICE_S11)
            if train:
                with torch.cuda.amp.autocast(enabled=CFG_S11.AMP):
                    out = self.model(f05,m05,f20,m20); logits = out["logits"]
                    loss_ce = focal_ce_loss(logits, y, self.cw)
                    probs = F.softmax(logits.float(), dim=1)
                    dist_mean, mse_mean = ordinal_penalties(probs, y)
                    loss = loss_ce + CFG_S11.ORDINAL_LAM * dist_mean + CFG_S11.EXP_LAM * mse_mean
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), CFG_S11.MAX_GRAD_NORM)
                self.scaler.step(self.opt); self.scaler.update()
                self.ema.update(self.model)
            else:
                self.ema.apply_shadow(self.model)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=CFG_S11.AMP):
                    out = self.model(f05,m05,f20,m20); logits = out["logits"]
                    loss_ce = focal_ce_loss(logits, y, self.cw, label_smooth=0.0)
                    probs = F.softmax(logits.float(), dim=1)
                    dist_mean, mse_mean = ordinal_penalties(probs, y)
                    loss = loss_ce + CFG_S11.ORDINAL_LAM * dist_mean + CFG_S11.EXP_LAM * mse_mean
                self.ema.restore(self.model)
            total_loss += float(loss.item())
            pred = logits.detach().float().softmax(1).argmax(1)
            y_true.extend(y.tolist()); y_pred.extend(pred.tolist())
        avg_loss = total_loss / len(loader)
        kappa = qwk_s11(np.array(y_true), np.array(y_pred))
        acc = accuracy_score(np.array(y_true), np.array(y_pred))
        return avg_loss, kappa, acc

    def fit(self, dl_tr, dl_va):
        for ep in range(1, CFG_S11.EPOCHS+1):
            trL, trK, trA = self.run_epoch(dl_tr, train=True)
            vaL, vaK, vaA = self.run_epoch(dl_va, train=False)
            if ep <= CFG_S11.WARMUP_EPOCHS: self.lr_warm.step()
            else: self.lr_main.step()
            print(f"[E{ep}] Train L={trL:.4f} k={trK:.4f} | Val L={vaL:.4f} k={vaK:.4f}")
            if vaK > self.best_kappa:
                self.best_kappa = vaK; self.no_improve = 0
                torch.save({"model": self.model.state_dict(), "ema": self.ema.shadow, "kappa": vaK}, self.ckpt)
            else:
                self.no_improve += 1
                if self.no_improve >= CFG_S11.PATIENCE: print("  Early stopping"); break

    @torch.no_grad()
    def predict_val(self, loader):
        if self.ckpt.exists():
            ck = torch.load(self.ckpt, map_location=DEVICE_S11)
            self.model.load_state_dict(ck["model"]); self.ema.shadow = ck.get("ema", self.ema.shadow)
        self.model.eval(); self.ema.apply_shadow(self.model)
        preds=[]; probs=[]; labels=[]
        for b in loader:
            f05 = b["feat_05"].to(DEVICE_S11); f20 = b["feat_20"].to(DEVICE_S11)
            m05 = b["mask_05"].to(DEVICE_S11); m20 = b["mask_20"].to(DEVICE_S11)
            out = self.model(f05,m05,f20,m20)
            p = F.softmax(out["logits"].float(), dim=1)
            probs.append(p.cpu().numpy()); preds.append(p.argmax(1).cpu().numpy())
            labels.extend(b["label"].tolist())
        self.ema.restore(self.model)
        return {"preds": np.concatenate(preds,0), "probs": np.concatenate(probs,0), "labels": np.array(labels)}

def main_panda():
    print(f"== PANDA MIL == Device: {DEVICE_S11} | GPU: {GPU_NAME_S11}")
    print(f"   Budgets: {CFG_S11.MAX_TILES_05} @ 0.5 um/px, {CFG_S11.MAX_TILES_20} @ 2.0 um/px")
    idx = PANDADatasetIndex(CFG_S11); splits = idx.splits()
    N = len(idx.df); y_true_all = idx.df["isup_grade"].values
    oof_prob_ens = np.zeros((N, CFG_S11.NUM_CLASSES), dtype=np.float32)

    for seed in CFG_S11.SEEDS:
        set_seed(seed); oof_prob = np.zeros((N, CFG_S11.NUM_CLASSES), dtype=np.float32)
        for f,(tr,va) in enumerate(splits, start=1):
            df_tr = idx.df.iloc[tr].reset_index(drop=True); df_va = idx.df.iloc[va].reset_index(drop=True)
            cw = compute_class_weights(df_tr["isup_grade"].values, CFG_S11.NUM_CLASSES)
            ds_tr = SlideBagDataset(df_tr, train=True); ds_va = SlideBagDataset(df_va, train=False)
            dl_tr = DataLoader(ds_tr, batch_size=CFG_S11.BATCH_SLIDES, shuffle=True, num_workers=CFG_S11.NUM_WORKERS, pin_memory=CFG_S11.PIN_MEMORY, collate_fn=collate_bags)
            dl_va = DataLoader(ds_va, batch_size=CFG_S11.BATCH_SLIDES, shuffle=False, num_workers=CFG_S11.NUM_WORKERS, pin_memory=CFG_S11.PIN_MEMORY, collate_fn=collate_bags)
            trainer = TrainerPanda(CFG_S11.INPUT_DIM, f, seed, cw)
            trainer.fit(dl_tr, dl_va)
            out = trainer.predict_val(dl_va)
            oof_prob[va] = out["probs"]
            kappa = qwk_s11(out["labels"], out["preds"])
            print(f"[Seed {seed} Fold {f}] k={kappa:.4f}")
            del ds_tr, ds_va, dl_tr, dl_va, trainer
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        oof_prob_ens += oof_prob / float(len(CFG_S11.SEEDS))

    oof_pred_ens = oof_prob_ens.argmax(1)
    kappa_oof = qwk_s11(y_true_all, oof_pred_ens)
    acc_oof = accuracy_score(y_true_all, oof_pred_ens)
    print(f"\nOOF ENSEMBLE: k={kappa_oof:.4f}  acc={acc_oof:.4f}")

    pd.DataFrame({
        "image_id": idx.df["image_id"], "true_isup": y_true_all, "pred_isup": oof_pred_ens,
        **{f"prob_{i}": oof_prob_ens[:,i] for i in range(CFG_S11.NUM_CLASSES)}
    }).to_csv(CFG_S11.OUTPUT / "oof_ensemble.csv", index=False)

    with open(CFG_S11.OUTPUT / "summary.json", "w") as f:
        json.dump({"oof_kappa": float(kappa_oof), "oof_accuracy": float(acc_oof),
                    "seeds": CFG_S11.SEEDS, "budgets": {"0p5": CFG_S11.MAX_TILES_05, "2p0": CFG_S11.MAX_TILES_20}}, f, indent=2)
    print(f"[DONE] Script 11 complete. Results: {CFG_S11.OUTPUT}")

if __name__ == "__main__":
    main_panda()


# SCRIPT 12: PANDA OUT-OF-FOLD METRICS
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS_DIR_S12 = WORKSPACE / "results" / "panda_mil"
OOF_CSV_S12 = next((p for p in [
    RESULTS_DIR_S12 / "oof_predictions.csv",
    RESULTS_DIR_S12 / "oof.csv",
    CFG_S11.OUTPUT / "oof_ensemble.csv"
] if p.exists()), None)

if OOF_CSV_S12 is None:
    print("[WARN] No OOF file found; skipping Script 12.")
else:
    df_s12 = pd.read_csv(OOF_CSV_S12)
    assert "true_isup" in df_s12.columns, "true_isup column not found"
    y_true_s12 = df_s12["true_isup"].astype(int).values
    num_classes_s12 = int(max(y_true_s12.max(), 5) + 1)

    prob_cols = sorted([c for c in df_s12.columns if c.startswith("prob_")], key=lambda c: int(c.split("_")[-1]))
    logit_cols = sorted([c for c in df_s12.columns if c.startswith("logit_")], key=lambda c: int(c.split("_")[-1]))

    if prob_cols:
        P_s12 = df_s12[prob_cols].to_numpy(float)
        s = P_s12.sum(axis=1, keepdims=True); s[s==0] = 1.0; P_s12 = P_s12 / s
    elif logit_cols:
        Z = df_s12[logit_cols].to_numpy(float)
        Z = Z - Z.max(axis=1, keepdims=True)
        P_s12 = np.exp(Z); P_s12 /= P_s12.sum(axis=1, keepdims=True)
    else:
        raise RuntimeError("Neither prob_* nor logit_* columns found.")

    def safe_ovr_macro_auroc(y, prob_mat):
        try: return float(roc_auc_score(y, prob_mat, multi_class="ovr", average="macro"))
        except: return float("nan")

    def thresh_scores(y, P, thr):
        y_bin = (y >= thr).astype(int); s_bin = P[:, thr:].sum(axis=1)
        return y_bin, s_bin

    def bin_metrics(y_bin, s_bin):
        return float(roc_auc_score(y_bin, s_bin)), float(average_precision_score(y_bin, s_bin))

    metrics_s12 = {"macro_auroc_ovr": safe_ovr_macro_auroc(y_true_s12, P_s12)}
    metrics_s12["thresholds"] = {}
    for t in [1,2,3,4,5]:
        yb, sb = thresh_scores(y_true_s12, P_s12, t)
        auroc, aupr = bin_metrics(yb, sb)
        metrics_s12["thresholds"][f">={t}"] = {"auroc": auroc, "auprc": aupr, "pos_rate": float(yb.mean())}

    prov_col = "data_provider" if "data_provider" in df_s12.columns else None
    by_prov = []
    if prov_col:
        for prov, dsub in df_s12.groupby(prov_col):
            y_sub = dsub["true_isup"].astype(int).values
            P_sub = dsub[prob_cols].to_numpy(float) if prob_cols else None
            if P_sub is not None:
                s = P_sub.sum(axis=1, keepdims=True); s[s==0]=1.0; P_sub /= s
            row = {"provider": prov, "macro_auroc_ovr": safe_ovr_macro_auroc(y_sub, P_sub), "n": int(len(dsub))}
            for t in [1,2,3,4,5]:
                yb, sb = thresh_scores(y_sub, P_sub, t)
                auroc, aupr = bin_metrics(yb, sb)
                row[f"AUROC_>={t}"] = auroc; row[f"AUPRC_>={t}"] = aupr
            by_prov.append(row)

    out_dir_s12 = OOF_CSV_S12.parent
    (out_dir_s12 / "figures").mkdir(exist_ok=True)
    with open(out_dir_s12 / "metrics_auc.json", "w") as f:
        json.dump(metrics_s12, f, indent=2)
    if by_prov:
        pd.DataFrame(by_prov).to_csv(out_dir_s12 / "metrics_auc_by_provider.csv", index=False)

    print("=== PANDA AUROC/AUPRC (from OOF) ===")
    print(f"Macro AUROC (OvR, {num_classes_s12}-class): {metrics_s12['macro_auroc_ovr']:.4f}")
    for t in [1,2,3,4,5]:
        m = metrics_s12["thresholds"][f'>={t}']
        print(f"  ISUP >={t}:  AUROC {m['auroc']:.4f} | AUPRC {m['auprc']:.4f} | prevalence {m['pos_rate']*100:.1f}%")
    if by_prov:
        for row in by_prov:
            print(f"  {row['provider']:10s} | n={row['n']:4d} | Macro AUROC {row['macro_auroc_ovr']:.4f}")
    print(f"[DONE] Script 12 complete. Saved: {out_dir_s12 / 'metrics_auc.json'}")
