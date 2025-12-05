#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Dataset Manifest & Provenance
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

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

# ---------------------------- Config knobs ----------------------------
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

# ---------------------------- Utilities ----------------------------
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

# ---------------------------- Scan & collect ----------------------------
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

# ---------------------------- DataFrame & save ----------------------------
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

# ---------------------------- Diagnostics (print) ----------------------------
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

# ---------------------------- Diagnostics (plots to file) ----------------------------
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

# ---------------------------- Append compute-passport ----------------------------
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
