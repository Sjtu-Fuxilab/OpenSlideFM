#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Environment, Paths, and Compute-Passport
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# OP_FM — Script 1: Environment, Paths, and Compute-Passport

import os, sys, json, time, math, platform, shutil, socket, datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ------------------------------- USER PATHS -------------------------------
# READ-ONLY: your TCGA WSI root (no writes will ever be performed here)
WSI_ROOT = Path(r"D:\个人文件夹\Sanwal\DL_V2\Histo slides 20k")

# WORKSPACE: all pipeline outputs go here (and only here)
WORKSPACE = Path(r"D:\个人文件夹\Sanwal\OpenSlide")

# ----------------------------- SUBFOLDER LAYOUT ---------------------------
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

# ------------------------------- IMPORTS ----------------------------------
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

# ------------------------------- HELPERS ----------------------------------
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

# ----------------------------- ENV SUMMARY --------------------------------
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

# ------------------------------ RUNTIME START -----------------------------
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

# ----------------------- SANITY: FIND & PROBE ONE WSI ---------------------
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

# ----------------------- LOG SUMMARY TO WORKSPACE -------------------------
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

# ----------------------- Diagnostics Checklist ------------------------
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
