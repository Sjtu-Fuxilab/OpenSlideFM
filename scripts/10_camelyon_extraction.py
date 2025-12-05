#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - CAMELYON16/17 Feature Extraction
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# Script 7 CAMELYON16/17y Feature Extraction 

import os, sys, re, json, time, math, random, shutil, gc, hashlib, subprocess, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# ----------------------- Paths -----------------------
WORKSPACE = Path(os.environ.get("WORKSPACE", "./workspace")))
RAW_CAM16 = WORKSPACE / r"Raw Data" / "CAMELYON16"
RAW_CAM17 = WORKSPACE / r"Raw Data" / "CAMELYON17"

MANIFESTS = WORKSPACE / "manifests"
QC_DIR    = WORKSPACE / "qc"
FEAT05    = WORKSPACE / "features" / "scale0p5"
FEAT20    = WORKSPACE / "features" / "scale2p0"
LOG_DIR   = WORKSPACE / "logs"
COMP_DIR  = WORKSPACE / "compute"
for p in [MANIFESTS, QC_DIR, FEAT05, FEAT20, LOG_DIR, COMP_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ----------------------- Deps -----------------------
def _ensure(pkgs):
    miss=[]
    for name, spec in pkgs:
        try: __import__(name)
        except Exception: miss.append(spec)
    if miss:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *miss])

_ensure([
    ("pandas","pandas>=2.0"),
    ("numpy","numpy>=1.24"),
    ("openslide","openslide-python>=1.2"),
    ("PIL","Pillow>=10.0"),
    ("torch","torch>=2.1"),
    ("torchvision","torchvision>=0.16"),
    ("pyarrow","pyarrow>=14"),
])

import pandas as pd, numpy as np
import openslide
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
from torchvision.transforms.functional import to_tensor as _to_tensor

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16 if DEVICE=="cuda" else torch.bfloat16
torch.set_float32_matmul_precision("high")

# ----------------------- Config (same as TCGA) -----------------------
CFG = {
  "scales_um_per_px": [0.5, 2.0],
  "tile_px": 256,
  "tile_overlap": 32,
  "token_budget": {0.5: 1200, 2.0: 400},
  "input_size": 224,
  "batch_size": 2048,
  "num_workers": 0,
  "seed": 1337,
  "print_every_slides": 50,
  # WSI-only filters
  "bad_name_substrings": ["_tissue", "mask", "prob", "heatmap", "anno", "overlay", "xml", "thumb", "down", "level"],
  "min_side_px": 5000  # require at least one side >= 5000 px to consider WSI-like
}

random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"]); torch.cuda.manual_seed_all(CFG["seed"])

# ----------------------- Helpers -----------------------
def list_tifs(root: Path):
    exts = {".svs",".tif",".tiff",".ndpi",".mrxs",".scn",".svslide",".bif",".vms",".vmu"}
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)

def looks_like_aux(name_stem: str) -> bool:
    low = name_stem.lower()
    return any(b in low for b in CFG["bad_name_substrings"])

def mpp_xy(slide: openslide.OpenSlide):
    x = slide.properties.get("openslide.mpp-x")
    y = slide.properties.get("openslide.mpp-y")
    try:
        return (float(x), float(y))
    except Exception:
        # heuristic fallback; CAMELYON often ~0.25 µm/px at base
        return (0.25, 0.25)

def level_for_um(slide: openslide.OpenSlide, target_um):
    base_x, _ = mpp_xy(slide)
    best = 0; best_diff = 1e9
    for lvl in range(slide.level_count):
        mpp = base_x * slide.level_downsamples[lvl]
        diff = abs(mpp - target_um)
        if diff < best_diff:
            best_diff, best = diff, lvl
    return best

def pil_to_tensor(img_rgb: Image.Image, size=224):
    if img_rgb.size != (size, size):
        img_rgb = img_rgb.resize((size,size), Image.BILINEAR)
    t = _to_tensor(img_rgb)
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (t - mean) / std

def grid_tiles(w, h, size, overlap):
    stride = size - overlap
    xs = list(range(0, max(1, w-size+1), stride))
    ys = list(range(0, max(1, h-size+1), stride))
    if len(xs)==0: xs=[0]
    if len(ys)==0: ys=[0]
    return [(x,y) for y in ys for x in xs]

def choose_tiles(slide: openslide.OpenSlide, lvl: int, size: int, overlap: int, budget: int):
    w, h = slide.level_dimensions[lvl]
    coords = grid_tiles(w, h, size, overlap)
    if len(coords) == 0:
        coords = [(0,0)]
    # sample uniformly up to budget (with replacement if needed)
    if len(coords) >= budget:
        idx = np.random.choice(len(coords), size=budget, replace=False)
    else:
        idx = np.random.choice(len(coords), size=budget, replace=True)
    return [coords[i] for i in idx]

# ----------------------- Model (frozen backbone → 768) -----------------------
class ResNet50Proj768(nn.Module):
    def __init__(self):
        super().__init__()
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # -> [B,2048,1,1]
        self.proj = nn.Linear(2048, 768)
    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = self.proj(x)
        return x

def build_model():
    model = ResNet50Proj768().to(DEVICE)
    model.eval()
    return model

# ----------------------- WSI-only manifest build -----------------------
def build_manifest_cam(tag: str, root: Path) -> pd.DataFrame:
    files = list_tifs(root)
    rows, skipped = [], []
    for p in files:
        stem = p.stem
        # 1) name filter
        if looks_like_aux(stem):
            skipped.append({"path":str(p),"reason":"aux_name"})
            continue
        # 2) probe + size filter
        try:
            with openslide.OpenSlide(str(p)) as s:
                w, h = s.dimensions
                if max(w,h) < CFG["min_side_px"]:
                    skipped.append({"path":str(p),"reason":f"too_small_{w}x{h}"})
                    continue
                vendor = s.properties.get("openslide.vendor","unknown")
                mppx, mppy = mpp_xy(s)
        except Exception as e:
            skipped.append({"path":str(p),"reason":f"open_fail:{type(e).__name__}"})
            continue
        rows.append({
           "dataset": tag,
           "slide_id": stem,
           "path": str(p),
           "w": int(w), "h": int(h),
           "mpp_x": float(mppx), "mpp_y": float(mppy),
           "vendor": vendor
        })
    df = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)
    sk = pd.DataFrame(skipped)
    df.to_csv(MANIFESTS / f"manifest_{tag.lower()}.csv", index=False)
    df.to_parquet(MANIFESTS / f"manifest_{tag.lower()}.parquet", index=False)
    if len(sk)>0:
        sk.to_csv(MANIFESTS / f"manifest_{tag.lower()}_skipped.csv", index=False)
    print(f"[OK] Manifest {tag}: {len(df)} slides (skipped {len(sk)}) → {MANIFESTS/f'manifest_{tag.lower()}.csv'}")
    return df

# ----------------------- Light QC -----------------------
def light_qc(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    kept, excl = [], []
    for _,row in df.iterrows():
        try:
            with openslide.OpenSlide(str(row["path"])) as s:
                lvl = s.get_best_level_for_downsample(64)
                img = s.read_region((0,0), lvl, s.level_dimensions[lvl]).convert("RGB")
                a = np.asarray(img)
                gray = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)
                tissue_frac = float((gray < 240).mean())
                white_frac  = float((a.mean(axis=2) > 240).mean())
                ok = (tissue_frac >= 0.05) and (white_frac <= 0.99)
        except Exception as e:
            ok = False
        (kept if ok else excl).append(row)
    kept_df = pd.DataFrame(kept).reset_index(drop=True)
    excl_df = pd.DataFrame(excl).reset_index(drop=True)
    kept_df.to_csv(QC_DIR / f"qc_pass_{tag.lower()}.csv", index=False)
    excl_df.to_csv(QC_DIR / f"qc_fail_{tag.lower()}.csv", index=False)
    print(f"[QC] {tag}: kept={len(kept_df)}  excluded={len(excl_df)}")
    return kept_df

# ----------------------- Extraction -----------------------
@torch.no_grad()
def extract_for_slide(model, slide_path: Path, budgets: dict, tile_sz=256, overlap=32, input_sz=224):
    out = {}
    with openslide.OpenSlide(str(slide_path)) as s:
        for scale in CFG["scales_um_per_px"]:
            lvl = level_for_um(s, scale)
            coords = choose_tiles(s, lvl, tile_sz, overlap, budgets[scale])
            batch=[]
            for (x,y) in coords:
                try:
                    img = s.read_region((x,y), lvl, (tile_sz, tile_sz)).convert("RGB")
                except Exception as e:
                    # if a coordinate is bad due to pyramid quirk, fallback to (0,0)
                    img = s.read_region((0,0), lvl, (tile_sz, tile_sz)).convert("RGB")
                batch.append(pil_to_tensor(img, size=input_sz))
            X = torch.stack(batch, dim=0).to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            outs=[]
            bs = CFG["batch_size"]
            for i in range(0, X.shape[0], bs):
                chunk = X[i:i+bs]
                with torch.amp.autocast(device_type=("cuda" if DEVICE=="cuda" else "cpu"), dtype=AMP_DTYPE, enabled=True):
                    z = model(chunk)
                outs.append(z.detach().cpu())
            out[scale] = torch.cat(outs, dim=0).numpy().astype(np.float32)
            del X, outs; gc.collect()
            if DEVICE=="cuda": torch.cuda.empty_cache()
    return out

def save_feats(slide_id: str, arr05: np.ndarray, arr20: np.ndarray):
    np.save(FEAT05 / f"{slide_id}.npy", arr05)
    np.save(FEAT20 / f"{slide_id}.npy", arr20)

# ----------------------- Main -----------------------
def main():
    # Passport update
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    passport = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "device": DEVICE,
        "torch": torch.__version__,
        "gpu": (torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu"),
        "workspace": str(WORKSPACE),
        "filters": {"bad_name_substrings": CFG["bad_name_substrings"], "min_side_px": CFG["min_side_px"]}
    }
    (COMP_DIR / "compute_passport.json").write_text(json.dumps(passport, indent=2), encoding="utf-8")

    # 1) Build WSI-only manifests (overwrites previous)
    df16 = build_manifest_cam("CAMELYON16", RAW_CAM16)
    df17 = build_manifest_cam("CAMELYON17", RAW_CAM17)

    # 2) Light QC
    df16 = light_qc(df16, "CAMELYON16")
    df17 = light_qc(df17, "CAMELYON17")

    # 3) Assemble TODO set (skip already done at both scales)
    todo = pd.concat([df16, df17], ignore_index=True)
    keep=[]
    for _,row in todo.iterrows():
        sid = row["slide_id"]
        if (FEAT05 / f"{sid}.npy").exists() and (FEAT20 / f"{sid}.npy").exists():
            continue
        keep.append(row)
    todo = pd.DataFrame(keep) if keep else pd.DataFrame(columns=todo.columns)
    total = len(todo)
    print(f"[RUN] Pending slides (2 scales missing): {total}")
    if total == 0:
        print("[DONE] Nothing to do.")
        return

    # 4) Model
    model = build_model()

    # 5) Loop
    t0=time.time(); last=t0; done=0
    for _,row in todo.iterrows():
        sid = row["slide_id"]; sp = Path(row["path"])
        try:
            d = extract_for_slide(model, sp, CFG["token_budget"],
                                  tile_sz=CFG["tile_px"], overlap=CFG["tile_overlap"], input_sz=CFG["input_size"])
            save_feats(sid, d[0.5], d[2.0])
            done += 1
        except Exception as e:
            # log the skip
            with open(MANIFESTS / "camelyon_errors.log", "a", encoding="utf-8") as fh:
                fh.write(f"{sid}\t{sp}\t{type(e).__name__}:{e}\n")
        # progress
        now=time.time()
        if (done % CFG["print_every_slides"]==0) or (done==total) or (now-last>60):
            dt = now - t0
            eps = (done*2) / max(1e-6, dt)  # entries/sec (two scales per slide)
            print(f"[{done:5d}/{total}] slide={sid}  elapsed={dt/60:.1f} min  entries/s={eps:.2f}  VRAM~{(torch.cuda.max_memory_allocated()/(1024**3) if DEVICE=='cuda' else 0):.2f} GB")
            last=now

    # 6) Summary
    s05 = len(list(FEAT05.glob("*.npy"))); s20 = len(list(FEAT20.glob("*.npy")))
    common = len(set([p.stem for p in FEAT05.glob("*.npy")]) & set([p.stem for p in FEAT20.glob("*.npy")]))
    diag = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "manifest_counts": {"CAMELYON16_qc": int(len(df16)), "CAMELYON17_qc": int(len(df17))},
        "features_written_scale0p5": s05,
        "features_written_scale2p0": s20,
        "features_2scale_intersection": common
    }
    (WORKSPACE / "diagnostics").mkdir(exist_ok=True)
    (WORKSPACE / "diagnostics" / "script7_camelyon_summary.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    (WORKSPACE / "diagnostics" / "script7_camelyon_summary.txt").write_text(
        "\n".join([f"{k}: {v}" for k,v in diag.items()]), encoding="utf-8")

    print("\n[OK] Script 7 complete.")
    print(f" - @0.5µm: {s05} files")
    print(f" - @2.0µm: {s20} files")
    print(f" - 2-scale intersection: {common}")
    print(f" - Summary: {WORKSPACE / 'diagnostics' / 'script7_camelyon_summary.txt'}")

if __name__ == "__main__":
    main()
