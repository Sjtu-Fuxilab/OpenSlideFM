#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Feature Extraction
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# Script 5 — OpenSlide extractor

import os, sys, json, time, math, random, shutil, subprocess, platform, gc
from pathlib import Path
from datetime import datetime
from time import perf_counter

# ---------- Paths (strict) ----------
WORKSPACE = Path(r"D:\个人文件夹\Sanwal\OpenSlide").resolve()
WSI_ROOT  = Path(r"D:\个人文件夹\Sanwal\DL_V2\Histo slides 20k").resolve()
SUBDIRS = {
    "features": WORKSPACE / "features",
    "tiles":    WORKSPACE / "tiles",
    "logs":     WORKSPACE / "logs",
    "figures":  WORKSPACE / "figures",
}
for p in SUBDIRS.values(): p.mkdir(parents=True, exist_ok=True)

TSUM = SUBDIRS["tiles"] / "tiling_summary_tcga.parquet"
assert TSUM.exists(), f"Missing tiling summary: {TSUM}"

# ---------- Quiet-install deps (no admin) ----------
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

# ---------- Config ----------
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

# ---------- Transforms ----------
IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
_to_tensor = T.ToTensor()
_resize    = T.Resize((MODEL_IN, MODEL_IN), interpolation=T.InterpolationMode.BILINEAR)
_normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
def to_model_tensor(img: Image.Image) -> torch.Tensor:
    if img.size != (MODEL_IN, MODEL_IN):
        img = _resize(img)
    t = _to_tensor(img); t = _normalize(t)
    return t

# ---------- Model (ConvNeXt-Tiny → 768D) ----------
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

# ---------- Tiling summary & manifest helpers ----------
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

# ---------- OpenSlide reader (level coords → level-0 coords) ----------
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

# ---------- Batching & forward ----------
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

# ---------- Output paths ----------
OUT05 = SUBDIRS["features"] / "scale0p5"; OUT20 = SUBDIRS["features"] / "scale2p0"
OUT05.mkdir(parents=True, exist_ok=True); OUT20.mkdir(parents=True, exist_ok=True)
def out_paths(slide_id:str, scale:float, ext="npy"):
    d = OUT05 if math.isclose(scale,0.5,abs_tol=1e-6) else OUT20
    return d / f"{slide_id}.{ext}", d / f"{slide_id}_meta.parquet"

# ---------- Env print ----------
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

# ---------- Build pending groups (both scales per slide) ----------
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

# ---------- Self-test (60 s) ----------
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

# ---------- Full run (only if self-test passed) ----------
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
