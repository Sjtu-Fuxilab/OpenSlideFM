#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - CAMELYON16 Slide-level CV
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# CAMELYON16 — Macenko stain-normalized re-extraction + slide-level CV (ResNet50 penultimate, multi-scale, robust)

import os, sys, json, math, time, random, subprocess
from pathlib import Path
from datetime import datetime

# ---------- Config ----------
WORKSPACE = Path(r"D:/个人文件夹/Sanwal/OpenSlide")
RAW_CAM16 = WORKSPACE / "Raw Data" / "CAMELYON16"
MANIFEST1 = WORKSPACE / "manifests" / "manifest_camelyon16_originals.csv"
MANIFEST2 = WORKSPACE / "manifests" / "manifest_cam16_CLEAN.csv"
OUT_FEAT  = WORKSPACE / "features" / "cam16_norm"
OUT_RES   = WORKSPACE / "results"  / "cam16_slide_norm"
DIAG_DIR  = WORKSPACE / "diagnostics"

DO_SCALE_20 = True     # 2.0 µm (context)
DO_SCALE_05 = True     # 0.5 µm (detail) — recommended
TILE_PX     = 256
STRIDE_PX   = 256      # increase to 384/512 to speed up
BATCH_SIZE  = 128
MAX_TILES_PER_SLIDE = 18000
SEED = 1337
random.seed(SEED)

# ---------- Deps ----------
def _need(mod, pipname=None):
    try: __import__(mod)
    except Exception: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pipname or mod])

for m in ["pandas","numpy","openslide","torch","torchvision","scikit-learn","tqdm"]:
    _need(m)

import numpy as np
import pandas as pd
import openslide
from openslide import OpenSlideError
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

# ---------- IO ----------
OUT_FEAT.mkdir(parents=True, exist_ok=True)
OUT_RES.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers: robust path resolution ----------
_ALLOWED_EXT = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".bif")

def slide_key(name: str) -> str:
    s = str(Path(name).stem).lower()
    if s.startswith("tumor_") or s.startswith("normal_"): return s
    if "tumor" in s:  return "tumor_" + "".join([c for c in s if c.isdigit()])[:3].zfill(3)
    if "normal" in s: return "normal_" + "".join([c for c in s if c.isdigit()])[:3].zfill(3)
    return s

def _is_mask_or_meta(p: Path) -> bool:
    n = p.name.lower()
    return ("mask" in n) or n.endswith(".xml") or n.endswith(".json")

def _can_open_with_openslide(p: Path) -> bool:
    try:
        s = openslide.OpenSlide(str(p))
        s.close()
        return True
    except Exception:
        return False

def resolve_wsi_path(root: Path, sid: str) -> Path | None:
    """
    Prefer true WSI over masks/meta. Validate by opening with OpenSlide.
    Search order: exact filename matches by allowed exts, then rglob.
    """
    sid_l = sid.lower()
    # 1) direct candidates (fast)
    for ext in _ALLOWED_EXT:
        p = root / f"{sid}{ext}"
        if p.exists() and not _is_mask_or_meta(p) and _can_open_with_openslide(p):
            return p
    # 2) recursive search
    cands = []
    for p in root.rglob(f"{sid}*"):
        if not p.is_file(): continue
        if _is_mask_or_meta(p): continue
        if p.suffix.lower() not in _ALLOWED_EXT: continue
        cands.append(p)
    # try to validate in a stable order (by suffix preference, then name length)
    pref = {".svs":0, ".tif":1, ".tiff":2, ".ndpi":3, ".mrxs":4, ".bif":5}
    for p in sorted(cands, key=lambda q: (pref.get(q.suffix.lower(), 9), len(q.name))):
        if _can_open_with_openslide(p):
            return p
    return None

def load_manifest() -> pd.DataFrame:
    if MANIFEST1.exists(): mf = MANIFEST1
    elif MANIFEST2.exists(): mf = MANIFEST2
    else: raise FileNotFoundError("No CAM16 manifest found.")
    df = pd.read_csv(mf)
    if "slide_id" not in df.columns:
        df["slide_id"] = df.get("sid", df.get("name", df.get("wsi", df.index.astype(str)))).astype(str)
    # normalize id format
    df["slide_id"] = df["slide_id"].map(slide_key)
    if "kind" not in df.columns:
        df["kind"] = df["slide_id"].map(lambda s: "tumor" if s.startswith("tumor_") else ("normal" if s.startswith("normal_") else "unknown"))
    # enforce CAM16 ids and resolve robust paths
    df = df[df["slide_id"].str.match(r"^(tumor|normal)_\d+$", na=False)].reset_index(drop=True)
    paths = []
    for sid in df["slide_id"]:
        p = resolve_wsi_path(RAW_CAM16, sid)
        paths.append(str(p) if p is not None else None)
    df["path"] = paths
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    return df

# ---------- MPP / level utils ----------
def get_base_mpp(slide: openslide.OpenSlide) -> float:
    props = slide.properties
    for k in ("aperio.MPP","openslide.mpp-x","openslide.mpp-y"):
        if k in props:
            try:
                v = float(props[k])
                if v > 0: return v
            except: pass
    return 0.243  # typical CAM16

def best_level_for_um(slide, target_um):
    base = get_base_mpp(slide)  # µm/px at level 0
    desired_ds = max(1.0, target_um / max(1e-6, base))
    lvl = slide.get_best_level_for_downsample(desired_ds)
    lvl = int(max(0, min(lvl, slide.level_count - 1)))
    ds_eff = float(slide.level_downsamples[lvl])
    return lvl, float(base * ds_eff), ds_eff

def tissue_mask_fast(img_rgb: np.ndarray) -> np.ndarray:
    I = img_rgb.astype(np.float32)
    v = I.mean(axis=2)
    sat = (I.max(axis=2) - I.min(axis=2))
    return (v < 235) | (sat > 10)

# ---------- Macenko ----------
def rgb_to_od(I: np.ndarray) -> np.ndarray:
    I = I.astype(np.float32) + 1.0
    return -np.log(I / 255.0)

def od_to_rgb(OD: np.ndarray) -> np.ndarray:
    return (np.exp(-OD) * 255.0).clip(0, 255).astype(np.uint8)

def _norm_cols(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)

def macenko_estimate(I_rgb: np.ndarray, alpha: float = 0.1):
    OD = rgb_to_od(I_rgb).reshape(-1, 3)
    tissue = (OD > alpha).any(axis=1)
    OD_t = OD[tissue]
    if OD_t.shape[0] < 500:
        return None, None
    U, S, Vt = np.linalg.svd(OD_t, full_matrices=False)
    v = Vt[:2, :].T
    proj = OD_t @ v
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    vmin = np.percentile(phi, 1)
    vmax = np.percentile(phi, 99)
    vH = (np.array([np.cos(vmin), np.sin(vmin)]) @ v.T)
    vE = (np.array([np.cos(vmax), np.sin(vmax)]) @ v.T)
    HE = _norm_cols(np.stack([vH, vE], axis=1))       # (3x2)
    C_sub = np.linalg.lstsq(HE, OD_t.T, rcond=None)[0]  # (2xN_tissue)
    C_sub = np.clip(C_sub, 0, np.percentile(C_sub, 99, axis=1, keepdims=True))
    return HE, C_sub

def estimate_reference_from_slides(manifest_df: pd.DataFrame, max_slides: int = 5):
    rng = np.random.default_rng(SEED)
    tumor_df = manifest_df[manifest_df["kind"] == "tumor"]
    if len(tumor_df) == 0:
        HE_ref = _norm_cols(np.array([[0.65, 0.07, 0.27],
                                      [0.07, 0.99, 0.11]]).T)
        C99_ref = np.array([1.0, 1.0], dtype=np.float32)
        return HE_ref, C99_ref
    cand = tumor_df.sample(n=min(max_slides, len(tumor_df)), random_state=SEED)
    HEs, C99s = [], []
    for _, r in cand.iterrows():
        try:
            s = openslide.OpenSlide(r["path"])
            lvl, _, ds = best_level_for_um(s, 2.0)
            w, h = s.level_dimensions[lvl]
            for _ in range(3):
                if w < 512 or h < 512: break
                x = int(rng.integers(0, max(1, w - 512)))
                y = int(rng.integers(0, max(1, h - 512)))
                img = np.asarray(s.read_region((int(round(x*ds)), int(round(y*ds))), lvl, (512, 512)).convert("RGB"))
                if tissue_mask_fast(img).mean() < 0.05: continue
                HE, C_sub = macenko_estimate(img)
                if HE is None: continue
                HEs.append(HE); C99s.append(np.percentile(C_sub, 99, axis=1))
        except Exception:
            pass
        finally:
            try: s.close()
            except: pass
    if not HEs:
        HE_ref = _norm_cols(np.array([[0.65, 0.07, 0.27],
                                      [0.07, 0.99, 0.11]]).T)
        C99_ref = np.array([1.0, 1.0], dtype=np.float32)
    else:
        HE_stack = np.stack(HEs, axis=2)
        HE_ref = _norm_cols(np.mean(HE_stack, axis=2))
        C99_ref = np.median(np.stack(C99s, axis=1), axis=1).astype(np.float32)
    return HE_ref, C99_ref

def macenko_apply_tile(tile_rgb: np.ndarray, HE_ref: np.ndarray, C99_ref: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    HE_src, C_sub = macenko_estimate(tile_rgb, alpha=alpha)
    if HE_src is None:
        return tile_rgb
    OD_full = rgb_to_od(tile_rgb).reshape(-1, 3)              # (N,3)
    C_full = np.linalg.lstsq(HE_src, OD_full.T, rcond=None)[0]# (2,N)
    c99_src = np.percentile(C_sub, 99, axis=1)                # (2,)
    scale = C99_ref / (c99_src + 1e-8)                        # (2,)
    C_full_scaled = (C_full.T * scale.reshape(1, 2)).T
    OD_norm = (HE_ref @ C_full_scaled).T                      # (N,3)
    return od_to_rgb(OD_norm).reshape(tile_rgb.shape)

# ---------- Embedding ----------
def load_backbone():
    import torchvision.models as tvm
    import torch.nn as nn
    base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(base.children())[:-1])  # [B,2048,1,1]
    backbone.eval()
    return backbone

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def to_tensor_bchw(img_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_uint8).permute(2,0,1).unsqueeze(0).float()/255.0
    return (t - IMAGENET_MEAN.to(t.device)) / IMAGENET_STD.to(t.device)

@torch.no_grad()
def embed_tiles(tiles_rgb, model, device="cuda"):
    if not tiles_rgb: return np.zeros((0,2048), dtype=np.float32)
    feats = []
    use_amp = (device == "cuda")
    for i in range(0, len(tiles_rgb), BATCH_SIZE):
        x = torch.cat([to_tensor_bchw(im) for im in tiles_rgb[i:i+BATCH_SIZE]], dim=0).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=("cuda" if use_amp else "cpu"), dtype=torch.float16, enabled=use_amp):
            z = model(x)               # [B,2048,1,1]
        z = z.view(z.size(0), -1).detach().float().cpu().numpy()
        feats.append(z)
    return np.concatenate(feats, axis=0)

# ---------- Reference cache ----------
REF_JSON = OUT_FEAT / "macenko_reference.json"
def get_or_make_reference(df: pd.DataFrame):
    if REF_JSON.exists():
        js = json.loads(REF_JSON.read_text(encoding="utf-8"))
        return np.array(js["HE_ref"], dtype=np.float32), np.array(js["C99_ref"], dtype=np.float32)
    HE_ref, C99_ref = estimate_reference_from_slides(df)
    REF_JSON.write_text(json.dumps({"HE_ref": HE_ref.tolist(),
                                    "C99_ref": C99_ref.tolist(),
                                    "time": datetime.now().isoformat(timespec="seconds")},
                                   indent=2), encoding="utf-8")
    return HE_ref, C99_ref

# ---------- Tiling / extraction ----------
def tile_coords_iter(w_lvl: int, h_lvl: int, ds_from0: float, tile_px=TILE_PX, stride_px=STRIDE_PX, limit=None):
    n = 0
    for y in range(0, h_lvl, stride_px):
        if y + tile_px > h_lvl: break
        for x in range(0, w_lvl, stride_px):
            if x + tile_px > w_lvl: break
            x0 = int(round(x * ds_from0))
            y0 = int(round(y * ds_from0))
            yield x, y, x0, y0
            n += 1
            if limit and n >= limit: return

def extract_one(slide_path: str, out_dir: Path, target_um: float, model, HE_ref, C99_ref, device="cuda"):
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = slide_key(slide_path)
    feat_path = out_dir / f"{sid}.npy"
    meta_path = out_dir / f"{sid}_meta.csv"
    if feat_path.exists() and meta_path.exists():
        return "exist", 0, 0.0

    tiles = []; mmxy = []
    s = None
    try:
        try:
            s = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            # Unsupported/corrupt file → skip gracefully
            return "bad_format", 0, 0.0

        lvl, mpp_lvl, ds = best_level_for_um(s, target_um)
        w_lvl, h_lvl = s.level_dimensions[lvl]
        kept = 0; base_mpp = get_base_mpp(s)

        for (xl, yl, x0, y0) in tile_coords_iter(w_lvl, h_lvl, ds, TILE_PX, STRIDE_PX, limit=MAX_TILES_PER_SLIDE*2):
            try:
                im = np.asarray(s.read_region((x0, y0), lvl, (TILE_PX, TILE_PX)).convert("RGB"))
            except Exception:
                continue
            if tissue_mask_fast(im).mean() < 0.15:
                continue
            imn = macenko_apply_tile(im, HE_ref, C99_ref)
            tiles.append(imn)
            cx0 = x0 + TILE_PX * ds / 2.0
            cy0 = y0 + TILE_PX * ds / 2.0
            mmx = float(cx0 * base_mpp / 1000.0)
            mmy = float(cy0 * base_mpp / 1000.0)
            mmxy.append((mmx, mmy))
            kept += 1
            if kept >= MAX_TILES_PER_SLIDE: break

        if not tiles:
            np.save(feat_path, np.zeros((0, 2048), dtype=np.float32))
            pd.DataFrame({"mm_x": [], "mm_y": [], "scale_um_per_px": []}).to_csv(meta_path, index=False)
            return "empty", 0, float(mpp_lvl)

        feats = embed_tiles(tiles, model, device=device)  # [T,2048]
        np.save(feat_path, feats.astype(np.float32))
        meta = pd.DataFrame(mmxy, columns=["mm_x", "mm_y"])
        meta["scale_um_per_px"] = mpp_lvl
        meta.to_csv(meta_path, index=False)
        return "ok", feats.shape[0], float(mpp_lvl)

    finally:
        try:
            if s is not None: s.close()
        except Exception:
            pass

# ---------- Slide-level CV ----------
def build_slide_vectors(df: pd.DataFrame, root_feat: Path):
    X, y, sids = [], [], []
    for _, r in df.iterrows():
        sid = r["slide_id"]; sids.append(sid)
        y.append(1 if r["kind"] == "tumor" else 0)
        parts = []
        f2 = root_feat / "scale2p0" / f"{sid}.npy"
        if f2.exists():
            a = np.load(f2)
            if a.size > 0:
                norms = np.linalg.norm(a, axis=1)
                K = min(64, a.shape[0])
                topk = a[np.argpartition(-norms, K-1)[:K]].mean(axis=0)
                parts.append(a.mean(axis=0)); parts.append(topk)
        f5 = root_feat / "scale0p5" / f"{sid}.npy"
        if DO_SCALE_05 and f5.exists():
            b = np.load(f5)
            if b.size > 0:
                norms = np.linalg.norm(b, axis=1)
                K = min(64, b.shape[0])
                topk = b[np.argpartition(-norms, K-1)[:K]].mean(axis=0)
                parts.append(b.mean(axis=0)); parts.append(topk)
        if not parts:
            parts = [np.zeros((2048,), dtype=np.float32)]
        X.append(np.concatenate(parts, axis=0).astype(np.float32))
    X = np.vstack(X)
    return X, np.array(y, dtype=np.int64), np.array(sids, dtype=object)

def run_slide_cv(df: pd.DataFrame, root_feat: Path, folds: int = 5):
    X, y, sids = build_slide_vectors(df, root_feat)
    ncomp = min(256, X.shape[1], max(1, X.shape[0] - 1))
    pipe = Pipeline([
        ("sc",  StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=ncomp, svd_solver="full", random_state=SEED)),
        ("clf", LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000, solver="lbfgs"))
    ])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros_like(y, dtype=np.float32)
    rows = []
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        Xt, Xv, yt, yv = X[tr], X[va], y[tr], y[va]
        pipe.fit(Xt, yt)
        pv = pipe.predict_proba(Xv)[:, 1]
        oof[va] = pv
        auc = roc_auc_score(yv, pv)
        ap  = average_precision_score(yv, pv)
        pred = (pv >= 0.5).astype(int)
        rows.append({"fold": i, "AUC": float(auc), "AP": float(ap),
                     "ACC": float(accuracy_score(yv, pred)),
                     "F1": float(f1_score(yv, pred, zero_division=0))})
        print(f"[FOLD {i}] AUC={auc:.4f} AP={ap:.4f} ACC={rows[-1]['ACC']:.3f} F1={rows[-1]['F1']:.3f}")

    auc_oof = roc_auc_score(y, oof)
    ap_oof  = average_precision_score(y, oof)
    acc_oof = accuracy_score(y, (oof >= 0.5).astype(int))
    f1_oof  = f1_score(y, (oof >= 0.5).astype(int), zero_division=0)

    summ = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "slides": int(len(y)),
        "pos": int(y.sum()),
        "neg": int((1 - y).sum()),
        "dim_in": int(X.shape[1]),
        "auc_roc_oof": float(auc_oof),
        "auc_pr_oof": float(ap_oof),
        "acc_oof": float(acc_oof),
        "f1_oof": float(f1_oof),
        "folds": int(folds),
        "scales_used": "2.0" + ("+0.5" if DO_SCALE_05 else "")
    }
    (OUT_RES / "slide_cv_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print("\n== Slide-level CV — Macenko normalized ==")
    print(json.dumps(summ, indent=2))
    return summ

# ---------- Main ----------
def main():
    print("== CAMELYON16 — Stain-normalized re-extraction + slide-level CV ==")
    print(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                      "workspace": str(WORKSPACE)}, indent=2))

    df = load_manifest()
    df = df[df["kind"].isin(["tumor","normal"])].reset_index(drop=True)
    print(f"[DATA] slides={len(df)}  tumor={(df['kind']=='tumor').sum()}  normal={(df['kind']=='normal').sum()}")

    HE_ref, C99_ref = get_or_make_reference(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_backbone().to(device).eval()

    scales = []
    if DO_SCALE_20: scales.append(2.0)
    if DO_SCALE_05: scales.append(0.5)

    t0 = time.time()
    for scale_um in scales:
        out_dir = OUT_FEAT / ("scale2p0" if abs(scale_um - 2.0) < 1e-6 else "scale0p5")
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = exist = fail = bad = tiles_total = 0
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"EXTRACT {scale_um}µm"):
            sid = r["slide_id"]
            feat_path = out_dir / f"{sid}.npy"
            meta_path = out_dir / f"{sid}_meta.csv"
            if feat_path.exists() and meta_path.exists():
                exist += 1
                continue
            st, ntiles, mpp = extract_one(r["path"], out_dir, scale_um, model, HE_ref, C99_ref, device=device)
            if st == "ok":
                ok += 1; tiles_total += ntiles
            elif st == "exist":
                exist += 1
            elif st == "bad_format":
                bad += 1
            else:
                fail += 1
        print(f"[SCALE {scale_um}µm] ok={ok} exist={exist} bad_format={bad} fail={fail} tiles_total≈{tiles_total}")
    print(f"[TIME] extraction wall={(time.time()-t0)/60.0:.1f} min")

    summary = run_slide_cv(df, OUT_FEAT, folds=5)
    (DIAG_DIR / "cam16_norm_reextract_summary.json").write_text(json.dumps({
        "time": datetime.now().isoformat(timespec="seconds"),
        "slides": int(len(df)),
        "scales": scales,
        "cv": summary
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
