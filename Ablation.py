import os, sys, re, json, math, time, gc, random, warnings, traceback
from pathlib import Path
from datetime import datetime
from time import perf_counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix
)

# Paths
WORKSPACE   = Path(r"D:\Data")
WSI_ROOT    = Path(r"D:\Data")
FEAT_05     = WORKSPACE / "features" / "scale0p5"
FEAT_20     = WORKSPACE / "features" / "scale2p0"
MANIFESTS   = WORKSPACE / "manifests"
MODELS_DIR  = WORKSPACE / "models"
RAW_CAM16   = WORKSPACE / "Raw Data" / "CAMELYON16"
RAW_CAM17   = WORKSPACE / "Raw Data" / "CAMELYON17"
CAM16_FEAT  = WORKSPACE / "features" / "cam16_convnext"
CAM17_FEAT  = WORKSPACE / "features" / "cam17_convnext"
UNI_PARQUET = WORKSPACE / "UNI features" / "results" / "uni_features_all_tcga.parquet"
OUT_ROOT    = WORKSPACE / "results" / "revision_v2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS       = [42, 123, 456]
N_FOLDS     = 5
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE   = torch.float16 if DEVICE == "cuda" else torch.float32
MODEL_IN    = 224
TILE_PX     = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def sanitize(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# TCGA ID utilities

def tcga_patient(sid: str) -> str:
    """TCGA-OR-A5J1-01Z-00-DX1.UUID -> TCGA-OR-A5J1"""
    m = re.match(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", sid)
    return m.group(1) if m else sid


def tcga_tss(sid: str) -> str:
    parts = sid.split("-")
    return parts[1] if len(parts) > 1 else "NA"


def safe_auc(y_true, y_proba, n_cls):
    if n_cls == 2:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_proba[:, 1])
    try:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except:
        aucs = []
        for k in range(n_cls):
            b = (y_true == k).astype(int)
            if 0 < b.sum() < len(b) and k < y_proba.shape[1]:
                try:
                    aucs.append(roc_auc_score(b, y_proba[:, k]))
                except:
                    pass
        return float(np.mean(aucs)) if aucs else np.nan


def bootstrap_ci(vals, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    boots = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# Models

class ConvNeXtTinyFeats(nn.Module):
    def __init__(self):
        super().__init__()
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = m.features
        self.gap = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.gap(self.features(x)).flatten(1)


_transform = T.Compose([
    T.Resize((MODEL_IN, MODEL_IN), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class PositionalEncoder(nn.Module):
    def __init__(self, d=768):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(3, d // 2), nn.GELU(), nn.Linear(d // 2, d))

    def forward(self, mmxy, s):
        return self.proj(torch.cat([mmxy, s], -1))


class MILTransformer(nn.Module):
    def __init__(self, d=768, h=8, L=6, ff=4, drop=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        el = nn.TransformerEncoderLayer(d, h, int(ff * d), drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, L)
        self.ln = nn.LayerNorm(d)
        self.pos = PositionalEncoder(d)
        self.proj_global = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.proj_token  = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pred_global = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.pred_token  = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))


class TransformerMIL_6B(nn.Module):
    def __init__(self, d=768, h=8, L=6, drop=0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        el = nn.TransformerEncoderLayer(
            d, h, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(el, L)
        self.ln = nn.LayerNorm(d)


class BYOLHead(nn.Module):
    def __init__(self, d=768, p=256):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, p))
        self.pred = nn.Sequential(nn.Linear(p, p), nn.GELU(), nn.Linear(p, p))


class EncoderWrapper_6B(nn.Module):
    def __init__(self, d=768, h=8, L=6, drop=0.1, p=256):
        super().__init__()
        self.backbone = TransformerMIL_6B(d, h, L, drop)
        self.head = BYOLHead(d, p)


# A5: Parameter count

def run_A5():
    log("A5: PARAMETER COUNT VERIFICATION")
    out = OUT_ROOT / "A5_params"
    out.mkdir(parents=True, exist_ok=True)
    r = {}
    for name, m in [
        ("ConvNeXt-Tiny backbone", ConvNeXtTinyFeats()),
        ("MILTransformer (Script6)", MILTransformer()),
        ("EncoderWrapper (Script6B)", EncoderWrapper_6B()),
    ]:
        n = sum(p.numel() for p in m.parameters())
        log(f"  {name}: {n:,} ({n / 1e6:.2f}M)")
        r[name] = n
        del m
    bb = r["ConvNeXt-Tiny backbone"]
    for agg_name in ["MILTransformer (Script6)", "EncoderWrapper (Script6B)"]:
        total = bb + r[agg_name]
        log(f"  TOTAL (backbone + {agg_name.split('(')[1][:-1]}): {total:,} ({total / 1e6:.2f}M)")
        r[f"total_{agg_name}"] = total
    for nm, ref in [("UNI", 307e6), ("Virchow", 632e6), ("Virchow2G", 1850e6)]:
        log(f"  {nm} / OpenSlideFM = {ref / (bb + r['EncoderWrapper (Script6B)']):.1f}x")
    save_json(r, out / "parameter_counts.json")
    return r


# Feature cache (load all 20K slides into RAM once)

class FeatureStore:
    """Pre-loads all .npy features into RAM for fast repeated access."""

    def __init__(self):
        self.f05 = {}
        self.f20 = {}
        self.patient_to_slides = defaultdict(list)
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        log("  Loading ALL features into RAM (one-time cost)...")
        t0 = time.time()

        for tag, feat_dir, store in [("0.5um", FEAT_05, self.f05), ("2.0um", FEAT_20, self.f20)]:
            files = sorted(feat_dir.glob("*.npy"))
            files = [f for f in files if not f.stem.endswith("_meta")]
            for i, p in enumerate(files):
                sid = p.stem
                try:
                    arr = np.load(p, mmap_mode="r")
                    store[sid] = arr
                except Exception:
                    pass
                if (i + 1) % 5000 == 0:
                    log(f"    {tag}: {i + 1}/{len(files)}")
            log(f"    {tag}: {len(store)} slides loaded")

        for sid in self.f05:
            pid = tcga_patient(sid)
            self.patient_to_slides[pid].append(sid)

        self._loaded = True
        log(
            f"  Feature store ready: {len(self.f05)} @0.5, {len(self.f20)} @2.0, "
            f"{len(self.patient_to_slides)} patients -- {time.time() - t0:.1f}s"
        )

    def get_slide_vector(self, sid, k_05=0, k_20=0, rng=None):
        """Mean-pool (optionally subsampled) features, concat both scales -> (1536,)"""
        parts = []
        for store, k in [(self.f05, k_05), (self.f20, k_20)]:
            if k > 0 and sid in store:
                arr = np.asarray(store[sid], dtype=np.float32)
                n = arr.shape[0]
                if n == 0:
                    parts.append(np.zeros(768, np.float32))
                elif n <= k:
                    parts.append(arr.mean(axis=0))
                else:
                    idx = rng.choice(n, k, replace=False) if rng else np.arange(k)
                    parts.append(arr[idx].mean(axis=0))
            else:
                parts.append(np.zeros(768, np.float32))
        return np.concatenate(parts)

    def get_patient_vector(self, patient_id, k_05=1200, k_20=400, rng=None):
        """Aggregate across all slides for a patient (mean of slide vectors)."""
        slide_ids = self.patient_to_slides.get(patient_id, [])
        if not slide_ids:
            return None
        vecs = []
        for sid in slide_ids:
            if sid in self.f05 and sid in self.f20:
                vecs.append(self.get_slide_vector(sid, k_05, k_20, rng))
        if not vecs:
            return None
        return np.stack(vecs).mean(axis=0)


STORE = FeatureStore()


# A1: Fair 10-class comparison (patient-level matching)

def run_A1():
    log("A1: FAIR 10-CLASS COMPARISON (patient-level matching)")
    out = OUT_ROOT / "A1_10class"
    out.mkdir(parents=True, exist_ok=True)

    STORE.load()

    uni_df = pd.read_parquet(UNI_PARQUET)
    feat_cols = sorted([c for c in uni_df.columns if c.startswith("f")])
    uni_df["label"] = uni_df["cancer_type"].str.replace("TCGA-", "", regex=False)
    uni_df["patient"] = uni_df["slide_id"].apply(tcga_patient)
    log(f"  UNI: {len(uni_df)} slides, {uni_df['patient'].nunique()} patients, {len(feat_cols)}-d")

    log("  Patient-level matching...")
    rng = np.random.default_rng(SEED)
    rows = []
    matched_patients = set()

    for pid, grp in uni_df.groupby("patient"):
        label = grp["label"].iloc[0]
        uni_vec = grp[feat_cols].values.astype(np.float32).mean(axis=0)
        osfm_vec = STORE.get_patient_vector(pid, k_05=1200, k_20=400, rng=rng)
        if osfm_vec is not None:
            rows.append({
                "patient": pid,
                "label": label,
                "uni_vec": uni_vec,
                "osfm_vec": osfm_vec,
                "n_uni_slides": len(grp),
                "n_osfm_slides": len(STORE.patient_to_slides.get(pid, [])),
            })
            matched_patients.add(pid)

    n_uni_patients = uni_df["patient"].nunique()
    log(f"  Matched patients: {len(rows)} / {n_uni_patients} ({100 * len(rows) / n_uni_patients:.1f}%)")

    labels = np.array([r["label"] for r in rows])
    for cls in sorted(set(labels)):
        log(f"    {cls}: {(labels == cls).sum()}")

    X_uni  = np.stack([r["uni_vec"] for r in rows]).astype(np.float32)
    X_osfm = np.stack([r["osfm_vec"] for r in rows]).astype(np.float32)
    patients = np.array([r["patient"] for r in rows])
    groups   = np.array([tcga_tss(p) for p in patients])

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = list(le.classes_)
    n_cls = len(classes)
    log(f"  Dataset: {len(y)} patients, {n_cls} classes, {len(set(groups))} TSS groups")

    log("  Running 5-fold x 3-seed CV (patient-level, TSS-grouped)...")
    all_uni, all_osfm = [], []
    fold_i = 0

    for r, seed_r in enumerate(SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_r)
        for f_idx, (tr, va) in enumerate(sgkf.split(np.arange(len(y)), y, groups)):
            fold_i += 1
            for X, res_list, nm in [(X_uni, all_uni, "UNI"), (X_osfm, all_osfm, "OSFM")]:
                sc = StandardScaler()
                Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
                clf = LogisticRegression(
                    solver="lbfgs", max_iter=2000, C=1.0,
                    class_weight="balanced", random_state=SEED
                )
                clf.fit(Xtr, y[tr])
                yp = clf.predict(Xva)
                proba = clf.predict_proba(Xva)
                res_list.append({
                    "seed": seed_r,
                    "fold": f_idx + 1,
                    "accuracy": accuracy_score(y[va], yp),
                    "balanced_accuracy": balanced_accuracy_score(y[va], yp),
                    "f1_macro": f1_score(y[va], yp, average="macro", zero_division=0),
                    "auc_macro": safe_auc(y[va], proba, n_cls),
                })
            log(f"    [{fold_i}/15] UNI acc={all_uni[-1]['accuracy']:.4f} | "
                f"OSFM acc={all_osfm[-1]['accuracy']:.4f}")

    df_uni, df_osfm = pd.DataFrame(all_uni), pd.DataFrame(all_osfm)

    def summarize(df, name):
        log(f"\n  {name}:")
        d = {}
        for m in ["accuracy", "balanced_accuracy", "f1_macro", "auc_macro"]:
            vals = df[m].dropna().values
            ci = bootstrap_ci(vals)
            log(f"    {m:22s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  CI [{ci[0]:.4f}, {ci[1]:.4f}]")
            d[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "ci": list(ci)}
        return d

    s_uni  = summarize(df_uni, "UNI2-h")
    s_osfm = summarize(df_osfm, "OpenSlideFM")
    gap   = s_uni["accuracy"]["mean"] - s_osfm["accuracy"]["mean"]
    ratio = s_osfm["accuracy"]["mean"] / max(s_uni["accuracy"]["mean"], 1e-9) * 100
    log(f"\n  GAP: {gap * 100:.1f}pp | Ratio: {ratio:.1f}%")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    yt_all, yp_all = [], []
    for tr, va in sgkf.split(np.arange(len(y)), y, groups):
        sc = StandardScaler()
        clf = LogisticRegression(
            solver="lbfgs", max_iter=2000, C=1.0,
            class_weight="balanced", random_state=SEED
        )
        clf.fit(sc.fit_transform(X_osfm[tr]), y[tr])
        yt_all.extend(y[va].tolist())
        yp_all.extend(clf.predict(sc.transform(X_osfm[va])).tolist())
    cm = confusion_matrix(yt_all, yp_all)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "osfm_confusion_matrix.csv")

    df_uni.to_csv(out / "uni_fold_metrics.csv", index=False)
    df_osfm.to_csv(out / "osfm_fold_metrics.csv", index=False)
    save_json({
        "matching": "patient-level (TCGA-XX-XXXX)",
        "n_patients": int(len(y)),
        "n_classes": n_cls,
        "classes": classes,
        "uni": s_uni,
        "osfm": s_osfm,
        "gap_pp": round(gap * 100, 2),
        "ratio_pct": round(ratio, 1),
    }, out / "summary.json")
    log(f"  Saved -> {out}")
    return s_uni, s_osfm


# A4: Ratio ablation (RAM-cached, fast)

def run_A4():
    log("A4: TOKEN RATIO ABLATION (total=1600, RAM-cached)")
    out = OUT_ROOT / "A4_ratio"
    out.mkdir(parents=True, exist_ok=True)

    STORE.load()

    df_man = pd.read_csv(MANIFESTS / "manifest_tcga.csv")
    df_man["slide_id"] = df_man["slide_id"].astype(str)

    both   = sorted(set(STORE.f05.keys()) & set(STORE.f20.keys()))
    df_man = df_man[df_man["slide_id"].isin(both)].reset_index(drop=True)
    sids   = df_man["slide_id"].values
    y_labels = df_man["cancer_code"].values
    groups = np.array([tcga_tss(s) for s in sids])
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    n_cls = len(le.classes_)
    log(f"  Slides: {len(y)}, Classes: {n_cls}")

    TOTAL = 1600
    RATIOS = [
        ("hi_only", 1.0,   0.0),
        ("4:1",     0.8,   0.2),
        ("3:1",     0.75,  0.25),
        ("2:1",     0.667, 0.333),
        ("1:1",     0.5,   0.5),
        ("1:2",     0.333, 0.667),
        ("lo_only", 0.0,   1.0),
    ]

    all_results = []

    for ratio_name, frac_05, frac_20 in RATIOS:
        k_05 = int(round(TOTAL * frac_05))
        k_20 = TOTAL - k_05
        log(f"\n  {ratio_name} ({k_05}:{k_20})")

        t0 = time.time()
        rng = np.random.default_rng(SEED)
        X = np.stack([STORE.get_slide_vector(sid, k_05, k_20, rng) for sid in sids])
        X = sanitize(X)
        log(f"    Vectors built: {X.shape} in {time.time() - t0:.1f}s")

        metrics = defaultdict(list)
        t1 = time.time()
        for seed_r in SEEDS:
            sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed_r)
            for tr, va in sgkf.split(np.arange(len(y)), y, groups):
                sc = StandardScaler()
                Xtr, Xva = sc.fit_transform(X[tr]), sc.transform(X[va])
                clf = LogisticRegression(
                    solver="lbfgs", max_iter=1000, C=1.0,
                    class_weight="balanced", random_state=SEED
                )
                clf.fit(Xtr, y[tr])
                yp = clf.predict(Xva)
                proba = clf.predict_proba(Xva)
                metrics["accuracy"].append(accuracy_score(y[va], yp))
                metrics["balanced_accuracy"].append(balanced_accuracy_score(y[va], yp))
                metrics["f1_macro"].append(f1_score(y[va], yp, average="macro", zero_division=0))
                metrics["auc_macro"].append(safe_auc(y[va], proba, n_cls))

        row = {"ratio": ratio_name, "k_05": k_05, "k_20": k_20}
        for m, vals in metrics.items():
            arr = np.array(vals)
            row[f"{m}_mean"] = round(float(np.mean(arr)), 4)
            row[f"{m}_std"]  = round(float(np.std(arr)), 4)
        all_results.append(row)
        log(f"    acc={row['accuracy_mean']:.4f}+/-{row['accuracy_std']:.4f} "
            f"auc={row['auc_macro_mean']:.4f} ({time.time() - t1:.0f}s)")

    df = pd.DataFrame(all_results)
    df.to_csv(out / "ratio_ablation.csv", index=False)
    log(f"\n  RESULTS:")
    log(df[["ratio", "k_05", "k_20", "accuracy_mean", "accuracy_std", "auc_macro_mean"]].to_string(index=False))
    log(f"  Saved -> {out / 'ratio_ablation.csv'}")
    return df


# CAMELYON shared: ConvNeXt extraction

def extract_camelyon_convnext(tag, raw_root, feat_root, budgets=None):
    import openslide
    from PIL import Image

    if budgets is None:
        budgets = {0.5: 1200, 2.0: 400}

    out_05 = feat_root / "scale0p5"
    out_20 = feat_root / "scale2p0"
    out_05.mkdir(parents=True, exist_ok=True)
    out_20.mkdir(parents=True, exist_ok=True)

    exts = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
    bad  = ["mask", "anno", "xml", "overlay", "thumb", "tissue", "prob", "heatmap"]

    slides = sorted([
        p for p in raw_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
        and not any(b in p.stem.lower() for b in bad)
    ])
    log(f"  {tag}: {len(slides)} candidate files")

    valid = []
    for p in slides:
        try:
            s = openslide.OpenSlide(str(p))
            w, h = s.dimensions
            s.close()
            if max(w, h) >= 5000:
                valid.append(p)
        except:
            pass
    log(f"  {tag}: {len(valid)} valid WSIs")

    model = ConvNeXtTinyFeats().to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)

    for si, sp in enumerate(valid):
        sid = sp.stem.lower()
        m = re.search(r'(tumor|normal|patient)[\-_]?(\d+)', sid)
        sid = f"{m.group(1)}_{int(m.group(2)):03d}" if m else sp.stem

        for target_um, budget in budgets.items():
            od  = out_05 if abs(target_um - 0.5) < 0.1 else out_20
            npy = od / f"{sid}.npy"
            if npy.exists():
                continue

            try:
                slide = openslide.OpenSlide(str(sp))
            except:
                continue

            base_mpp = 0.25
            for k in ("openslide.mpp-x", "aperio.MPP"):
                try:
                    base_mpp = float(slide.properties[k])
                    break
                except:
                    pass

            best_lvl, best_mpp = 0, base_mpp
            for lvl in range(slide.level_count):
                mpp = base_mpp * slide.level_downsamples[lvl]
                if abs(mpp - target_um) < abs(best_mpp - target_um):
                    best_mpp, best_lvl = mpp, lvl

            w_l, h_l = slide.level_dimensions[best_lvl]
            ds = slide.level_downsamples[best_lvl]

            coords = [
                (x, y)
                for y in range(0, max(1, h_l - TILE_PX + 1), TILE_PX)
                for x in range(0, max(1, w_l - TILE_PX + 1), TILE_PX)
            ]

            rng = np.random.default_rng(SEED)
            if len(coords) > budget * 5:
                sample_idx = rng.choice(len(coords), budget * 5, replace=False)
                coords = [coords[i] for i in sample_idx]

            tissue = []
            for xc, yc in coords:
                x0, y0 = int(round(xc * ds)), int(round(yc * ds))
                try:
                    t = np.asarray(
                        slide.read_region((x0, y0), best_lvl, (TILE_PX, TILE_PX)).convert("RGB")
                    )
                    if t.mean() < 235 and t.std() > 10:
                        tissue.append((xc, yc, x0, y0))
                except:
                    pass
                if len(tissue) >= budget:
                    break

            if not tissue:
                np.save(npy, np.zeros((0, 768), np.float32))
                slide.close()
                continue

            all_feats = []
            batch = []
            for xc, yc, x0, y0 in tissue:
                try:
                    img = slide.read_region((x0, y0), best_lvl, (TILE_PX, TILE_PX)).convert("RGB")
                    batch.append(_transform(img))
                except:
                    continue
                if len(batch) >= 2048:
                    bt = torch.stack(batch).to(DEVICE, memory_format=torch.channels_last)
                    with torch.no_grad(), torch.amp.autocast("cuda", AMP_DTYPE, DEVICE == "cuda"):
                        all_feats.append(model(bt).cpu().numpy())
                    batch.clear()
                    del bt
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

            if batch:
                bt = torch.stack(batch).to(DEVICE, memory_format=torch.channels_last)
                with torch.no_grad(), torch.amp.autocast("cuda", AMP_DTYPE, DEVICE == "cuda"):
                    all_feats.append(model(bt).cpu().numpy())
                del bt

            np.save(
                npy,
                np.concatenate(all_feats).astype(np.float16) if all_feats
                else np.zeros((0, 768), np.float32)
            )
            slide.close()

        if (si + 1) % 10 == 0 or si + 1 == len(valid):
            log(f"    [{si + 1}/{len(valid)}] {sid}")

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    n05 = len(list(out_05.glob("*.npy")))
    n20 = len(list(out_20.glob("*.npy")))
    log(f"  {tag} done: {n05} @0.5, {n20} @2.0")
    return out_05, out_20


# A2: CAMELYON16

def run_A2():
    log("A2: CAMELYON16 -- ConvNeXt-Tiny + Logistic Regression")
    out = OUT_ROOT / "A2_cam16"
    out.mkdir(parents=True, exist_ok=True)

    f05, f20 = extract_camelyon_convnext("CAM16", RAW_CAM16, CAM16_FEAT)

    both = sorted(
        set(p.stem for p in f05.glob("*.npy")) &
        set(p.stem for p in f20.glob("*.npy"))
    )
    sids, ys = [], []
    for s in both:
        if s.startswith("tumor"):
            sids.append(s); ys.append(1)
        elif s.startswith("normal"):
            sids.append(s); ys.append(0)
    y = np.array(ys)
    log(f"  Slides: {len(y)} (tumor={y.sum()}, normal={(1 - y).sum()})")
    if len(y) < 20:
        log("  [ERROR] Too few slides")
        return None

    X = sanitize(np.stack([
        np.concatenate([
            np.load(f05 / f"{s}.npy").astype(np.float32).mean(0)
            if np.load(f05 / f"{s}.npy").shape[0] > 0 else np.zeros(768, np.float32),
            np.load(f20 / f"{s}.npy").astype(np.float32).mean(0)
            if np.load(f20 / f"{s}.npy").shape[0] > 0 else np.zeros(768, np.float32),
        ])
        for s in sids
    ]))

    oof = np.zeros(len(y), np.float32)
    folds = []
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        sc  = StandardScaler()
        clf = LogisticRegression(
            C=0.3, class_weight="balanced", max_iter=5000,
            solver="lbfgs", random_state=SEED
        )
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        p   = clf.predict_proba(sc.transform(X[va]))[:, 1]
        oof[va] = p
        auc = roc_auc_score(y[va], p) if len(np.unique(y[va])) > 1 else np.nan
        acc = accuracy_score(y[va], (p >= 0.5).astype(int))
        f1  = f1_score(y[va], (p >= 0.5).astype(int), zero_division=0)
        folds.append({"fold": fold, "auc": auc, "acc": acc, "f1": f1})
        log(f"    Fold {fold}: AUC={auc:.4f} ACC={acc:.3f} F1={f1:.3f}")

    auc_oof = roc_auc_score(y, oof)
    ci = bootstrap_ci(np.array([f["auc"] for f in folds]))
    log(f"\n  OOF AUC: {auc_oof:.4f} CI [{ci[0]:.4f}, {ci[1]:.4f}]")

    pd.DataFrame(folds).to_csv(out / "cam16_folds.csv", index=False)
    save_json({
        "auc_oof": float(auc_oof),
        "ci95": list(ci),
        "acc_oof": float(accuracy_score(y, (oof >= 0.5).astype(int))),
        "backbone": "ConvNeXt-Tiny",
        "classifier": "LogReg(C=0.3)",
    }, out / "cam16_results.json")
    log(f"  Saved -> {out}")
    return folds


# A3: CAMELYON17

def run_A3():
    log("A3: CAMELYON17 -- ConvNeXt-Tiny + LOCO-CV")
    out = OUT_ROOT / "A3_cam17"
    out.mkdir(parents=True, exist_ok=True)

    f05, f20 = extract_camelyon_convnext("CAM17", RAW_CAM17, CAM17_FEAT)

    label_file = None
    for c in RAW_CAM17.rglob("*.csv"):
        try:
            df = pd.read_csv(c)
            cols = set(x.lower() for x in df.columns)
            if ("patient" in cols or "case" in cols) and ("pn" in cols or "stage" in cols):
                label_file = c
                break
        except:
            pass
    if not label_file:
        log("  [ERROR] No label CSV found")
        return None

    df = pd.read_csv(label_file)
    df.columns = [c.lower().strip() for c in df.columns]
    if "case" in df.columns and "patient" not in df.columns:
        df["patient"] = df["case"]
    if "stage" in df.columns and "pn" not in df.columns:
        df["pn"] = df["stage"]

    def norm_p(p):
        m = re.search(r"patient[\-_]?(\d+)", str(p).lower())
        return f"patient_{int(m.group(1)):03d}" if m else str(p)

    df["patient"] = df["patient"].apply(norm_p)

    def parse_pn(v):
        m = re.search(r"(\d)", str(v))
        return int(m.group(1)) if m else -1

    df["pn_int"] = df["pn"].apply(parse_pn)
    df = df[df["pn_int"] >= 0]

    if "center" not in df.columns:
        df["center"] = df.get("centerid", pd.Series([-1] * len(df)))
    df["center"] = df["center"].apply(
        lambda v: int(re.search(r"(\d+)", str(v)).group(1))
        if re.search(r"(\d+)", str(v)) else -1
    )
    log(f"  Labels: {len(df)} patients, centers={sorted(df['center'].unique())}")

    both = set(p.stem for p in f05.glob("*.npy")) & set(p.stem for p in f20.glob("*.npy"))

    def pid_from_slide(s):
        m = re.search(r"patient[\-_]?(\d+)", s.lower())
        return f"patient_{int(m.group(1)):03d}" if m else s

    pvecs = defaultdict(list)
    for sid in both:
        pid = pid_from_slide(sid)
        a   = np.load(f05 / f"{sid}.npy").astype(np.float32)
        b   = np.load(f20 / f"{sid}.npy").astype(np.float32)
        v05 = a.mean(0) if a.shape[0] > 0 else np.zeros(768, np.float32)
        v20 = b.mean(0) if b.shape[0] > 0 else np.zeros(768, np.float32)
        pvecs[pid].append(np.concatenate([v05, v20]))

    pfeat = {pid: np.stack(vs).max(0) for pid, vs in pvecs.items()}

    pts, X_l, y_l, c_l = [], [], [], []
    for _, r in df.iterrows():
        if r["patient"] in pfeat:
            pts.append(r["patient"])
            X_l.append(pfeat[r["patient"]])
            y_l.append(r["pn_int"])
            c_l.append(r["center"])
    X = sanitize(np.stack(X_l))
    y = np.array(y_l)
    c = np.array(c_l)
    log(f"  Matched: {len(y)}, pN dist: {dict(zip(*np.unique(y, return_counts=True)))}")

    centers = sorted([int(v) for v in np.unique(c) if v >= 0])
    folds = []
    yt_all, yp_all = [], []

    if len(centers) >= 2:
        log(f"  LOCO-CV, centers: {centers}")
        for ctr in centers:
            tr, va = np.where(c != ctr)[0], np.where(c == ctr)[0]
            if len(va) == 0 or len(np.unique(y[tr])) < 2:
                continue
            sc  = StandardScaler()
            clf = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=5000,
                solver="lbfgs", random_state=SEED
            )
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            yp  = clf.predict(sc.transform(X[va]))
            k   = cohen_kappa_score(y[va], yp, weights="quadratic")
            acc = accuracy_score(y[va], yp)
            folds.append({"center": ctr, "n": len(va), "kappa": float(k), "acc": float(acc)})
            yt_all.extend(y[va].tolist())
            yp_all.extend(yp.tolist())
            log(f"    Center {ctr}: n={len(va)} kappa={k:.4f} acc={acc:.3f}")
    else:
        log("  [WARN] <2 centers, using 5-fold CV")
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            sc  = StandardScaler()
            clf = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=5000,
                solver="lbfgs", random_state=SEED
            )
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            yp  = clf.predict(sc.transform(X[va]))
            k   = cohen_kappa_score(y[va], yp, weights="quadratic")
            folds.append({"fold": fold, "kappa": float(k), "acc": float(accuracy_score(y[va], yp))})
            yt_all.extend(y[va].tolist())
            yp_all.extend(yp.tolist())

    df_f    = pd.DataFrame(folds)
    mean_k  = df_f["kappa"].mean()
    std_k   = df_f["kappa"].std()
    ci      = bootstrap_ci(df_f["kappa"].values)
    cm      = confusion_matrix(yt_all, yp_all)
    log(f"\n  kappa={mean_k:.4f}+/-{std_k:.4f} CI [{ci[0]:.4f},{ci[1]:.4f}]")
    log(f"  Confusion:\n{cm}")

    df_f.to_csv(out / "cam17_folds.csv", index=False)
    save_json({
        "kappa_mean": float(mean_k),
        "kappa_std": float(std_k),
        "ci95": list(ci),
        "acc_mean": float(df_f["acc"].mean()),
        "confusion": cm.tolist(),
        "backbone": "ConvNeXt-Tiny",
        "protocol": "LOCO-CV" if len(centers) >= 2 else "5-fold",
    }, out / "cam17_results.json")
    log(f"  Saved -> {out}")
    return df_f


# Main

def main():
    t0 = time.time()
    log("OPENSLIDEFM REVISION V2")
    log(f"  Workspace: {WORKSPACE}")
    log(f"  Device: {DEVICE}" + (f" -- {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""))

    run_A5()
    run_A1()
    run_A4()
    run_A2()
    run_A3()

    log(f"\nALL DONE -- {(time.time() - t0) / 3600:.1f} hours")
    log(f"Results: {OUT_ROOT}")


if __name__ == "__main__":
    main()
