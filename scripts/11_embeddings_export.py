#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Slide Embeddings Export
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# Script8 — Slide Embeddings Export 

import os, sys, json, time, math, shutil, gc
from pathlib import Path
from datetime import datetime
import subprocess, warnings
warnings.filterwarnings("ignore")

# --------------- Workspace ---------------
WORKSPACE = Path(r"D:\个人文件夹\Sanwal\OpenSlide")
FEAT05 = WORKSPACE / "features" / "scale0p5"
FEAT20 = WORKSPACE / "features" / "scale2p0"
EMB_DIR = WORKSPACE / "embeddings"
MANIFESTS = WORKSPACE / "manifests"
DIAG = WORKSPACE / "diagnostics"
for p in [EMB_DIR, DIAG]:
    p.mkdir(parents=True, exist_ok=True)

# --------------- Deps ---------------
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

# --------------- Load manifests (dataset tagging) ---------------
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

# --------------- Discover features present at both scales ---------------
def available_two_scale_ids():
    s05 = set([p.stem for p in FEAT05.glob("*.npy")])
    s20 = set([p.stem for p in FEAT20.glob("*.npy")])
    both = sorted(list(s05 & s20))
    return both

TWO_SCALE_IDS = available_two_scale_ids()

# --------------- Decide dataset for each slide_id ---------------
def dataset_of(slide_id: str) -> str:
    # Priority: camelyon16 / camelyon17 / tcga / other
    if slide_id in SLIDESETS.get("camelyon16", set()): return "CAMELYON16"
    if slide_id in SLIDESETS.get("camelyon17", set()): return "CAMELYON17"
    if slide_id in SLIDESETS.get("tcga", set()):       return "TCGA"
    return "OTHER"

# --------------- Export logic ---------------
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

# --------------- Driver ---------------
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
