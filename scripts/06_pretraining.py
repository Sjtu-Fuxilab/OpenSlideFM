#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Two-Scale Feature-Space Pretraining
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# Script 6 — Two-Scale Feature-Space Pretraining 

import os, sys, json, math, random, gc, subprocess, platform
from pathlib import Path
from time import perf_counter
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional

# --------------------------- Workspace ---------------------------
WORKSPACE = Path(r"D:\个人文件夹\Sanwal\OpenSlide").resolve()
FEATURES05 = WORKSPACE / "features" / "scale0p5"
FEATURES20 = WORKSPACE / "features" / "scale2p0"
LOGS       = WORKSPACE / "logs"
WEIGHTS    = WORKSPACE / "weights"
FIGS       = WORKSPACE / "figures"
EMBED      = WORKSPACE / "embeddings" / "student_final"
for p in [LOGS, WEIGHTS, FIGS, EMBED]:
    p.mkdir(parents=True, exist_ok=True)
    assert str(p).startswith(str(WORKSPACE)), f"Output path escapes WORKSPACE: {p}"

# --------------------------- Robust deps (no hard failures for optional libs) ---------------------------
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

# --------------------------- Config ---------------------------
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

# --------------------------- Reproducibility ---------------------------
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

# --------------------------- Slide inventory (require both scales) ---------------------------
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

# --------------------------- Meta loading (robust to column names) ---------------------------
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

# --------------------------- MIL model ---------------------------
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

# --------------------------- Losses & EMA ---------------------------
def cosine_loss(p: torch.Tensor, z: torch.Tensor):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z.detach(), dim=-1)
    return (1.0 - (p * z).sum(dim=-1)).mean()

@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, tau: float):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.data.mul_(tau).add_(ps.data, alpha=(1.0 - tau))

# --------------------------- Build models/opt ---------------------------
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

# --------------------------- Token sampling & batching ---------------------------
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

# --------------------------- Cosine scheduler w/ warmup ---------------------------
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

# --------------------------- Logging & checkpoints ---------------------------
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

# --------------------------- Training loop ---------------------------
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

# --------------------------- Optional: quick curve (skips if matplotlib missing) ---------------------------
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

# --------------------------- Optional: export slide embeddings ---------------------------
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
