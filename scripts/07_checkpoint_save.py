#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSlideFM - Finalize & Save Encoder Checkpoint
============================================================

This script is part of the OpenSlideFM pipeline for computational pathology.

Paper: "OpenSlideFM: A Resource-Efficient Foundation Model for 
        Computational Pathology on Whole Slide Images"

Authors: Sanwal Ahmad Zafar, Wei Qin
Institution: Shanghai Jiao Tong University

License: Apache 2.0
"""

# Script 6B — Finalize & Save Encoder Checkpoint 

import os, sys, json, time, random, shutil, subprocess, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

# ----------------------- Paths -----------------------
WORKSPACE   = Path(os.environ.get("WORKSPACE", "./workspace")))
FEAT05_DIR  = WORKSPACE / "features" / "scale0p5"
FEAT20_DIR  = WORKSPACE / "features" / "scale2p0"
MODELS_DIR  = WORKSPACE / "models"
LOGS_DIR    = WORKSPACE / "logs"
for p in (MODELS_DIR, LOGS_DIR): p.mkdir(parents=True, exist_ok=True)

STUDENT_OUT = MODELS_DIR / "openslidefm_student.pt"
TEACHER_OUT = MODELS_DIR / "openslidefm_teacher_ema.pt"
MANIFEST    = MODELS_DIR / "openslidefm_checkpoint_manifest.json"
TRAIN_LOG   = LOGS_DIR / "script6c_finalize_log.csv"

# ----------------------- Deps ------------------------
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

# ------------------- Config -------------------
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

# ------------------- Utilities -------------------
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

# ------------------- Model -------------------
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

# ------------------- Main -------------------
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

        # ---- batch: fixed-shape tokens for all slides ----
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

    # ---- Save final checkpoints ----
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
