#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_rag.py — tool-using bibliographic agent for clandestine-printer attribution
================================================================================


NEW IN THIS VERSION:

  1. CONTRASTIVE ENCODER — Trains a small CNN on within-printer same-character positive pairs using NT-Xent.
     Toggle USE_CONTRASTIVE_ENCODER=True. Cached weights and embeddings are
     namespaced so A/B compare the two encoders.
  2. AGGRESSIVE TOOL CHAINING — system prompt now includes explicit
     multi-step patterns. Agent typically chains 3-5 tool calls before
     answering rather than stopping at 1.
  3. TRACE PERSISTENCE — every investigate() call writes its full trace to
     sort_rag_run/traces/<timestamp>.json with a running index.json. Lets
     you audit agent reasoning across many investigations.

Run:
    ollama serve &
    ollama pull qwen2.5:14b
    pip install lancedb sentence-transformers
    python real_rag.py

Interactive:
    >>> investigate("Who printed ESTC R21865? Treat the imprint as unknown.")
    >>> investigate("Audit cluster C::0000")
    >>> investigate("Compare Tyler and Simmons fingerprints")
"""

#CONFIG  ────────────────────────────────────────────────────────────────
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import sys as _sys
from pathlib import Path

OUT_DIR     = Path("./sort_rag_run")
DATA_DIR    = OUT_DIR / "data" / "cdt"

CROP_SIZE         = 96
DAMAGE_THRESHOLD  = 0.18
DAMAGE_DILATE     = 1
BATCH_SIZE        = 64

# === Encoder selection ===
# False -> use frozen DINOv2 (ViT-S/14). Faster setup
# True  -> train a contrastive CNN on damage-masked residuals. learns damage-specific
#          features the frozen ViT can't.
#
# Can be overridden at the CLI: `python real_rag.py --encoder dinov2` or
# `--encoder contrastive`. Useful for A/B comparison without editing the file.
USE_CONTRASTIVE_ENCODER = True

# CLI override for encoder
if "--encoder" in _sys.argv:
    _idx = _sys.argv.index("--encoder")
    if _idx + 1 < len(_sys.argv):
        _choice = _sys.argv[_idx + 1].lower()
        if _choice == "dinov2":
            USE_CONTRASTIVE_ENCODER = False
            print(f"[cli] --encoder dinov2 -> USE_CONTRASTIVE_ENCODER=False")
        elif _choice == "contrastive":
            USE_CONTRASTIVE_ENCODER = True
            print(f"[cli] --encoder contrastive -> USE_CONTRASTIVE_ENCODER=True")
        else:
            print(f"[cli] unknown --encoder value '{_choice}'; "
                  f"using USE_CONTRASTIVE_ENCODER={USE_CONTRASTIVE_ENCODER}")

# Retrain contrastive encoder (deletes cached weights and embeddings).
# Use after changing CONTRASTIVE_* hyperparameters.
FORCE_RETRAIN_CONTRASTIVE = "--retrain" in _sys.argv

DINOV2_MODEL          = "dinov2_vits14"
TEXT_EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"

# Contrastive training hyperparameters (only used if USE_CONTRASTIVE_ENCODER=True)
CONTRASTIVE_EMBED_DIM = 128
CONTRASTIVE_EPOCHS    = 100
CONTRASTIVE_LR        = 1e-3
CONTRASTIVE_BATCH     = 128
CONTRASTIVE_TEMPERATURE = 0.07

# Namespaced paths so the two encoders coexist without colliding
ENCODER_TAG = "contrastive" if USE_CONTRASTIVE_ENCODER else "dinov2"
RUN_DIR     = OUT_DIR / "runs"  / f"cdt_{ENCODER_TAG}"
LANCE_DIR   = OUT_DIR / "lancedb" / ENCODER_TAG
TRACE_DIR   = OUT_DIR / "traces"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LANCE_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# Agent
AGENT_MODEL       = "qwen2.5:14b"
AGENT_MAX_STEPS   = 8
AGENT_TEMPERATURE = 0.2

HDBSCAN_MIN_CLUSTER_FRAC = 0.04
KMEANS_K_RANGE           = (2, 12)

# Calibration thresholds
LEAKAGE_COSINE      = 0.90
HIGH_CONFIDENCE     = 0.70
MOD_CONFIDENCE      = 0.40
NARROW_GAP_DEMOTION = 0.10
TINY_GAP_DEMOTION   = 0.05

REBUILD_LANCEDB = False

# Empirical baselines for THIS encoder. Auto-loaded from the most recent
# evaluate_encoder() output for the active encoder. If no eval has been
# run yet, falls back to the DINOv2.
_BASELINES_FALLBACK = {
    "n_books_evaluated": 81,
    "honest_recall_at_3": 0.284,
    "chance_baseline_at_3": 0.158,
    "lift_over_chance": 1.8,
    "median_cosine_when_rank_1": 0.656,
    "median_cosine_when_rank_not_1": 0.532,
    "encoder": "dinov2",
    "interpretation": (
        "Book-level attribution is a narrowing tool, not point-attribution. "
        "A cosine of 0.6 has roughly 50/50 odds of being correct. Real "
        "confidence requires corroborating bibliographic evidence."
    ),
    "source": "fallback (DINOv2 )",
}


def _load_baselines_for_current_encoder():
    """Load empirical baselines from disk if evaluate_encoder() has been run
    for the current encoder. Otherwise return the fallback with a warning.

    This prevents the stale-baseline bug where the agent would report
    DINOv2 numbers under the contrastive encoder."""
    import json as _json
    eval_path = RUN_DIR / f"recall_at_3_{ENCODER_TAG}.json"
    if not eval_path.exists():
        warned = dict(_BASELINES_FALLBACK)
        warned["encoder"] = ENCODER_TAG  # tag matches even if numbers don't
        warned["source"] = (
            f"FALLBACK — no eval has been run for encoder={ENCODER_TAG}. "
            f"Run `python real_rag.py --encoder {ENCODER_TAG} --evaluate` "
            f"to populate real numbers. Reported numbers below are from "
            f"the original DINOv2 validation and DO NOT reflect this "
            f"encoder's actual performance."
        )
        print(f"[baselines] WARNING: no eval file at {eval_path}")
        print(f"[baselines] using DINOv2 fallback numbers; "
              f"run --evaluate to refresh")
        return warned

    try:
        d = _json.loads(eval_path.read_text())
    except Exception as e:
        print(f"[baselines] failed to read {eval_path}: {e}; using fallback")
        return dict(_BASELINES_FALLBACK)

    # Newer eval JSON has {"overall": {...}}; older has flat keys. Both
    # also have top-level aliases (honest_recall_at_k, etc.) for compat.
    n_books = d.get("n_books_evaluated") or d.get("overall", {}).get("n")
    honest = d.get("honest_recall_at_k") or d.get("overall", {}).get("honest_recall_at_k")
    raw    = d.get("raw_recall_at_k")    or d.get("overall", {}).get("raw_recall_at_k")
    chance = d.get("chance_baseline_at_k")
    lift   = d.get("honest_lift")
    cos1   = (d.get("median_cosine_when_rank_1")
              or d.get("overall", {}).get("median_cos_top1"))
    cos_o  = (d.get("median_cosine_when_rank_not_1")
              or d.get("overall", {}).get("median_cos_other"))
    n_leaked = d.get("n_leaked") or d.get("overall", {}).get("n_leaked")

    loaded = {
        "n_books_evaluated": n_books,
        "honest_recall_at_3": honest,
        "raw_recall_at_3":    raw,
        "chance_baseline_at_3": chance,
        "lift_over_chance":  lift,
        "median_cosine_when_rank_1":     cos1,
        "median_cosine_when_rank_not_1": cos_o,
        "n_leaked_suspects":  n_leaked,
        "encoder": ENCODER_TAG,
        "interpretation": _BASELINES_FALLBACK["interpretation"],
        "source": f"loaded from {eval_path.name}",
    }
    # Also include the stratification gap if present, since it's the
    # honest answer to "how much of this is house style?"
    if "stratification_gap" in d:
        loaded["house_style_drop_from_printer_holdout"] = d.get(
            "stratification_gap")

    print(f"[baselines] loaded {ENCODER_TAG} encoder baselines from "
          f"{eval_path.name}: honest recall@3={honest}, "
          f"lift={lift}x, leaked={n_leaked}")
    return loaded


EMPIRICAL_BASELINES = _load_baselines_for_current_encoder()

CDT_BASE = "https://cdt.library.cmu.edu"


# %% 1. SETUP  ────────────────────────────────────────────────────────────────
import json, os, re, sys, time, urllib.request, io, socket, random, math
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import hdbscan; HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.ndimage import binary_dilation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

OUT_DIR.mkdir(parents=True, exist_ok=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[setup] DEVICE={DEVICE} encoder={ENCODER_TAG} "
      f"hdbscan={HAS_HDBSCAN} sklearn={HAS_SKLEARN}")


# %% 2. CORPUS  ───────────────────────────────────────────────────────────────
@dataclass
class CDTRecord:
    id: str
    printer_slug: str
    char: str
    year: str
    image_path: str
    estc: str = ""
    title: str = ""
    publisher: str = ""
    iiif_url: str = ""


class CDTIndex:
    def __init__(self, root, crop_size=96):
        self.root = Path(root); self.crop_size = crop_size
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"No manifest at {manifest_path}. Run history_rag.py first "
                f"to download the CDT corpus.")
        m = json.loads(manifest_path.read_text())
        self.records = []
        for r in m["records"]:
            try:
                with Image.open(self.root / r["path"]) as im:
                    im.verify()
            except Exception:
                continue
            self.records.append(CDTRecord(
                id=r["id"], printer_slug=r["printer_slug"],
                char=r.get("char", "?"), year=r.get("year", ""),
                image_path=str(self.root / r["path"]),
                estc=r.get("estc", ""), title=r.get("title", ""),
                publisher=r.get("publisher", ""),
                iiif_url=r.get("iiif_url", ""),
            ))

    def __len__(self): return len(self.records)

    def get_crop(self, idx):
        r = self.records[idx]
        arr = np.array(Image.open(r.image_path).convert("L"))
        h, w = arr.shape; side = max(h, w)
        canvas = np.full((side, side), 255, dtype=np.uint8)
        canvas[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = arr
        return np.array(Image.fromarray(canvas).resize(
            (self.crop_size, self.crop_size), Image.BILINEAR))


def align_glyph(crop, size):
    arr = crop.astype(np.float32)
    if arr.shape != (size, size):
        arr = np.array(Image.fromarray(arr.astype(np.uint8))
                       .resize((size, size), Image.BILINEAR), dtype=np.float32)
    ink = 255.0 - arr
    if ink.sum() < 1e-3:
        return arr / 255.0
    ys, xs = np.indices(ink.shape)
    cy = (ys*ink).sum()/ink.sum(); cx = (xs*ink).sum()/ink.sum()
    dy, dx = int(round(size/2 - cy)), int(round(size/2 - cx))
    out = np.full_like(arr, 255.0)
    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(size, size+dy), min(size, size+dx)
    sy0, sx0 = max(0, -dy), max(0, -dx)
    out[y0:y1, x0:x1] = arr[sy0:sy0+(y1-y0), sx0:sx0+(x1-x0)]
    return out / 255.0


def build_templates(index, max_per_char=300):
    by_char = defaultdict(list)
    for i, r in enumerate(index.records):
        if r.char and r.char != "?":
            by_char[r.char].append(i)
    rng = np.random.default_rng(0); templates = {}
    for ch, idxs in by_char.items():
        sample = list(rng.choice(idxs, size=min(max_per_char, len(idxs)),
                                 replace=False))
        aligned = np.stack([align_glyph(index.get_crop(j), index.crop_size)
                            for j in sample])
        if aligned.shape[0] >= 5:
            s = np.sort(aligned, axis=0)
            lo = int(aligned.shape[0]*0.2)
            hi = max(lo+1, aligned.shape[0] - int(aligned.shape[0]*0.2))
            templates[ch] = s[lo:hi].mean(axis=0).astype(np.float32)
        else:
            templates[ch] = aligned.mean(axis=0).astype(np.float32)
    return templates


def damage_mask_input(crop, char, templates, size,
                      threshold=DAMAGE_THRESHOLD, dilate=DAMAGE_DILATE):
    aligned = align_glyph(crop, size)
    tpl = templates.get(char)
    if tpl is None:
        return aligned * 2.0 - 1.0
    res = aligned - tpl
    mask = np.abs(res) > threshold
    if HAS_SCIPY and dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)
    canvas = np.full_like(aligned, 0.5)
    canvas[mask] = aligned[mask]
    return canvas * 2.0 - 1.0


# %% 3. ENCODERS  ─────────────────────────────────────────────────────────────
class FrozenDINOv2(nn.Module):
    def __init__(self, name="dinov2_vits14"):
        super().__init__()
        self.backbone = None; self.embed_dim = None; self.kind = None
        errors = []
        try:
            import timm
            tn = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
                  "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m"}.get(
                name, "vit_small_patch14_dinov2.lvd142m")
            bb = timm.create_model(tn, pretrained=True, num_classes=0)
            bb.eval(); self.backbone = bb
            self.embed_dim = bb.num_features; self.kind = "timm"
            print(f"[encoder] loaded {tn} via timm; dim={self.embed_dim}")
        except Exception as e:
            errors.append(f"timm: {e}")
        if self.backbone is None:
            try:
                from transformers import AutoModel
                hf = {"dinov2_vits14": "facebook/dinov2-small"}.get(
                    name, "facebook/dinov2-small")
                bb = AutoModel.from_pretrained(hf)
                bb.eval(); self.backbone = bb
                self.embed_dim = bb.config.hidden_size
                self.kind = "transformers"
            except Exception as e:
                errors.append(f"transformers: {e}")
        if self.backbone is None:
            raise RuntimeError("DINOv2 unavailable:\n  " + "\n  ".join(errors))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.input_size = 518 if self.kind == "timm" else 224
        self.register_buffer("_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size,
                              mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        with torch.no_grad():
            if self.kind == "transformers":
                out = self.backbone(pixel_values=x)
                z = (out.pooler_output if getattr(out, "pooler_output", None)
                     is not None else out.last_hidden_state[:, 0])
            else:
                z = self.backbone(x)
        return F.normalize(z, p=2, dim=-1)


class ContrastiveCNN(nn.Module):
    """Small CNN trained on damage-masked residuals with NT-Xent loss.

    Architecture: 4 conv blocks (32->64->128->256 channels) with GELU and
    max-pool, then a 2-layer MLP projection to embed_dim. Operates directly
    on the 96x96 damage-masked input (no upsample, no patch embed)."""
    def __init__(self, embed_dim=128, in_channels=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(), nn.Linear(512, embed_dim),
        )
        self.embed_dim = embed_dim
        self.input_size = 96  # operates on raw crop size

    def forward(self, x):
        return F.normalize(self.proj(self.backbone(x)), p=2, dim=-1)


class _ContrastiveDataset(Dataset):
    """Positive pairs = same (printer, char). All other items in the batch
    that don't share the same character class become available negatives;
    same-char-different-printer pairs are masked out as ambiguous."""
    def __init__(self, index, templates):
        self.index = index; self.templates = templates
        self.size = index.crop_size
        self._by_key = defaultdict(list)
        for i, r in enumerate(index.records):
            self._by_key[(r.printer_slug, r.char)].append(i)

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        r = self.index.records[idx]
        pool = [j for j in self._by_key[(r.printer_slug, r.char)] if j != idx]
        pos = random.choice(pool) if pool else idx
        a = damage_mask_input(self.index.get_crop(idx), r.char,
                              self.templates, self.size)
        p = damage_mask_input(self.index.get_crop(pos),
                              self.index.records[pos].char,
                              self.templates, self.size)
        return (torch.from_numpy(a).float().unsqueeze(0),
                torch.from_numpy(p).float().unsqueeze(0), r.char)


def _nt_xent(z_a, z_p, char_ids, temperature):
    """Supervised contrastive loss: same-char items become potential
    positives (so the CNN doesn't waste capacity learning to separate A vs
    B, which is already given). Cross-char pairs are always negatives."""
    B = z_a.shape[0]
    z = torch.cat([z_a, z_p], dim=0)
    sim = z @ z.t() / temperature
    eye = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(eye, float("-inf"))
    targets = torch.cat([torch.arange(B, 2*B),
                          torch.arange(0, B)]).to(z.device)
    all_c = torch.cat([char_ids, char_ids], dim=0)
    same_char = all_c.unsqueeze(0) == all_c.unsqueeze(1)
    # Only positives we count: the anchor's pair AND any same-char same-batch
    # item. Negatives: any different-char item.
    same_char[torch.arange(2*B), targets] = True
    sim = sim.masked_fill(~same_char, float("-inf"))
    return F.cross_entropy(sim, targets)


def _collate_contrastive(batch):
    a, p, ch = zip(*batch); seen = {}; char_ids = []
    for c in ch:
        if c not in seen: seen[c] = len(seen)
        char_ids.append(seen[c])
    return torch.stack(a), torch.stack(p), torch.tensor(char_ids)


def train_contrastive_encoder(index, templates, epochs, lr, batch_size,
                              temperature, embed_dim):
    """Train ContrastiveCNN. Saves to RUN_DIR/encoder.pt and returns the
    trained model + loss history."""
    print(f"[train] contrastive CNN: {epochs} epochs, lr={lr}, "
          f"batch={batch_size}, temp={temperature}")
    ds = _ContrastiveDataset(index, templates)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=_collate_contrastive,
                        drop_last=True)
    model = ContrastiveCNN(embed_dim=embed_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total, n = 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for a, p, ch in pbar:
            a, p, ch = a.to(DEVICE), p.to(DEVICE), ch.to(DEVICE)
            loss = _nt_xent(model(a), model(p), ch, temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * a.shape[0]; n += a.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        history.append(total / max(n, 1))
    ckpt_path = RUN_DIR / "encoder.pt"
    torch.save({"model": model.state_dict(),
                "embed_dim": embed_dim,
                "history": history}, ckpt_path)
    print(f"[train] best loss = {min(history):.4f}; saved {ckpt_path}")
    return model, history


def load_or_train_encoder(index, templates):
    """Returns an encoder ready to embed glyphs. If contrastive is selected
    and a checkpoint exists, loads it; otherwise trains. Pass --retrain on
    the CLI to force retraining (deletes cached encoder + embeddings)."""
    if not USE_CONTRASTIVE_ENCODER:
        print(f"[encoder] loading frozen {DINOV2_MODEL}")
        return FrozenDINOv2(DINOV2_MODEL).to(DEVICE)

    ckpt_path = RUN_DIR / "encoder.pt"
    emb_path = RUN_DIR / "glyph_embeddings.npy"

    if FORCE_RETRAIN_CONTRASTIVE:
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"[encoder] --retrain: removed cached {ckpt_path}")
        if emb_path.exists():
            emb_path.unlink()
            print(f"[encoder] --retrain: removed cached embeddings")

    if ckpt_path.exists():
        print(f"[encoder] loading cached contrastive CNN from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=DEVICE)
        model = ContrastiveCNN(embed_dim=state["embed_dim"]).to(DEVICE)
        model.load_state_dict(state["model"])
        model.eval()
        return model

    print(f"[encoder] training contrastive CNN  "
          f"(epochs={CONTRASTIVE_EPOCHS}, lr={CONTRASTIVE_LR}, "
          f"temp={CONTRASTIVE_TEMPERATURE})")
    model, _ = train_contrastive_encoder(
        index, templates,
        epochs=CONTRASTIVE_EPOCHS, lr=CONTRASTIVE_LR,
        batch_size=CONTRASTIVE_BATCH,
        temperature=CONTRASTIVE_TEMPERATURE,
        embed_dim=CONTRASTIVE_EMBED_DIM,
    )
    model.eval()
    return model


@torch.no_grad()
def encode_all(model, index, templates, batch_size=64):
    model.eval(); embs = []; buf = []
    for i in tqdm(range(len(index)), desc="encoding glyphs"):
        r = index.records[i]
        buf.append(damage_mask_input(index.get_crop(i), r.char,
                                     templates, index.crop_size))
        if len(buf) >= batch_size:
            t = torch.from_numpy(np.stack(buf)).float().unsqueeze(1).to(DEVICE)
            embs.append(model(t).cpu().numpy()); buf = []
    if buf:
        t = torch.from_numpy(np.stack(buf)).float().unsqueeze(1).to(DEVICE)
        embs.append(model(t).cpu().numpy())
    return np.concatenate(embs, axis=0)


# %% 4. CLUSTERING  ──────────────────────────────────────────────────────────
def _hdbscan_with_relaxation(X, base_mcs):
    D = 1.0 - X @ X.T
    D = np.clip(D, 0.0, 2.0); np.fill_diagonal(D, 0.0); D = (D + D.T) / 2.0
    for mcs in (base_mcs, max(3, base_mcs - 1), 3):
        if mcs < 2: continue
        try:
            labels = hdbscan.HDBSCAN(
                min_cluster_size=mcs, min_samples=1,
                metric="precomputed", cluster_selection_method="eom",
            ).fit_predict(D.astype(np.float64))
        except Exception:
            continue
        if len({l for l in labels if l >= 0}) >= 1:
            return labels, mcs
    return np.full(len(X), -1, dtype=int), base_mcs


def _kmeans_silhouette(X, k_min, k_max):
    best = None
    upper = min(k_max, max(k_min + 1, len(X) - 1))
    for k in range(k_min, upper + 1):
        if k >= len(X): break
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if len(set(labels)) < 2: continue
        try:
            s = silhouette_score(X, labels, metric="cosine")
        except Exception:
            continue
        if best is None or s > best[1]:
            best = (labels, s)
    return best[0] if best else np.zeros(len(X), dtype=int)


def cluster_by_char(embs, chars, restrict_idxs=None):
    n = len(chars)
    assignments = np.array([""] * n, dtype=object)
    if restrict_idxs is None:
        restrict_idxs = list(range(n))
    by_char = defaultdict(list)
    for i in restrict_idxs:
        by_char[chars[i]].append(i)
    summary = []
    for ch, idxs in by_char.items():
        if len(idxs) < 6: continue
        X = embs[idxs]; labels = None
        if HAS_HDBSCAN:
            mcs = max(3, int(HDBSCAN_MIN_CLUSTER_FRAC * len(idxs)))
            mcs = min(mcs, max(3, len(idxs) // 4))
            labels, _ = _hdbscan_with_relaxation(X, mcs)
        if labels is None or len({l for l in labels if l >= 0}) == 0:
            if HAS_SKLEARN:
                k_min, k_max = KMEANS_K_RANGE
                k_max_eff = min(k_max, max(k_min, len(idxs) // 5))
                if k_max_eff >= k_min:
                    labels = _kmeans_silhouette(X, k_min, k_max_eff)
                else:
                    labels = np.zeros(len(idxs), dtype=int)
        for lab in sorted(set(labels)):
            if lab < 0: continue
            members = [idxs[j] for j, l in enumerate(labels) if l == lab]
            if len(members) < 3: continue
            cid = f"{ch}::{lab:04d}"
            cent = embs[members].mean(axis=0)
            cent /= max(np.linalg.norm(cent), 1e-9)
            for g in members:
                assignments[g] = cid
            summary.append({"cluster_id": cid, "char": ch,
                            "members": members, "centroid": cent,
                            "size": len(members)})
    return assignments, summary


# %% 5. CALIBRATED CONFIDENCE  ────────────────────────────────────────────────
def confidence_band(top_cos, second_cos=None):
    gap = (top_cos - second_cos) if second_cos is not None else top_cos
    if top_cos >= HIGH_CONFIDENCE:
        band = "high"
    elif top_cos >= MOD_CONFIDENCE:
        band = "moderate"
    else:
        band = "low"
    if (second_cos is not None and gap < NARROW_GAP_DEMOTION
            and band == "high"):
        band = "moderate"
    if (second_cos is not None and gap < TINY_GAP_DEMOTION
            and band == "moderate"):
        band = "low-to-moderate"
    return band, round(gap, 3)


def is_leakage_suspect(top_cos, rank_of_true=None):
    if top_cos >= LEAKAGE_COSINE and (rank_of_true is None or rank_of_true == 1):
        return True
    return False


# %% 6. INIT GLYPH CORPUS  ────────────────────────────────────────────────────
print("[glyph] loading index")
index = CDTIndex(DATA_DIR, crop_size=CROP_SIZE)
print(f"[glyph] {len(index)} crops")

print("[glyph] building templates")
templates = build_templates(index, max_per_char=300)

glyph_encoder = load_or_train_encoder(index, templates)

emb_cache = RUN_DIR / "glyph_embeddings.npy"
if emb_cache.exists():
    print(f"[glyph] loading cached embeddings from {emb_cache}")
    embeddings = np.load(emb_cache)
    if embeddings.shape[0] != len(index):
        print(f"[glyph] cache mismatch; recomputing")
        embeddings = encode_all(glyph_encoder, index, templates, BATCH_SIZE)
        np.save(emb_cache, embeddings)
else:
    embeddings = encode_all(glyph_encoder, index, templates, BATCH_SIZE)
    np.save(emb_cache, embeddings)
print(f"[glyph] embeddings: {embeddings.shape}  ({ENCODER_TAG})")

chars = [r.char for r in index.records]
printer_slugs = [r.printer_slug for r in index.records]

print("[cluster] sort-piece discovery (full corpus)")
assignments, cluster_summary = cluster_by_char(embeddings, chars)
print(f"[cluster] {len(cluster_summary)} clusters, "
      f"{(assignments == '').sum()}/{len(assignments)} noise")


# %% 7. TEXT EMBEDDINGS  ──────────────────────────────────────────────────────
print(f"[text] loading {TEXT_EMBED_MODEL}")
text_encoder = SentenceTransformer(TEXT_EMBED_MODEL, device=DEVICE)
try:
    TEXT_DIM = text_encoder.get_embedding_dimension()
except AttributeError:
    TEXT_DIM = text_encoder.get_sentence_embedding_dimension()


def embed_text(texts, batch_size=32):
    if isinstance(texts, str):
        texts = [texts]
    return text_encoder.encode(texts, batch_size=batch_size,
                                normalize_embeddings=True,
                                show_progress_bar=False)


# %% 8. LITERATURE SEED CORPUS  ───────────────────────────────────────────────
LITERATURE_SEED = [
    {"source": "CDT About page", "year": 2024,
     "text": "Recent estimates suggest that the printers of more than half of "
             "all books, pamphlets, and broadsides printed in Restoration "
             "London remain unknown. The CDT is a resource to help identify "
             "unknown printers. Made of a pliant lead alloy, letterpress "
             "letters were often damaged during presswork. Since no two "
             "pieces of type degrade in precisely the same way, damaged type "
             "offers a typographical fingerprint that can reveal the "
             "identities of the individuals responsible for a book's making."},
    {"source": "CDT About page", "year": 2024,
     "text": "The CDT is limited to uppercase, non-ligature characters. The "
             "catalogue represents more than 15,000 characters extracted from "
             "more than 1900 editions printed between 1660 and 1700."},
    {"source": "Print & Probability project page", "year": 2024,
     "text": "Print & Probability is an interdisciplinary project at "
             "Carnegie Mellon University and UC San Diego that uses machine "
             "learning and computer vision tools to identify type "
             "impressions for what the team calls 'computational "
             "bibliography.'"},
    {"source": "CDT About page", "year": 2024,
     "text": "Some Restoration printers worked exclusively in collaboration "
             "with a second printer. The respective types — if they had "
             "separate type cases at all — are impossible to reliably "
             "assign or differentiate. In these cases, the names of both "
             "collaborators are given as a single entry."},
    {"source": "General bibliographic context",
     "text": "Clandestine printing in Restoration England was driven by the "
             "1662 Licensing of the Press Act and its successors. Dissenting "
             "religious works, political pamphlets, and works critical of "
             "the Crown were routinely printed with false imprints — common "
             "forms include 'Printed for the booksellers of London', "
             "'Printed in the year', or imprints attributing the work to "
             "deceased printers or fictional shops."},
    {"source": "General bibliographic context",
     "text": "After the Licensing Act lapsed in 1695, the volume of "
             "underground printing dropped sharply, but the period 1662-1695 "
             "produced thousands of editions whose true printers remain "
             "unidentified."},
    {"source": "General bibliographic context",
     "text": "Robert Roberts (active 1681-1697) was a London printer "
             "specialising in nonconformist religious works, often printing "
             "for the booksellers Thomas Cockerill and Thomas Parkhurst. "
             "Evan Tyler (active 1639-1682) was Royal Printer in Scotland "
             "but also operated presses in London; his shop's output is "
             "often confused with that of Matthew Simmons, with whom he had "
             "business overlaps in the 1640s and 1650s."},
    {"source": "General bibliographic context",
     "text": "Sort-sharing — the migration of damaged type between shops "
             "via apprenticeship, partnership, sale, or estate transfer — "
             "complicates the assumption that each printer had a sealed "
             "type case. Strong cosine similarity between two printers may "
             "indicate (a) joint operation, (b) sequential ownership of the "
             "same type, or (c) the same printer working under two names."},
    {"source": "Methodology note (empirical baseline from this corpus)",
     "text": "TF-IDF over sort-piece clusters identifies which clusters are "
             "distinctive (low document frequency) and which are common to "
             "most shops. A held-out attribution with cosine > 0.7 AND a gap "
             ">= 0.10 to the second-place candidate is strong evidence. "
             "0.4-0.7 is moderate. Below 0.4 is weak. Cosines >= 0.90 in "
             "book-level held-out evaluation are almost always leakage: "
             "other books by the same printer remained in the cluster space."},
    {"source": "Methodology note (empirical baseline from this corpus)",
     "text": "Empirical book-level evaluation on the CDT corpus: 81 books "
             "across 18 printers, honest recall@3 = 28%, lift 1.8x over "
             "chance. Median cosine for correctly-attributed books = 0.66; "
             "median for misattributed = 0.53. These distributions overlap "
             "substantially. A cosine of 0.6 has approximately 50/50 odds "
             "of being correct at book level. The method is a candidate-"
             "narrowing tool, not a point-attribution system."},
    {"source": "Methodology note",
     "text": "When evaluating a clandestine attribution, corroborating "
             "evidence to look for: (1) date overlap between the suspect "
             "shop's known activity and the book's imprint date; (2) "
             "subject matter — does the suspect shop print this kind of "
             "work? (3) bookseller — is the imprint's bookseller a known "
             "business partner of the suspect shop? (4) format and "
             "ornament — woodcut initials and headpieces also fingerprint "
             "shops, though less uniquely than damaged sorts."},
]


# %% 9. LANCEDB  ──────────────────────────────────────────────────────────────
print(f"[lancedb] connecting to {LANCE_DIR}")
db = lancedb.connect(str(LANCE_DIR))


def _existing_tables():
    try:
        return list(db.list_tables())
    except AttributeError:
        return list(db.table_names())


def _rebuild_glyphs_table():
    rows = []
    for i, r in enumerate(index.records):
        rows.append({
            "vector": embeddings[i].astype(np.float32),
            "glyph_id": r.id, "printer_slug": r.printer_slug,
            "char": r.char, "year": r.year,
            "estc": r.estc, "title": r.title,
            "publisher": r.publisher, "iiif_url": r.iiif_url,
            "cluster_id": assignments[i] if assignments[i] else "",
            "image_path": r.image_path,
        })
    schema = pa.schema([
        ("vector", pa.list_(pa.float32(), embeddings.shape[1])),
        ("glyph_id", pa.string()), ("printer_slug", pa.string()),
        ("char", pa.string()), ("year", pa.string()),
        ("estc", pa.string()), ("title", pa.string()),
        ("publisher", pa.string()), ("iiif_url", pa.string()),
        ("cluster_id", pa.string()), ("image_path", pa.string()),
    ])
    tbl = db.create_table("glyphs", data=rows, schema=schema,
                          mode="overwrite")
    print(f"[lancedb] glyphs: {len(rows)} rows")
    return tbl


def _rebuild_books_table():
    seen = {}
    for r in index.records:
        key = r.estc or r.title
        if not key or key in seen: continue
        seen[key] = r
    rows = []; texts = []
    for r in seen.values():
        text = " | ".join(filter(None, [
            r.title, f"printer: {r.printer_slug}",
            f"publisher: {r.publisher}", f"year: {r.year}",
            f"ESTC: {r.estc}",
        ]))
        texts.append(text)
        rows.append({"vector": None, "estc": r.estc, "title": r.title,
                     "year": r.year, "printer_slug": r.printer_slug,
                     "publisher": r.publisher, "search_text": text})
    if not rows: return None
    print(f"[lancedb] embedding {len(rows)} book texts")
    vecs = embed_text(texts)
    for row, v in zip(rows, vecs):
        row["vector"] = v.astype(np.float32)
    schema = pa.schema([
        ("vector", pa.list_(pa.float32(), TEXT_DIM)),
        ("estc", pa.string()), ("title", pa.string()),
        ("year", pa.string()), ("printer_slug", pa.string()),
        ("publisher", pa.string()), ("search_text", pa.string()),
    ])
    tbl = db.create_table("books", data=rows, schema=schema,
                          mode="overwrite")
    try:
        tbl.create_fts_index("search_text", replace=True)
    except Exception as e:
        print(f"[lancedb] books FTS index failed: {e}")
    print(f"[lancedb] books: {len(rows)} rows")
    return tbl


def _rebuild_literature_table():
    rows = []
    texts = [item["text"] for item in LITERATURE_SEED]
    vecs = embed_text(texts)
    for item, v in zip(LITERATURE_SEED, vecs):
        rows.append({"vector": v.astype(np.float32),
                     "source": item["source"],
                     "year": str(item.get("year", "")),
                     "text": item["text"]})
    schema = pa.schema([
        ("vector", pa.list_(pa.float32(), TEXT_DIM)),
        ("source", pa.string()), ("year", pa.string()),
        ("text", pa.string()),
    ])
    tbl = db.create_table("literature", data=rows, schema=schema,
                          mode="overwrite")
    try:
        tbl.create_fts_index("text", replace=True)
    except Exception as e:
        print(f"[lancedb] literature FTS index failed: {e}")
    print(f"[lancedb] literature: {len(rows)} rows")
    return tbl


def _ensure_tables():
    existing = _existing_tables()
    need_glyphs = "glyphs" not in existing or REBUILD_LANCEDB
    need_books  = "books"  not in existing or REBUILD_LANCEDB
    need_lit    = "literature" not in existing or REBUILD_LANCEDB

    if not need_glyphs:
        existing_t = db.open_table("glyphs")
        if existing_t.count_rows() != len(index):
            need_glyphs = True
    glyphs_tbl = (_rebuild_glyphs_table() if need_glyphs
                  else db.open_table("glyphs"))
    books_tbl  = (_rebuild_books_table() if need_books
                  else db.open_table("books"))
    lit_tbl    = (_rebuild_literature_table() if need_lit
                  else db.open_table("literature"))
    if not need_glyphs:
        print(f"[lancedb] glyphs: reusing {glyphs_tbl.count_rows()} rows")
    if not need_books:
        print(f"[lancedb] books:  reusing {books_tbl.count_rows()} rows")
    if not need_lit:
        print(f"[lancedb] lit:    reusing {lit_tbl.count_rows()} rows")
    return glyphs_tbl, books_tbl, lit_tbl


glyphs_tbl, books_tbl, literature_tbl = _ensure_tables()


# %% 10. AGENT TOOLS  ─────────────────────────────────────────────────────────
def find_similar_glyphs(glyph_id: str = None, char: str = None,
                         top_k: int = 10) -> dict:
    """Find glyphs visually similar to a given glyph (by id) or sample from
    a character class. Returns top-k nearest neighbours by cosine."""
    if glyph_id:
        idx = next((i for i, r in enumerate(index.records)
                    if r.id == glyph_id), None)
        if idx is None:
            return {"error": f"glyph_id '{glyph_id}' not found"}
        q = embeddings[idx]
        results = glyphs_tbl.search(q).limit(top_k + 1).to_list()
        results = [r for r in results if r["glyph_id"] != glyph_id][:top_k]
    elif char:
        ix = [i for i, r in enumerate(index.records) if r.char == char.upper()]
        if not ix:
            return {"error": f"no glyphs for character '{char}'"}
        q = embeddings[ix].mean(axis=0)
        q /= max(np.linalg.norm(q), 1e-9)
        results = glyphs_tbl.search(q).where(
            f"char = '{char.upper()}'", prefilter=True).limit(top_k).to_list()
    else:
        return {"error": "must specify either glyph_id or char"}

    return {"results": [{
        "glyph_id": r["glyph_id"], "printer": r["printer_slug"],
        "char": r["char"], "year": r["year"],
        "estc": r["estc"], "cluster_id": r["cluster_id"],
        "distance": float(r.get("_distance", 0.0)),
    } for r in results]}


def compare_printer_fingerprints(query_printer: str,
                                  top_k: int = 5) -> dict:
    """PRINTER-level held-out attribution. 
    Returns ranking + calibrated confidence_band (DO NOT OVERRIDE)."""
    if query_printer not in set(printer_slugs):
        return {"error": f"printer '{query_printer}' not in corpus",
                "available": sorted(set(printer_slugs))[:20]}
    keep_idxs = [i for i, s in enumerate(printer_slugs) if s != query_printer]
    h_assignments, h_summary = cluster_by_char(embeddings, chars,
                                                restrict_idxs=keep_idxs)
    if not h_summary:
        return {"error": "no clusters formed"}
    all_cids = sorted({s["cluster_id"] for s in h_summary})
    cid_to_col = {c: i for i, c in enumerate(all_cids)}
    keep_printers = sorted(s for s in set(printer_slugs) if s != query_printer)
    p2row = {p: i for i, p in enumerate(keep_printers)}
    M = np.zeros((len(keep_printers), len(all_cids)), dtype=np.float32)
    for slug, cid in zip(printer_slugs, h_assignments):
        if slug == query_printer or not cid: continue
        M[p2row[slug], cid_to_col[cid]] += 1
    df = (M > 0).sum(axis=0)
    idf = np.log((M.shape[0] + 1) / (df + 1)) + 1.0
    tf = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    tfidf = tf * idf
    fp = tfidf / np.maximum(np.linalg.norm(tfidf, axis=1, keepdims=True), 1e-9)
    centroids_by_char = defaultdict(list)
    for s in h_summary:
        centroids_by_char[s["char"]].append((s["cluster_id"], s["centroid"]))
    q_counts = np.zeros(len(all_cids), dtype=np.float32)
    for i, slug in enumerate(printer_slugs):
        if slug != query_printer: continue
        cands = centroids_by_char.get(chars[i], [])
        if not cands: continue
        best_cid, best_sim = None, -1.0
        for cid, cent in cands:
            sim = float(embeddings[i] @ cent)
            if sim > best_sim:
                best_sim, best_cid = sim, cid
        if best_cid:
            q_counts[cid_to_col[best_cid]] += 1.0
    if q_counts.sum() < 1.0:
        return {"error": "query projected to no clusters"}
    q_vec = q_counts / q_counts.sum() * idf
    q_vec /= max(np.linalg.norm(q_vec), 1e-9)
    sims = fp @ q_vec
    order = np.argsort(-sims)[:top_k]
    ranking = [{"printer": keep_printers[int(i)],
                "cosine": float(sims[int(i)])} for i in order]
    top_other = order[0]
    contribs = []
    for j, cid in enumerate(all_cids):
        c = float(q_vec[j] * fp[top_other, j])
        if c > 0:
            contribs.append({"cluster_id": cid,
                              "char": cid.split("::")[0],
                              "contribution": c})
    contribs.sort(key=lambda x: -x["contribution"])
    top_cos = ranking[0]["cosine"]
    second_cos = ranking[1]["cosine"] if len(ranking) > 1 else None
    band, gap = confidence_band(top_cos, second_cos)
    return {
        "query_printer": query_printer,
        "ranking": ranking,
        "top_shared_clusters": contribs[:8],
        "n_clusters_built": len(h_summary),
        "n_query_glyphs": int(q_counts.sum()),
        "confidence_band": band, "gap_to_second": gap,
        "confidence_rubric": (
            "cosine>=0.70 with gap>=0.10 = high; 0.40-0.70 = moderate; "
            "<0.40 = low. Narrow gap (<0.10) demotes one band. "
            "AGENT MUST NOT OVERRIDE THIS BAND."
        ),
    }


def audit_cluster(cluster_id: str) -> dict:
    """Check whether a cluster is shared-sort evidence (multiple
    printers, no single printer dominating > 85%).
    Cluster IDs have the form '<CHAR>::<NNNN>' e.g. 'C::0000', 'A::0001'.
    """
    members = [(i, index.records[i]) for i, a in enumerate(assignments)
               if a == cluster_id]
    if not members:
        # Helpful error: show a sample of real cluster IDs so the agent
        # can self-correct instead of guessing more fake IDs.
        all_cids = sorted({a for a in assignments if a})
        sample = all_cids[:12] + ["..."] + all_cids[-4:] if len(all_cids) > 16 else all_cids
        return {
            "cluster_id": cluster_id,
            "error": (f"cluster '{cluster_id}' not found. Cluster IDs have "
                      f"the form '<CHAR>::<NNNN>' (e.g. 'C::0000'). "
                      f"Use the exact cluster_id strings returned by "
                      f"attribute_book or compare_printer_fingerprints."),
            "example_valid_ids": sample,
            "n_total_clusters": len(all_cids),
        }
    by_printer = Counter(r.printer_slug for _, r in members)
    dominant_p, dominant_n = by_printer.most_common(1)[0]
    purity = dominant_n / len(members)
    is_single = len(by_printer) == 1
    is_genuine = (not is_single) and (purity < 0.85)
    if is_single:
        verdict = ("single-printer cluster — NOT evidence of sort-sharing, "
                   "just intra-shop variation")
    elif purity > 0.85:
        verdict = (f"dominated by {dominant_p} ({purity:.0%}) — weak "
                   f"evidence of sharing")
    else:
        verdict = (f"shared across {len(by_printer)} printers — GENUINE "
                   f"evidence of sort-sharing")
    return {
        "cluster_id": cluster_id, "size": len(members),
        "n_printers": len(by_printer),
        "dominant_printer": dominant_p,
        "dominant_share": round(purity, 3),
        "printer_distribution": dict(by_printer),
        "is_genuine_shared_sort": is_genuine, "verdict": verdict,
    }


_BOOKS_PER_PRINTER_CACHE = None


def _books_per_printer():
    """Cached count of distinct books per printer (by ESTC).

    Used by attribution functions to detect 'cold-start' cases: books
    whose catalogued printer has no OTHER examples in the corpus. For
    such books, leave-one-book-out evaluation removes the printer from
    the candidate fingerprint space entirely, making attribution
    structurally impossible. We flag this loudly so the agent doesn't
    confidently misattribute.
    """
    global _BOOKS_PER_PRINTER_CACHE
    if _BOOKS_PER_PRINTER_CACHE is None:
        estc_to_printer = {}
        for r in index.records:
            if r.estc:
                estc_to_printer[r.estc] = r.printer_slug
        cnt = Counter()
        for estc, slug in estc_to_printer.items():
            cnt[slug] += 1
        _BOOKS_PER_PRINTER_CACHE = dict(cnt)
    return _BOOKS_PER_PRINTER_CACHE


def _is_cold_start(catalogued_printer):
    """True if the catalogued printer has only this book in the corpus.

    Cold-start books cannot be attributed by our pipeline: the
    held-out evaluation strips all examples of the printer from the
    cluster fingerprint space, leaving the system to pick the most
    similar OTHER printer rather than identifying the true one.
    """
    if not catalogued_printer:
        return False
    return _books_per_printer().get(catalogued_printer, 0) <= 1


COLD_START_WARNING_TEXT = (
    "COLD-START CASE: the catalogued printer has only this single book "
    "in the corpus. Held-out attribution removes them entirely from the "
    "candidate fingerprint space. The ranking below is the system's best "
    "guess among OTHER printers and is structurally unable to identify "
    "the true printer. Treat as exploratory, NOT definitive."
    "answer is: this method cannot attribute this book reliably."
)


def attribute_book(estc_id: str, top_k: int = 5) -> dict:
    """BOOK-level held-out attribution. The realistic clandestine task.
    Returns ranking + leakage_suspect +
    confidence_band."""
    book_glyphs = [i for i, r in enumerate(index.records) if r.estc == estc_id]
    if not book_glyphs:
        return {"error": f"no glyphs for ESTC {estc_id}"}
    book_recs = [index.records[i] for i in book_glyphs]
    true_printer = book_recs[0].printer_slug

    keep_idxs = [i for i in range(len(index)) if i not in set(book_glyphs)]
    h_assignments, h_summary = cluster_by_char(embeddings, chars,
                                                restrict_idxs=keep_idxs)
    if not h_summary:
        return {"error": "no clusters formed"}
    all_cids = sorted({s["cluster_id"] for s in h_summary})
    cid_to_col = {c: i for i, c in enumerate(all_cids)}
    keep_printers = sorted(set(printer_slugs[i] for i in keep_idxs))
    p2row = {p: i for i, p in enumerate(keep_printers)}
    M = np.zeros((len(keep_printers), len(all_cids)), dtype=np.float32)
    for i in keep_idxs:
        slug = printer_slugs[i]; cid = h_assignments[i]
        if cid and cid in cid_to_col:
            M[p2row[slug], cid_to_col[cid]] += 1
    df = (M > 0).sum(axis=0)
    idf = np.log((M.shape[0] + 1) / (df + 1)) + 1.0
    tf = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    tfidf = tf * idf
    fp = tfidf / np.maximum(np.linalg.norm(tfidf, axis=1, keepdims=True), 1e-9)
    centroids_by_char = defaultdict(list)
    for s in h_summary:
        centroids_by_char[s["char"]].append((s["cluster_id"], s["centroid"]))
    q_counts = np.zeros(len(all_cids), dtype=np.float32)
    n_projected = 0
    for i in book_glyphs:
        cands = centroids_by_char.get(chars[i], [])
        if not cands: continue
        best_cid, best_sim = None, -1.0
        for cid, cent in cands:
            sim = float(embeddings[i] @ cent)
            if sim > best_sim:
                best_sim, best_cid = sim, cid
        if best_cid:
            q_counts[cid_to_col[best_cid]] += 1.0
            n_projected += 1
    if q_counts.sum() < 1.0:
        return {"error": "book projected to no clusters"}
    q_vec = q_counts / q_counts.sum() * idf
    q_vec /= max(np.linalg.norm(q_vec), 1e-9)
    sims = fp @ q_vec
    order = np.argsort(-sims)
    ranking = [{"printer": keep_printers[int(i)],
                "cosine": float(sims[int(i)])} for i in order[:top_k]]
    rank_of_true = None
    if true_printer in p2row:
        for k_idx, idx_i in enumerate(order):
            if keep_printers[int(idx_i)] == true_printer:
                rank_of_true = k_idx + 1; break
    top_other = order[0]
    cluster_contribs = []
    for j, cid in enumerate(all_cids):
        c = float(q_vec[j] * fp[top_other, j])
        if c > 0:
            cluster_contribs.append({"cluster_id": cid,
                                      "char": cid.split("::")[0],
                                      "contribution": c})
    cluster_contribs.sort(key=lambda x: -x["contribution"])
    top_cos = ranking[0]["cosine"]
    second_cos = ranking[1]["cosine"] if len(ranking) > 1 else None
    band, gap = confidence_band(top_cos, second_cos)
    leaked = is_leakage_suspect(top_cos, rank_of_true)
    cold_start = _is_cold_start(true_printer)

    # Reorder so action-relevant fields come first. Long descriptive prose
    # last so it doesn't push the IDs out of the agent's context window.
    return {
        # === Action-relevant header — the agent reads these first ===
        "estc": estc_id,
        # Cold-start flag at the very top: agents see this immediately and
        # should refuse to give a confident attribution when set.
        "cold_start_warning": cold_start,
        "cold_start_message": COLD_START_WARNING_TEXT if cold_start else "",
        "ranking": ranking,
        "confidence_band": "cold_start" if cold_start else band,
        "gap_to_second": gap,
        "leakage_suspect": leaked,
        "leakage_warning": (
            f"cosine >= {LEAKAGE_COSINE} — likely other books by same "
            f"printer remained in cluster space; attribution may be "
            f"inflated" if leaked else ""
        ),
        # Flat list of cluster IDs — easiest possible thing for the agent
        # to copy verbatim into audit_cluster calls. Use these EXACT strings.
        "cluster_ids_to_audit": [c["cluster_id"]
                                  for c in cluster_contribs[:5]],
        "top_shared_clusters": cluster_contribs[:8],
        # === Metadata follows ===
        "catalogued_printer": true_printer,
        "rank_of_catalogued_printer": rank_of_true,
        "n_glyphs": len(book_glyphs),
        "n_projected": n_projected,
        "book_info": {
            "title": book_recs[0].title[:80],   # short — agent doesn't need full
            "year": book_recs[0].year,
            "publisher": book_recs[0].publisher[:80] if book_recs[0].publisher else "",
        },
    }


# %% 9c. MULTI-EVIDENCE BAYESIAN ATTRIBUTION  ─────────────────────────────────
# Module A: extend cosine-only attribution with multiple evidence streams
# (vision, imprint, temporal, content) combined via likelihood ratios into
# calibrated posterior probabilities over candidate printers.
#
# Each stream produces an LR per candidate printer:
#   LR_E(P) = P(observed evidence E | true printer = P)
#            / P(observed evidence E | true printer != P)
# We combine streams assuming approximate conditional independence:
#   log-posterior(P) = log-prior(P) + sum_E log-LR_E(P)
# then softmax to get a posterior distribution over printers.
#
# This is the standard hybrid-evidence pattern from author disambiguation
# (LEAD 2025, WhoIs 2022) applied to historical printer attribution for
# the first time we are aware of. The vision stream is calibrated from
# the empirical distribution of cosines under same-printer vs
# different-printer pairs — turning our cosines into actual probabilities.

# ── Calibration: build empirical P(cosine | same/diff printer) ─────────
_VISION_CALIBRATION_CACHE = None


def _build_vision_calibration():
    """Compute empirical histograms of cosine scores under the two
    hypotheses: H_same (the candidate is the true printer) and H_diff
    (the candidate is not the true printer).

    We use book-level held-out attribution data: for each evaluable book,
    the cosine to its TRUE printer is a sample from H_same; the cosines
    to all OTHER printers are samples from H_diff.

    Returned object provides a method `lr(cosine)` returning the
    likelihood ratio P(cos|H_same) / P(cos|H_diff). The LR is what we
    actually need for Bayesian combination.
    """
    global _VISION_CALIBRATION_CACHE
    if _VISION_CALIBRATION_CACHE is not None:
        return _VISION_CALIBRATION_CACHE

    cache_path = RUN_DIR / f"vision_calibration_{ENCODER_TAG}.json"
    if cache_path.exists():
        d = json.loads(cache_path.read_text())
        same_p = np.array(d["same_p"])
        diff_p = np.array(d["diff_p"])
        bin_edges = np.array(d["bin_edges"])

        def _bin_idx(cos):
            cos_clip = max(bin_edges[0], min(bin_edges[-1] - 1e-9, float(cos)))
            return max(0, min(len(same_p) - 1,
                              int(np.searchsorted(bin_edges, cos_clip,
                                                    side="right") - 1)))

        def lr(cos):
            i = _bin_idx(cos)
            return float(same_p[i] / diff_p[i])

        d["lr"] = lr
        _VISION_CALIBRATION_CACHE = d
        print(f"[vision-calib] loaded cached calibration from "
              f"{cache_path.name}")
        return _VISION_CALIBRATION_CACHE

    print("[vision-calib] computing P(cosine | same/diff printer) "
          "from leave-one-out evaluations (slow first time)...")
    same_cosines = []
    diff_cosines = []

    estc_ids = sorted({r.estc for r in index.records if r.estc})
    for estc in estc_ids:
        result = attribute_book(estc, top_k=100)
        if "error" in result:
            continue
        true_printer = result.get("catalogued_printer", "")
        for entry in result.get("ranking", []):
            cos = float(entry["cosine"])
            if entry["printer"] == true_printer:
                same_cosines.append(cos)
            else:
                diff_cosines.append(cos)

    # Build smoothed histograms with shared bin edges.
    # Add-1 smoothing avoids zero probabilities at the tails.
    bin_edges = np.linspace(-0.05, 1.05, 23)  # 22 bins of width 0.05
    same_hist, _ = np.histogram(np.array(same_cosines), bins=bin_edges)
    diff_hist, _ = np.histogram(np.array(diff_cosines), bins=bin_edges)
    same_p = (same_hist + 1) / (same_hist.sum() + len(same_hist))
    diff_p = (diff_hist + 1) / (diff_hist.sum() + len(diff_hist))

    def _bin_idx(cos):
        cos_clip = max(bin_edges[0], min(bin_edges[-1] - 1e-9, float(cos)))
        return max(0, min(len(same_p) - 1,
                          int(np.searchsorted(bin_edges, cos_clip,
                                                side="right") - 1)))

    def lr(cos):
        i = _bin_idx(cos)
        return float(same_p[i] / diff_p[i])

    calib_data = {
        "n_same": len(same_cosines),
        "n_diff": len(diff_cosines),
        "same_p": same_p.tolist(),
        "diff_p": diff_p.tolist(),
        "bin_edges": bin_edges.tolist(),
        "encoder": ENCODER_TAG,
    }
    cache_path.write_text(json.dumps(calib_data, indent=2))
    calib_data["lr"] = lr
    _VISION_CALIBRATION_CACHE = calib_data
    print(f"[vision-calib] {len(same_cosines)} same-printer samples, "
          f"{len(diff_cosines)} different-printer samples")
    print(f"[vision-calib] sample LR check: "
          f"cos=0.3 → LR={lr(0.3):.2f}, "
          f"cos=0.6 → LR={lr(0.6):.2f}, "
          f"cos=0.9 → LR={lr(0.9):.2f}")
    print(f"[vision-calib] saved: {cache_path}")
    return _VISION_CALIBRATION_CACHE


# ── Imprint parsing: extract printer signatures from publisher text ────
# Many Restoration imprints explicitly name the printer in coded forms:
#   "Printed by R. Roberts for Tho. Cockerill"
#   "Excudebat E. Cotes pro J. Stafford"
#   "Imprinted by J.M. for R. Royston"

def _build_imprint_signatures():
    """Build regex patterns from printer slugs that could appear in
    imprint text. Each printer's signature includes:
      - Full first + last name ("Robert Roberts")
      - First-initial last name ("R. Roberts")
      - Initials-only ("R.R.")
      - Last name alone (only for distinctive last names)
    """
    signatures = {}
    for slug in set(printer_slugs):
        parts = slug.split("_")
        if len(parts) != 2:
            continue
        last, first = parts[0].title(), parts[1].title()
        last_init = last[0]; first_init = first[0]

        patterns = [
            re.compile(rf"\b{first}\s+{last}\b", re.IGNORECASE),
            re.compile(rf"\b{last},?\s+{first}\b", re.IGNORECASE),
            re.compile(rf"\b{first_init}\.\s*{last}\b", re.IGNORECASE),
            re.compile(rf"\bExcudebat\s+{first_init}\.?\s*{last}\b",
                       re.IGNORECASE),
            re.compile(rf"\b{first_init}\.\s*{last_init}\.\b"),
        ]
        # Last-name-only is risky; only include for unambiguous surnames.
        distinct_printers_with_this_lastname = {
            s for s in set(printer_slugs)
            if s.split("_") and s.split("_")[0] == parts[0]
        }
        if len(distinct_printers_with_this_lastname) == 1:
            patterns.append(re.compile(rf"\b{last}\b", re.IGNORECASE))
        signatures[slug] = patterns
    return signatures


_IMPRINT_SIGS_CACHE = None


def _get_imprint_signatures():
    global _IMPRINT_SIGS_CACHE
    if _IMPRINT_SIGS_CACHE is None:
        _IMPRINT_SIGS_CACHE = _build_imprint_signatures()
    return _IMPRINT_SIGS_CACHE


def _imprint_lr(publisher_text, candidate_slug):
    """Likelihood ratio for the imprint stream.
    - candidate's name found: LR = 10 (supportive)
    - a DIFFERENT printer's name found: LR = 0.3 (evidence against)
    - no printer name detected: LR = 1.0 (no information)
    """
    if not publisher_text:
        return 1.0
    sigs = _get_imprint_signatures()
    matched = set()
    for slug, patterns in sigs.items():
        for p in patterns:
            if p.search(publisher_text):
                matched.add(slug)
                break
    if not matched:
        return 1.0
    if candidate_slug in matched:
        return 10.0
    else:
        return 0.3


# ── Temporal LR: does the book's year fall within the printer's active span ──
_PRINTER_DATE_RANGES = None


def _get_printer_date_ranges():
    global _PRINTER_DATE_RANGES
    if _PRINTER_DATE_RANGES is not None:
        return _PRINTER_DATE_RANGES
    ranges = defaultdict(list)
    for r in index.records:
        if r.year and r.year.isdigit():
            ranges[r.printer_slug].append(int(r.year))
    _PRINTER_DATE_RANGES = {
        slug: (min(years), max(years))
        for slug, years in ranges.items() if years
    }
    return _PRINTER_DATE_RANGES


def _temporal_lr(book_year, candidate_slug):
    """LR for temporal compatibility."""
    if not book_year or not str(book_year).isdigit():
        return 1.0
    yr = int(book_year)
    ranges = _get_printer_date_ranges()
    if candidate_slug not in ranges:
        return 1.0
    lo, hi = ranges[candidate_slug]
    if lo <= yr <= hi:
        return 3.0
    elif lo - 5 <= yr <= hi + 5:
        return 1.0
    else:
        return 0.1


# ── Bookseller LR: known printer↔bookseller partnerships ──
KNOWN_BOOKSELLER_PARTNERSHIPS = {
    "roberts_robert": ["cockerill", "chiswell"],
    "macock_john": ["royston"],
    "cotes_ellen": ["seile", "stafford"],
    "hayes_john": ["thomson"],
    "newcomb_thomas": ["basset"],
    "streater_john": ["pawlet"],
}


def _bookseller_lr(publisher_text, candidate_slug):
    """LR for bookseller-partnership signal."""
    if not publisher_text:
        return 1.0
    text_low = publisher_text.lower()
    partners = KNOWN_BOOKSELLER_PARTNERSHIPS.get(candidate_slug, [])
    if any(p in text_low for p in partners):
        return 4.0
    for other_slug, other_partners in KNOWN_BOOKSELLER_PARTNERSHIPS.items():
        if other_slug == candidate_slug:
            continue
        if any(p in text_low for p in other_partners):
            return 0.5
    return 1.0


# ── Main multi-evidence attribution tool ───────────────────────────────
def attribute_book_multi_evidence(estc_id: str, top_k: int = 5,
                                    use_streams: list = None) -> dict:
    """Multi-evidence Bayesian attribution.

    Combines four independent evidence streams via likelihood ratios:
      1. vision       — calibrated from cosine via empirical histograms
      2. imprint      — printer-name signatures in the publisher text
      3. temporal     — does the book's year fall in the printer's range
      4. bookseller   — known printer and bookseller partnerships

    Returns posterior probabilities over candidate printers, plus a
    per-stream breakdown showing what each stream contributed.

    use_streams: list of stream names to include (default: all).
                 Useful for ablations.
    """
    all_streams = ["vision", "imprint", "temporal", "bookseller"]
    use_streams = use_streams or all_streams
    invalid = set(use_streams) - set(all_streams)
    if invalid:
        return {"error": f"unknown streams: {invalid}; "
                f"valid options: {all_streams}"}

    vision_result = attribute_book(estc_id, top_k=100)
    if "error" in vision_result:
        return vision_result

    book_info = vision_result.get("book_info", {})
    book_year = book_info.get("year", "")
    publisher = book_info.get("publisher", "")
    true_printer = vision_result.get("catalogued_printer", "")

    cos_by_printer = {entry["printer"]: float(entry["cosine"])
                      for entry in vision_result.get("ranking", [])}

    calib = _build_vision_calibration() if "vision" in use_streams else None

    candidates = sorted(cos_by_printer.keys())
    log_lrs = {p: {} for p in candidates}
    for p in candidates:
        if "vision" in use_streams:
            lr = calib["lr"](cos_by_printer[p])
            log_lrs[p]["vision"] = math.log(max(lr, 1e-9))
        if "imprint" in use_streams:
            lr = _imprint_lr(publisher, p)
            log_lrs[p]["imprint"] = math.log(max(lr, 1e-9))
        if "temporal" in use_streams:
            lr = _temporal_lr(book_year, p)
            log_lrs[p]["temporal"] = math.log(max(lr, 1e-9))
        if "bookseller" in use_streams:
            lr = _bookseller_lr(publisher, p)
            log_lrs[p]["bookseller"] = math.log(max(lr, 1e-9))

    log_post = {p: sum(log_lrs[p].values()) for p in candidates}
    max_lp = max(log_post.values()) if log_post else 0.0
    exp_logp = {p: math.exp(log_post[p] - max_lp) for p in candidates}
    z = sum(exp_logp.values())
    posterior = {p: v / z for p, v in exp_logp.items()}

    ranked = sorted(candidates, key=lambda p: -posterior[p])
    top_candidates = []
    for p in ranked[:top_k]:
        top_candidates.append({
            "printer": p,
            "posterior_prob": round(posterior[p], 4),
            "log_posterior": round(log_post[p], 3),
            "log_LR_breakdown": {s: round(v, 3)
                                  for s, v in log_lrs[p].items()},
            "vision_cosine": round(cos_by_printer.get(p, 0.0), 3),
        })

    rank_of_true = None
    if true_printer in candidates:
        rank_of_true = ranked.index(true_printer) + 1

    entropy = -sum(p * math.log(p + 1e-12) for p in posterior.values())
    max_entropy = math.log(len(candidates)) if len(candidates) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    top_prob = posterior[ranked[0]] if ranked else 0.0
    if top_prob >= 0.70:
        band = "high"
    elif top_prob >= 0.40:
        band = "moderate"
    elif top_prob >= 0.20:
        band = "low-to-moderate"
    else:
        band = "low"

    cold_start = _is_cold_start(true_printer)

    return {
        "estc": estc_id,
        "method": "multi_evidence_bayesian",
        "cold_start_warning": cold_start,
        "cold_start_message": COLD_START_WARNING_TEXT if cold_start else "",
        "streams_used": use_streams,
        "ranking": top_candidates,
        "top_posterior_probability": round(top_prob, 4),
        "confidence_band": "cold_start" if cold_start else band,
        "normalized_entropy": round(normalized_entropy, 3),
        "interpretation_notes": (
            f"Posterior P(top) = {top_prob:.1%}. "
            f"Lower normalized_entropy = more decisive posterior. "
            f"The log_LR_breakdown shows which streams contributed."
        ),
        "catalogued_printer": true_printer,
        "rank_of_catalogued_printer_by_posterior": rank_of_true,
        "vision_leakage_suspect": vision_result.get("leakage_suspect", False),
        "book_info": book_info,
    }


# ─────────────────────────────────────────────────────────────────────────
# MODULE B: ACTIVE EVIDENCE ACQUISITION
# ─────────────────────────────────────────────────────────────────────────
#
# Instead of running ALL streams up front, the active variant
# decides which stream to run NEXT to maximally reduce uncertainty about
# the printer. This matches how human bibliographers work: start with the
# cheapest evidence (imprint), drill into typographic comparison only if
# the cheap evidence doesn't resolve.
#
# The decision rule is Expected Information Gain per unit cost:
#   IG(stream) = H(posterior_now) - E[H(posterior_after_stream)]
# where the expectation is over the stream's possible outputs, weighted by
# the current posterior over printers.
#
# Each stream has a cost (vision is expensive and imprint is cheap) and a
# generative model: "if printer X is the true answer, what LR is this
# stream likely to return?" These models are simplifying but follow
# standard active-inference practice.

# Cost per stream (relative units; vision is the most expensive)
_STREAM_COSTS = {
    "imprint":    1.0,
    "temporal":   1.0,
    "bookseller": 1.0,
    "vision":     10.0,
}

# Generative model parameters. For each stream and each "printer
# scenario," we specify a probability distribution over the LR the stream
# would return. We discretize LR into bins: [strong_against, weak_against,
# neutral, weak_for, strong_for] with representative values.
_LR_BINS = [0.1, 0.5, 1.0, 3.0, 10.0]
_LR_BIN_LABELS = ["strong_against", "weak_against", "neutral",
                  "weak_for", "strong_for"]


def _generative_pmf_for_stream(stream_name, is_true_printer):
    """Return a probability distribution over LR bins, given whether the
    candidate is the true printer.

    These distributions encode our prior belief about how each stream
    behaves. They are simplifying but plausible. Calibrating them from
    training data would be a follow-up improvement."""
    if stream_name == "imprint":
        if is_true_printer:
            # If the candidate IS the true printer, the imprint mentions
            # them ~50% of the time (the rest being silent or coded
            # because the book is clandestine). Strong_for if mentioned;
            # neutral otherwise.
            return [0.0, 0.05, 0.45, 0.0, 0.50]
        else:
            # If NOT the true printer, the imprint rarely mentions them.
            # Most of the time the candidate's name isn't there.
            return [0.10, 0.05, 0.80, 0.03, 0.02]
    if stream_name == "temporal":
        if is_true_printer:
            # If true, the book's year is in the candidate's range with
            # high probability (≥0.9): LR=3.
            return [0.05, 0.05, 0.0, 0.0, 0.90]  # 0.90 prob LR>=1; treat as "strong"
        else:
            # If NOT the true printer, the book year may or may not fall
            # in the candidate's range depending on overlap.
            return [0.10, 0.0, 0.40, 0.0, 0.50]
    if stream_name == "bookseller":
        if is_true_printer:
            # Sometimes the candidate's known bookseller partner appears.
            return [0.0, 0.0, 0.60, 0.0, 0.40]
        else:
            return [0.10, 0.0, 0.85, 0.0, 0.05]
    if stream_name == "vision":
        if is_true_printer:
            # Vision is strongest when the printer is correct.
            return [0.0, 0.0, 0.05, 0.20, 0.75]
        else:
            return [0.40, 0.30, 0.20, 0.08, 0.02]
    # Fallback: uniform
    return [0.2] * 5


def _expected_information_gain(stream_name, current_posterior, candidates):
    """Compute the expected reduction in posterior entropy if we run this
    stream. Marginalizes over the stream's possible LR outputs weighted by
    the current posterior over which printer is true.

    Returns EIG (in nats). Higher = more informative.
    """
    if not candidates:
        return 0.0

    current_entropy = -sum(p * math.log(p + 1e-12)
                           for p in current_posterior.values())

    # For each possible LR output bin, compute P(that bin) under the
    # current posterior, and the resulting posterior if we observed it.
    expected_post_entropy = 0.0
    for bin_idx, lr in enumerate(_LR_BINS):
        # P(observe this LR for the candidate set, marginalized over
        # which printer is true)
        # Simplifying: we compute one EIG per candidate by treating the
        # stream as outputting LRs independently for each candidate. To
        # keep this tractable, we compute the average per-candidate EIG.
        pass

    # Tractable simplification: compute the per-candidate average EIG.
    # For each candidate c, treat "is c the true printer" as a binary
    # question, and ask how much running this stream would update P(c).
    total_eig = 0.0
    for c in candidates:
        p_c = current_posterior.get(c, 1.0 / len(candidates))
        if p_c <= 1e-9 or p_c >= 1 - 1e-9:
            continue  # already decided; running stream won't help
        # Pmf if c IS true
        pmf_true = _generative_pmf_for_stream(stream_name, True)
        # Pmf if c is NOT true (averaged over the other candidates)
        pmf_false = _generative_pmf_for_stream(stream_name, False)

        # Marginal P(LR bin)
        marginal = [p_c * pt + (1 - p_c) * pf
                    for pt, pf in zip(pmf_true, pmf_false)]
        # Posterior P(c | LR bin) via Bayes
        eig_c = 0.0
        for i, m in enumerate(marginal):
            if m <= 1e-9: continue
            p_c_given_lr = p_c * pmf_true[i] / m
            # Information gain for this candidate: H(p_c) - H(p_c | lr)
            h_before = -(p_c * math.log(p_c + 1e-12) +
                          (1 - p_c) * math.log(1 - p_c + 1e-12))
            h_after  = -(p_c_given_lr * math.log(p_c_given_lr + 1e-12) +
                          (1 - p_c_given_lr) *
                          math.log(1 - p_c_given_lr + 1e-12))
            eig_c += m * (h_before - h_after)
        total_eig += p_c * eig_c
    return total_eig


def _mask_printer_from_publisher(publisher_text, true_printer_slug,
                                   strict=False):
    """Strip mentions of the true printer from the publisher field to
    simulate clandestine attribution conditions. Returns the sanitized
    publisher string.

    Modes:
      - default: strip the printer's NAME patterns only. Leaves bookseller
        names intact. (The default --clandestine mode.)
      - strict=True: also strip the printer's known bookseller partners,
        because those would leak the answer through the bookseller stream.
        Use this for an test where NO bibliographic feature that
        correlates with the remaining printer.

    The default mode answers: "what if the imprint is silent about the
    printer but the bookseller is still listed?" -- a common real-world
    case for clandestine books.

    The strict mode answers: "what if NO bibliographic identification
    is possible, only the typographic evidence and date?" -- another
    real-world case, where we have anonymous output with no associates."""
    if not publisher_text or not true_printer_slug:
        return publisher_text
    sigs = _get_imprint_signatures()
    patterns = sigs.get(true_printer_slug, [])
    masked = publisher_text
    for p in patterns:
        masked = p.sub("[imprint suppressed]", masked)
    if strict:
        # Also strip any known booksellers associated with the true
        # printer, since these leak the answer through the bookseller
        # stream's regex.
        partners = KNOWN_BOOKSELLER_PARTNERSHIPS.get(true_printer_slug, [])
        for partner in partners:
            masked = re.sub(rf"\b{re.escape(partner)}\b",
                             "[partner suppressed]", masked,
                             flags=re.IGNORECASE)
    return masked


def active_attribute_book(estc_id: str, top_k: int = 5,
                           stop_threshold: float = 0.80,
                           max_streams: int = 4,
                           clandestine: bool = False,
                           strict_clandestine: bool = False,
                           strictest_clandestine: bool = False,
                           verbose: bool = False) -> dict:
    """Active-inference attribution. Iteratively picks the most
    informative evidence stream given the current posterior, runs it,
    updates the posterior, and stops when P(top printer) >= stop_threshold
    OR all streams have been used.

    Parameters:
      clandestine: If True, the true printer's name is masked from the
        publisher field before any text-based stream runs. This simulates
        a clandestine attribution where the imprint is suppressed or
        false. Without this, evaluating on the CDT corpus is trivial
        because the publisher field usually names the true printer
        explicitly.

    Returns a record with:
      - acquisition_path: which streams ran, in what order, and what they
        contributed (the audit trail)
      - posterior: final probabilities
      - savings_estimate: total cost saved vs running all streams
    """

    all_candidates = sorted(set(printer_slugs))
    if not all_candidates:
        return {"error": "no candidates in corpus"}

    book_meta = lookup_estc(estc_id)
    if "error" in book_meta:
        return book_meta
    publisher = book_meta.get("publisher", "")
    book_year = book_meta.get("year", "")
    true_printer = book_meta.get("catalogued_printer", "")

    # In clandestine mode, we mask the true printer's name from the publisher
    # field. This is essential for evaluating clandestine-attribution
    # methods on a non-clandestine corpus like CDT.
    # In strict_clandestine mode, also mask known booksellers (which
    # otherwise leak the answer through the bookseller stream).
    # In strictest_clandestine mode, ALSO suppress the year (enabling
    # temporal LR=1.0 for all candidates). This simulates books with a
    # false imprint date -- the genuinely worst-case clandestine scenario.
    if strictest_clandestine:
        # mask printer name AND booksellers
        if true_printer:
            publisher = _mask_printer_from_publisher(
                publisher, true_printer, strict=True)
        # Suppress the year so temporal stream returns LR=1.0 always
        book_year = ""
    elif (clandestine or strict_clandestine) and true_printer:
        publisher = _mask_printer_from_publisher(
            publisher, true_printer,
            strict=strict_clandestine)

    # Uniform prior
    n = len(all_candidates)
    log_posterior = {p: 0.0 for p in all_candidates}
    posterior = {p: 1.0 / n for p in all_candidates}

    acquired_streams = []
    cost_spent = 0.0
    log_lr_breakdown = {p: {} for p in all_candidates}

    available = ["imprint", "temporal", "bookseller", "vision"]

    for step in range(max_streams):
        # Check stopping condition
        top_p = max(posterior.values())
        if top_p >= stop_threshold:
            if verbose:
                print(f"  [active] stop: top posterior {top_p:.3f} "
                      f">= threshold {stop_threshold}")
            break

        if not available:
            if verbose:
                print(f"  [active] stop: all streams exhausted")
            break

        # Compute EIG-per-cost for each remaining stream
        eig_by_stream = {}
        for s in available:
            eig = _expected_information_gain(s, posterior, all_candidates)
            cost = _STREAM_COSTS.get(s, 1.0)
            eig_by_stream[s] = eig / cost

        # Pick the best stream
        chosen = max(eig_by_stream, key=eig_by_stream.get)
        chosen_eig = eig_by_stream[chosen] * _STREAM_COSTS[chosen]

        if verbose:
            print(f"  [active step {step + 1}] EIG: " +
                  ", ".join(f"{s}={e * _STREAM_COSTS[s]:.3f}"
                             for s, e in eig_by_stream.items()))
            print(f"  [active step {step + 1}] choosing {chosen} "
                  f"(EIG={chosen_eig:.3f}, cost={_STREAM_COSTS[chosen]})")

        # Run the chosen stream and update
        if chosen == "vision":
            v = attribute_book(estc_id, top_k=100)
            if "error" in v:
                available.remove(chosen)
                continue
            cos_by_printer = {e["printer"]: float(e["cosine"])
                              for e in v.get("ranking", [])}
            calib = _build_vision_calibration()
            for p in all_candidates:
                cos = cos_by_printer.get(p, 0.0)
                lr = calib["lr"](cos)
                log_posterior[p] += math.log(max(lr, 1e-9))
                log_lr_breakdown[p]["vision"] = round(
                    math.log(max(lr, 1e-9)), 3)
        elif chosen == "imprint":
            for p in all_candidates:
                lr = _imprint_lr(publisher, p)
                log_posterior[p] += math.log(max(lr, 1e-9))
                log_lr_breakdown[p]["imprint"] = round(
                    math.log(max(lr, 1e-9)), 3)
        elif chosen == "temporal":
            for p in all_candidates:
                lr = _temporal_lr(book_year, p)
                log_posterior[p] += math.log(max(lr, 1e-9))
                log_lr_breakdown[p]["temporal"] = round(
                    math.log(max(lr, 1e-9)), 3)
        elif chosen == "bookseller":
            for p in all_candidates:
                lr = _bookseller_lr(publisher, p)
                log_posterior[p] += math.log(max(lr, 1e-9))
                log_lr_breakdown[p]["bookseller"] = round(
                    math.log(max(lr, 1e-9)), 3)

        cost_spent += _STREAM_COSTS[chosen]
        acquired_streams.append({
            "step": step + 1,
            "stream": chosen,
            "cost": _STREAM_COSTS[chosen],
            "expected_information_gain": round(chosen_eig, 3),
        })
        available.remove(chosen)

        # Renormalize to get posterior probabilities
        max_lp = max(log_posterior.values())
        exp_logp = {p: math.exp(log_posterior[p] - max_lp)
                    for p in all_candidates}
        z = sum(exp_logp.values())
        posterior = {p: v / z for p, v in exp_logp.items()}

        if verbose:
            top3 = sorted(posterior.items(), key=lambda x: -x[1])[:3]
            print(f"  [active step {step + 1}] top3 posterior: " +
                  ", ".join(f"{p}={v:.2%}" for p, v in top3))

    # Final summary
    ranked = sorted(all_candidates, key=lambda p: -posterior[p])
    top_candidates = []
    for p in ranked[:top_k]:
        top_candidates.append({
            "printer": p,
            "posterior_prob": round(posterior[p], 4),
            "log_LR_breakdown": log_lr_breakdown[p],
        })

    rank_of_true = None
    if true_printer in ranked:
        rank_of_true = ranked.index(true_printer) + 1

    top_prob = posterior[ranked[0]] if ranked else 0.0
    if top_prob >= 0.70:
        band = "high"
    elif top_prob >= 0.40:
        band = "moderate"
    elif top_prob >= 0.20:
        band = "low-to-moderate"
    else:
        band = "low"

    # Total cost if we'd run all streams
    cost_all = sum(_STREAM_COSTS[s] for s in
                   ["imprint", "temporal", "bookseller", "vision"])

    cold_start = _is_cold_start(true_printer)

    return {
        "estc": estc_id,
        "method": "active_evidence_acquisition",
        "cold_start_warning": cold_start,
        "cold_start_message": COLD_START_WARNING_TEXT if cold_start else "",
        "acquisition_path": acquired_streams,
        "n_streams_used": len(acquired_streams),
        "cost_spent": round(cost_spent, 2),
        "cost_all_streams": round(cost_all, 2),
        "cost_saved_fraction": round(1 - cost_spent / cost_all, 3),
        "ranking": top_candidates,
        "top_posterior_probability": round(top_prob, 4),
        "confidence_band": "cold_start" if cold_start else band,
        "stopped_at_threshold": top_prob >= stop_threshold,
        "catalogued_printer": true_printer,
        "rank_of_catalogued_printer_by_posterior": rank_of_true,
        "book_info": book_meta,
    }


def evaluate_active(k=3, stop_threshold=0.80, clandestine=False,
                     strict_clandestine=False, strictest_clandestine=False,
                     verbose=True):
    """Evaluate active attribution across all books. Reports recall@k AND
    average cost-savings over running all streams.

    Modes (in increasing strictness):
      clandestine=True: mask printer's name patterns from publisher field.
        Imprint stream cannot identify them. Bookseller stream may still
        leak the answer via known partners.
      strict_clandestine=True: mask BOTH the printer's name AND the
        printer's known bookseller partners. Imprint+bookseller cannot
        leak. Temporal still anchors via the publication year.
      strictest_clandestine=True: also suppress the year. Temporal stream
        returns LR=1.0 for all candidates. This is the genuine
        worst-case clandestine baseline — only vision carries signal."""
    estc_ids = sorted({r.estc for r in index.records if r.estc})
    if verbose:
        if strictest_clandestine:
            mode = "STRICTEST-CLANDESTINE (printer + bookseller + year suppressed)"
        elif strict_clandestine:
            mode = "STRICT-CLANDESTINE (printer name AND bookseller masked)"
        elif clandestine:
            mode = "CLANDESTINE (printer name masked, bookseller visible)"
        else:
            mode = "OPEN (publisher visible)"
        print(f"\n[active-eval] running active attribution recall@{k}, "
              f"stop_threshold={stop_threshold}, mode={mode}")

    per_book = []
    for estc in estc_ids:
        result = active_attribute_book(estc, top_k=10,
                                         stop_threshold=stop_threshold,
                                         clandestine=clandestine,
                                         strict_clandestine=strict_clandestine,
                                         strictest_clandestine=strictest_clandestine,
                                         verbose=False)
        if "error" in result:
            continue
        per_book.append({
            "estc": estc,
            "true_printer": result["catalogued_printer"],
            "n_streams_used": result["n_streams_used"],
            "cost_spent": result["cost_spent"],
            "cost_saved_fraction": result["cost_saved_fraction"],
            "rank_of_true": result["rank_of_catalogued_printer_by_posterior"],
            "top_posterior": result["top_posterior_probability"],
            "stopped_early": result["stopped_at_threshold"],
            "acquisition_path": [s["stream"]
                                  for s in result["acquisition_path"]],
        })

    if not per_book:
        return {"error": "no books evaluable"}

    hits = sum(1 for r in per_book if r["rank_of_true"]
               and r["rank_of_true"] <= k)
    n = len(per_book)
    recall_at_k = hits / n
    avg_cost_saved = sum(r["cost_saved_fraction"] for r in per_book) / n
    avg_streams = sum(r["n_streams_used"] for r in per_book) / n
    n_stopped_early = sum(1 for r in per_book if r["stopped_early"])

    # Path frequency analysis
    path_counter = Counter()
    for r in per_book:
        path_counter[tuple(r["acquisition_path"])] += 1

    summary = {
        "encoder": ENCODER_TAG,
        "method": "active_evidence_acquisition",
        "k": k,
        "stop_threshold": stop_threshold,
        "n_books": n,
        "recall_at_k": round(recall_at_k, 3),
        "avg_streams_used": round(avg_streams, 2),
        "avg_cost_saved_fraction": round(avg_cost_saved, 3),
        "n_stopped_early": n_stopped_early,
        "stopped_early_fraction": round(n_stopped_early / n, 3),
        "clandestine_mode": clandestine,
        "common_acquisition_paths": [
            {"path": list(path), "count": cnt}
            for path, cnt in path_counter.most_common(5)
        ],
    }

    if verbose:
        if strictest_clandestine:
            mode_label = "STRICTEST-CLANDESTINE"
        elif strict_clandestine:
            mode_label = "STRICT-CLANDESTINE"
        elif clandestine:
            mode_label = "CLANDESTINE"
        else:
            mode_label = "OPEN"
        print(f"\n[active-eval] results ({mode_label} mode):")
        print(f"  n_books:            {n}")
        print(f"  recall@{k}:           {recall_at_k:.1%}")
        print(f"  avg streams used:   {avg_streams:.2f} / 4")
        print(f"  avg cost saved:     {avg_cost_saved:.1%}")
        print(f"  stopped early:      {n_stopped_early}/{n} "
              f"({n_stopped_early/n:.1%})")
        print(f"\n  most common acquisition paths:")
        for path, cnt in path_counter.most_common(5):
            print(f"    {cnt:3d}× {' -> '.join(path)}")

    if strictest_clandestine:
        mode_tag = "strictest_clandestine"
    elif strict_clandestine:
        mode_tag = "strict_clandestine"
    elif clandestine:
        mode_tag = "clandestine"
    else:
        mode_tag = "open"
    out_path = (RUN_DIR /
                f"active_eval_{ENCODER_TAG}_{mode_tag}_t{stop_threshold}.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"\n  saved: {out_path}")
    return summary


def _imprint_leakage_subsets():
    """Partition the corpus into two subsets:
      - imprint_leaks: books whose publisher field NAMES the true printer
        (the imprint stream can read the answer)
      - imprint_clean: books whose publisher field does NOT name the true
        printer and these are the naturally-occurring 'closer to clandestine'
        books in the corpus

    Returns (leaks_set, clean_set) of ESTC IDs.
    """
    estc_ids = sorted({r.estc for r in index.records if r.estc})
    leaks, clean = set(), set()
    for estc in estc_ids:
        meta = lookup_estc(estc)
        if "error" in meta:
            continue
        publisher = meta.get("publisher", "")
        true_printer = meta.get("catalogued_printer", "")
        if not true_printer or not publisher:
            clean.add(estc)
            continue
        sigs = _get_imprint_signatures()
        patterns = sigs.get(true_printer, [])
        # Did any pattern for the TRUE printer match the publisher?
        if any(p.search(publisher) for p in patterns):
            leaks.add(estc)
        else:
            clean.add(estc)
    return leaks, clean


def evaluate_active_stratified(k=3, stop_threshold=0.80, verbose=True):
    """Run active attribution stratified by whether the publisher field
    naturally names the true printer.

    Reports recall separately for:
      - imprint_leaks subset: publisher names the printer (the easy case;
        imprint stream alone resolves it)
      - imprint_clean subset: publisher does not name the printer (this is
        the closer-to-clandestine subset)

    The imprint_clean number is the honest indication of how well the
    active pipeline does when it has to actually attribute, rather than
    just read."""
    leaks, clean = _imprint_leakage_subsets()
    if verbose:
        print(f"\n[active-eval/stratified] corpus split:")
        print(f"  imprint_leaks: {len(leaks)} books "
              f"(publisher names the true printer)")
        print(f"  imprint_clean: {len(clean)} books "
              f"(publisher does NOT name the true printer)")

    by_subset = {}
    for label, subset in [("imprint_leaks", leaks), ("imprint_clean", clean)]:
        rows = []
        for estc in sorted(subset):
            result = active_attribute_book(estc, top_k=10,
                                             stop_threshold=stop_threshold,
                                             clandestine=False,
                                             verbose=False)
            if "error" in result:
                continue
            rows.append({
                "estc": estc,
                "true_printer": result["catalogued_printer"],
                "rank_of_true": result["rank_of_catalogued_printer_by_posterior"],
                "top_posterior": result["top_posterior_probability"],
                "n_streams_used": result["n_streams_used"],
                "cost_spent": result["cost_spent"],
                "cost_saved_fraction": result["cost_saved_fraction"],
                "stopped_early": result["stopped_at_threshold"],
                "acquisition_path": [s["stream"]
                                      for s in result["acquisition_path"]],
            })
        if not rows:
            by_subset[label] = None
            continue
        n = len(rows)
        hits = sum(1 for r in rows if r["rank_of_true"]
                   and r["rank_of_true"] <= k)
        path_counter = Counter()
        for r in rows:
            path_counter[tuple(r["acquisition_path"])] += 1
        avg_cost = sum(r["cost_saved_fraction"] for r in rows) / n
        avg_streams = sum(r["n_streams_used"] for r in rows) / n
        n_early = sum(1 for r in rows if r["stopped_early"])
        by_subset[label] = {
            "n_books": n,
            "recall_at_k": round(hits / n, 3),
            "avg_streams_used": round(avg_streams, 2),
            "avg_cost_saved_fraction": round(avg_cost, 3),
            "n_stopped_early": n_early,
            "stopped_early_fraction": round(n_early / n, 3),
            "common_paths": [{"path": list(p), "count": c}
                              for p, c in path_counter.most_common(3)],
        }

    summary = {
        "encoder": ENCODER_TAG,
        "k": k,
        "stop_threshold": stop_threshold,
        "imprint_leaks": by_subset.get("imprint_leaks"),
        "imprint_clean": by_subset.get("imprint_clean"),
    }

    if verbose:
        print(f"\n[active-eval/stratified] results @ threshold "
              f"{stop_threshold}:")
        for label in ("imprint_leaks", "imprint_clean"):
            s = by_subset.get(label)
            if not s:
                continue
            print(f"\n  --- {label} ({s['n_books']} books) ---")
            print(f"    recall@{k}:          {s['recall_at_k']:.1%}")
            print(f"    avg streams used:    {s['avg_streams_used']:.2f}")
            print(f"    avg cost saved:      "
                  f"{s['avg_cost_saved_fraction']:.1%}")
            print(f"    stopped early:       {s['n_stopped_early']}/"
                  f"{s['n_books']} ({s['stopped_early_fraction']:.0%})")
            print(f"    common paths:")
            for p in s["common_paths"]:
                print(f"      {p['count']:3d}× "
                      f"{' -> '.join(p['path'])}")
        print(f"\n  INTERPRETATION:")
        leaks_r = (by_subset.get("imprint_leaks") or {}).get("recall_at_k")
        clean_r = (by_subset.get("imprint_clean") or {}).get("recall_at_k")
        if leaks_r is not None and clean_r is not None:
            print(f"    imprint_leaks recall = {leaks_r:.1%} (trivial — "
                  f"system reads the answer)")
            print(f"    imprint_clean recall = {clean_r:.1%}  (HONEST — "
                  f"this is the real attribution number)")
            if leaks_r - clean_r > 0.20:
                print(f"    The {(leaks_r-clean_r)*100:.0f}-point gap "
                      f"shows the headline depends heavily on imprint "
                      f"leakage.")

    out_path = (RUN_DIR /
                f"active_stratified_{ENCODER_TAG}_t{stop_threshold}.json")
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"\n  saved: {out_path}")
    return summary


def evaluate_active_report(k=3, stop_threshold=0.80, verbose=True):
    """Combined report. Runs FIVE evaluations in sequence and prints
    a single comparison table:
      1. Open mode: full publisher visible.
      2. Clandestine mode: printer name masked, bookseller visible
         
      3. Strict-clandestine: printer name AND associated bookseller masked
         
      4. Strictest-clandestine: ALSO suppress year (temporal goes neutral)
         
      5. Stratified: open mode split by whether publisher naturally
        

    The strictest-clandestine number is the floor what the framework
    achieves with zero usable bibliographic context. The strict-clandestine
    number is the realistic-honest case where date is trusted.
    """
    print(f"\n{'='*72}")
    print(f"ACTIVE ATTRIBUTION -- REPORT")
    print(f"  encoder={ENCODER_TAG}, k={k}, threshold={stop_threshold}")
    print(f"{'='*72}")

    print(f"\n--- (1/5) OPEN MODE (publisher visible) ---")
    open_result = evaluate_active(k=k, stop_threshold=stop_threshold,
                                    clandestine=False, verbose=True)

    print(f"\n--- (2/5) CLANDESTINE MODE (printer name masked, "
          f"bookseller visible) ---")
    clan_result = evaluate_active(k=k, stop_threshold=stop_threshold,
                                    clandestine=True, verbose=True)

    print(f"\n--- (3/5) STRICT-CLANDESTINE MODE "
          f"(printer name AND bookseller masked) ---")
    strict_result = evaluate_active(k=k, stop_threshold=stop_threshold,
                                      strict_clandestine=True, verbose=True)

    print(f"\n--- (4/5) STRICTEST-CLANDESTINE MODE "
          f"(printer + bookseller + year suppressed) ---")
    strictest_result = evaluate_active(k=k, stop_threshold=stop_threshold,
                                         strictest_clandestine=True,
                                         verbose=True)

    print(f"\n--- (5/5) STRATIFIED DIAGNOSTIC ---")
    strat_result = evaluate_active_stratified(k=k,
                                                stop_threshold=stop_threshold,
                                                verbose=True)

    print(f"\n{'='*72}")
    print(f"HEADLINE NUMBERS")
    print(f"{'='*72}")
    print(f"{'mode':<52} {'recall@'+str(k):>10} {'cost saved':>12}")
    print(f"{'-'*76}")
    print(f"{'open (publisher visible)':<52} "
          f"{open_result['recall_at_k']:>10.1%} "
          f"{open_result['avg_cost_saved_fraction']:>12.1%}")
    print(f"{'clandestine (printer name masked)':<52} "
          f"{clan_result['recall_at_k']:>10.1%} "
          f"{clan_result['avg_cost_saved_fraction']:>12.1%}")
    print(f"{'strict-clandestine (printer + bookseller masked)':<52} "
          f"{strict_result['recall_at_k']:>10.1%} "
          f"{strict_result['avg_cost_saved_fraction']:>12.1%}")
    print(f"{'strictest-clandestine (+ year suppressed)':<52} "
          f"{strictest_result['recall_at_k']:>10.1%} "
          f"{strictest_result['avg_cost_saved_fraction']:>12.1%}")
    leaks_r = (strat_result['imprint_leaks'] or {}).get('recall_at_k')
    clean_r = (strat_result['imprint_clean'] or {}).get('recall_at_k')
    if leaks_r is not None:
        leaks_n = strat_result['imprint_leaks']['n_books']
        print(f"{'  open, imprint-leaks subset':<52} "
              f"{leaks_r:>10.1%}   (n={leaks_n})")
    if clean_r is not None:
        clean_n = strat_result['imprint_clean']['n_books']
        print(f"{'  open, imprint-clean subset':<52} "
              f"{clean_r:>10.1%}   (n={clean_n})")

    print(f"\n{'='*72}")
    print(f" INTERPRETATION")
    print(f"{'='*72}")
    print("  Mode-by-mode story of where the idea comes from:")
    print()
    print(f"  OPEN ({open_result['recall_at_k']:.1%}):")
    print("    Imprint reads the printer's name from the publisher field.")
    print("    Performance dominated by regex matching, not vision.")
    print()
    print(f"  CLANDESTINE ({clan_result['recall_at_k']:.1%}):")
    print("    Printer name masked. Bookseller stream may still reveal")
    print("    the answer via known partners (Cockerill→Roberts etc).")
    print()
    print(f"  STRICT-CLANDESTINE ({strict_result['recall_at_k']:.1%}):")
    print("    Printer name AND bookseller partners masked. Temporal +")
    print("    vision carry the signal. This is the case when the")
    print("    date is trusted.")
    print()
    print(f"  STRICTEST-CLANDESTINE ({strictest_result['recall_at_k']:.1%}):")
    print("    Date ALSO suppressed (worst-case: false imprint, fake")
    print("    date). Only vision carries signal. This is the FLOOR.")
    print()

    # Component-attribution analysis
    print(f"  COMPONENT ATTRIBUTION:")
    delta_clan_strict = clan_result['recall_at_k'] - strict_result['recall_at_k']
    delta_strict_strictest = strict_result['recall_at_k'] - strictest_result['recall_at_k']
    print(f"    bookseller stream contribution: "
          f"{delta_clan_strict * 100:+.0f} points "
          f"(clandestine - strict)")
    print(f"    temporal stream contribution:   "
          f"{delta_strict_strictest * 100:+.0f} points "
          f"(strict - strictest)")
    print(f"    vision-only baseline (matches strictest):")
    print(f"      strictest = {strictest_result['recall_at_k']:.1%}")
    print()

    print(f"  For a writeup, report all four numbers with the story:")
    print(f"    'Performance ranges from {strictest_result['recall_at_k']:.1%}")
    print(f"     (vision-only, false-imprint scenario) through")
    print(f"     {strict_result['recall_at_k']:.1%} (vision + reliable date) to")
    print(f"     {open_result['recall_at_k']:.1%} (full bibliographic context).'")

    combined = {
        "open": open_result,
        "clandestine": clan_result,
        "strict_clandestine": strict_result,
        "strictest_clandestine": strictest_result,
        "stratified": strat_result,
        "encoder": ENCODER_TAG,
        "stop_threshold": stop_threshold,
    }
    out_path = RUN_DIR / f"active_report_{ENCODER_TAG}_t{stop_threshold}.json"
    out_path.write_text(json.dumps(combined, indent=2, default=str))
    print(f"\n  full report saved: {out_path}")
    return combined


def evaluate_multi_evidence(k=3, verbose=True, use_streams=None):
    """Run multi-evidence attribution across all evaluable books."""
    estc_ids = sorted({r.estc for r in index.records if r.estc})
    if verbose:
        print(f"\n[multi-eval] running multi-evidence recall@{k} "
              f"({use_streams or 'all streams'})")

    per_book = []
    for estc in estc_ids:
        result = attribute_book_multi_evidence(estc, top_k=10,
                                                 use_streams=use_streams)
        if "error" in result:
            continue
        rk = result["rank_of_catalogued_printer_by_posterior"]
        top_prob = result["top_posterior_probability"]
        per_book.append({
            "estc": estc,
            "true_printer": result["catalogued_printer"],
            "rank_of_true": rk,
            "top_prob": top_prob,
            "leaked_vision": result["vision_leakage_suspect"],
        })

    if not per_book:
        print("[multi-eval] no books"); return None

    n = len(per_book)
    hits = sum(1 for r in per_book if r["rank_of_true"]
               and r["rank_of_true"] <= k)
    honest_hits = sum(1 for r in per_book if r["rank_of_true"]
                       and r["rank_of_true"] <= k and not r["leaked_vision"])
    ranks = [r["rank_of_true"] for r in per_book if r["rank_of_true"]]
    n_printers = len(set(printer_slugs))
    chance = k / n_printers

    summary = {
        "method": "multi_evidence_bayesian",
        "streams_used": use_streams or ["vision", "imprint",
                                          "temporal", "bookseller"],
        "encoder": ENCODER_TAG,
        "k": k,
        "n_books_evaluated": n,
        "raw_recall_at_k": round(hits / n, 3),
        "honest_recall_at_k": round(honest_hits / n, 3),
        "chance_baseline_at_k": round(chance, 3),
        "raw_lift": round((hits / n) / max(chance, 1e-9), 2),
        "honest_lift": round((honest_hits / n) / max(chance, 1e-9), 2),
        "median_rank": int(np.median(ranks)) if ranks else None,
        "median_top_prob": round(float(np.median(
            [r["top_prob"] for r in per_book])), 3),
        "n_leaked_vision": sum(1 for r in per_book if r["leaked_vision"]),
    }

    if verbose:
        print(f"\n[multi-eval] {ENCODER_TAG} encoder, "
              f"{n} books, streams={summary['streams_used']}")
        print(f"  raw recall@{k}:    {summary['raw_recall_at_k']:.1%}")
        print(f"  honest recall@{k}: {summary['honest_recall_at_k']:.1%}  "
              f"(lift {summary['honest_lift']}x)")
        print(f"  median rank:       {summary['median_rank']}")
        print(f"  median top posterior prob: {summary['median_top_prob']}")
        print(f"  n_leaked (vision): {summary['n_leaked_vision']}/{n}")

    streams_tag = "_".join(sorted(summary["streams_used"]))
    out_path = RUN_DIR / f"multi_evidence_{streams_tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"  saved: {out_path}")
    return summary


def ablate_evidence_streams(k=3, verbose=True):
    """Run leave-one-stream-out ablation: vision-only, +imprint, +temporal,
    +bookseller, and all four. Report how each stream contributes."""
    all_streams = ["vision", "imprint", "temporal", "bookseller"]
    configurations = [
        ("vision-only", ["vision"]),
        ("vision + imprint", ["vision", "imprint"]),
        ("vision + temporal", ["vision", "temporal"]),
        ("vision + bookseller", ["vision", "bookseller"]),
        ("all streams", all_streams),
    ]
    results = []
    for label, streams in configurations:
        print(f"\n=== {label} ===")
        r = evaluate_multi_evidence(k=k, verbose=verbose, use_streams=streams)
        if r:
            r["config_label"] = label
            results.append(r)

    print("\n" + "=" * 72)
    print(f"ABLATION SUMMARY (recall@{k})")
    print("=" * 72)
    print(f"{'configuration':<25} {'honest_r@k':>12} {'lift':>8} "
          f"{'med_rank':>10} {'med_top_p':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r['config_label']:<25} "
              f"{r['honest_recall_at_k']:>12.1%} "
              f"{r['honest_lift']:>8.2f}x "
              f"{str(r['median_rank']):>10} "
              f"{r['median_top_prob']:>12.3f}")

    out_path = RUN_DIR / "evidence_ablation.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved: {out_path}")
    return results


def find_books_by_printer(printer_slug: str,
                           year_min: int = None,
                           year_max: int = None) -> dict:
    """List books in the corpus attributed to a printer."""
    seen = {}
    for r in index.records:
        if r.printer_slug != printer_slug: continue
        try: y = int(r.year) if r.year else None
        except ValueError: y = None
        if year_min and (y is None or y < year_min): continue
        if year_max and (y is None or y > year_max): continue
        key = r.estc or r.title
        if not key or key in seen: continue
        seen[key] = {"estc": r.estc, "title": r.title[:200],
                     "year": r.year, "publisher": r.publisher,
                     "n_glyphs_in_corpus": 0}
    for r in index.records:
        if r.printer_slug != printer_slug: continue
        key = r.estc or r.title
        if key in seen:
            seen[key]["n_glyphs_in_corpus"] += 1
    return {"printer": printer_slug, "n_books": len(seen),
            "books": list(seen.values())}


def search_imprints(query: str, top_k: int = 10,
                     printer_slug: str = None) -> dict:
    """Hybrid (dense + BM25) search over book titles, imprints, publishers."""
    if books_tbl is None:
        return {"error": "books collection not available"}
    q_vec = embed_text(query)[0].astype(np.float32)
    try:
        search = books_tbl.search(query_type="hybrid").text(query).vector(q_vec)
    except Exception:
        search = books_tbl.search(q_vec)
    if printer_slug:
        search = search.where(f"printer_slug = '{printer_slug}'",
                              prefilter=True)
    results = search.limit(top_k).to_list()
    return {"query": query, "results": [{
        "estc": r["estc"], "title": r["title"][:200],
        "year": r["year"], "printer": r["printer_slug"],
        "publisher": r["publisher"],
        "relevance": float(r.get("_score", r.get("_distance", 0.0))),
    } for r in results]}


def lookup_estc(estc_id: str) -> dict:
    """Full bibliographic record for a specific ESTC identifier."""
    matches = [r for r in index.records if r.estc == estc_id]
    if not matches:
        return {"error": f"no records for ESTC {estc_id}"}
    first = matches[0]
    return {
        "estc": estc_id, "title": first.title, "year": first.year,
        "catalogued_printer": first.printer_slug,
        "publisher": first.publisher,
        "n_glyphs_in_corpus": len(matches),
        "characters_represented": sorted({r.char for r in matches}),
        "glyph_ids": [r.id for r in matches[:20]],
    }


def search_literature(query: str, top_k: int = 5) -> dict:
    """Hybrid search over scholarly literature on Restoration printing,
    including empirical baselines for this corpus."""
    q_vec = embed_text(query)[0].astype(np.float32)
    try:
        search = literature_tbl.search(query_type="hybrid").text(query).vector(q_vec)
    except Exception:
        search = literature_tbl.search(q_vec)
    results = search.limit(top_k).to_list()
    return {"query": query, "results": [{
        "source": r["source"], "year": r["year"],
        "text": r["text"][:600],
        "relevance": float(r.get("_score", r.get("_distance", 0.0))),
    } for r in results]}


def list_printers(min_glyphs: int = 1) -> dict:
    """List printers in the corpus with glyph counts and year range."""
    by_p = defaultdict(list)
    for r in index.records:
        by_p[r.printer_slug].append(r)
    out = []
    for slug, rs in by_p.items():
        if len(rs) < min_glyphs: continue
        years = [int(r.year) for r in rs if r.year.isdigit()]
        out.append({
            "printer_slug": slug, "n_glyphs": len(rs),
            "n_books": len({r.estc or r.title for r in rs}),
            "year_range": [min(years), max(years)] if years else [None, None],
            "characters": "".join(sorted({r.char for r in rs if r.char != "?"})),
        })
    out.sort(key=lambda x: -x["n_glyphs"])
    return {"n_printers": len(out), "printers": out}


def get_glyph_metadata(glyph_id: str) -> dict:
    """Full record for one glyph including cluster and book context."""
    for i, r in enumerate(index.records):
        if r.id == glyph_id:
            return {
                "glyph_id": r.id, "char": r.char, "year": r.year,
                "printer": r.printer_slug, "estc": r.estc,
                "title": r.title, "publisher": r.publisher,
                "cluster_id": assignments[i] or "(unassigned/noise)",
                "iiif_url": r.iiif_url,
            }
    return {"error": f"glyph_id '{glyph_id}' not found"}


def get_empirical_baselines() -> dict:
    """Validated performance baselines for this corpus."""
    return EMPIRICAL_BASELINES


# %% 9b. ENCODER EVALUATION  ──────────────────────────────────────────────────
# Run the full book-level held-out validation, the way history_rag.py's
# auto-validation cell does. Lets you compare DINOv2 vs contrastive encoder
# on the same recall@k metric.

def evaluate_encoder(k=3, verbose=True):
    """Run book-level held-out attribution across all evaluable books and
    report recall@k. Also stratifies by whether the catalogued
    printer has only this book in the corpus (singleton) vs multiple books
    (multi-book) — this is the key test for whether the encoder learned
    damaged-sort identity vs printer house style.

    If the encoder learns DAMAGED-SORT IDENTITY: singleton recall ≈
    multi-book recall (the specific damage carries the info, regardless
    of whether other same-printer books holds the cluster space).

    If the encoder learns PRINTER HOUSE : singleton recall drops
    sharply (no other same-printer books to anchor the projection).

    Returns a dict with both overall and stratified numbers.
    """
    estc_ids = sorted({r.estc for r in index.records if r.estc})

    # Pre-compute how many books each printer has, so we can stratify.
    # Counted by unique ESTC (deduped books, not glyph counts).
    books_per_printer = Counter()
    estc_to_printer = {}
    for r in index.records:
        if r.estc:
            estc_to_printer[r.estc] = r.printer_slug
    for estc, slug in estc_to_printer.items():
        books_per_printer[slug] += 1

    if verbose:
        print(f"\n[eval] running recall@{k} over {len(estc_ids)} books "
              f"with encoder={ENCODER_TAG}...")
        n_singletons = sum(1 for e in estc_ids
                            if books_per_printer[estc_to_printer.get(e, "")] == 1)
        print(f"  {n_singletons} singleton-printer books, "
              f"{len(estc_ids) - n_singletons} multi-book-printer books")

    # Per-book records — we'll stratify after collecting everything.
    per_book = []  # one dict per evaluable book

    for estc in estc_ids:
        result = attribute_book(estc, top_k=10)
        if "error" in result:
            continue
        true_printer = result.get("catalogued_printer", "")
        rk = result["rank_of_catalogued_printer"]
        top_cos = float(result["ranking"][0]["cosine"])
        is_leaked = result["leakage_suspect"]
        n_books_for_printer = books_per_printer.get(true_printer, 0)
        per_book.append({
            "estc": estc,
            "true_printer": true_printer,
            "n_books_for_printer": n_books_for_printer,
            "rank_of_true": rk,
            "top_cosine": top_cos,
            "is_leaked": is_leaked,
            "confidence_band": result["confidence_band"],
        })

    if not per_book:
        print("[eval] no books evaluated"); return None

    def _summarise(rows, label):
        """Compute recall@k summary over a subset of per_book rows."""
        n = len(rows)
        ranks = [r["rank_of_true"] for r in rows if r["rank_of_true"] is not None]
        hits = sum(1 for r in rows if r["rank_of_true"] and r["rank_of_true"] <= k)
        honest_hits = sum(1 for r in rows
                          if r["rank_of_true"] and r["rank_of_true"] <= k
                          and not r["is_leaked"])
        n_leaked = sum(1 for r in rows if r["is_leaked"])
        cos_top1 = [r["top_cosine"] for r in rows if r["rank_of_true"] == 1]
        cos_other = [r["top_cosine"] for r in rows if r["rank_of_true"] != 1]
        return {
            "label": label,
            "n": n,
            "raw_recall_at_k": round(hits / max(n, 1), 3),
            "honest_recall_at_k": round(honest_hits / max(n, 1), 3),
            "n_leaked": n_leaked,
            "median_rank": int(np.median(ranks)) if ranks else None,
            "median_cos_top1": (round(float(np.median(cos_top1)), 3)
                                 if cos_top1 else None),
            "median_cos_other": (round(float(np.median(cos_other)), 3)
                                  if cos_other else None),
        }

    overall = _summarise(per_book, "overall")
    singletons = _summarise(
        [r for r in per_book if r["n_books_for_printer"] == 1],
        "singleton-printer")
    multi_book = _summarise(
        [r for r in per_book if r["n_books_for_printer"] >= 2],
        "multi-book-printer")

    n_printers = len(set(printer_slugs))
    chance = k / n_printers

    # Per-band counts for the full set
    band_counts = Counter(r["confidence_band"] for r in per_book)

    leaked_books = [{"estc": r["estc"], "cos": r["top_cosine"],
                      "printer": r["true_printer"]}
                     for r in per_book if r["is_leaked"]]

    summary = {
        "encoder": ENCODER_TAG,
        "k": k,
        "n_books_evaluated": overall["n"],
        "n_printers": n_printers,
        "chance_baseline_at_k": round(chance, 3),
        "overall": overall,
        "stratified": {
            "singleton": singletons,
            "multi_book": multi_book,
        },
        # Diagnostic: gap between singleton and multi-book recall.
        # Large positive gap (multi-book >> singleton) indicates the encoder
        # is learning printer house style rather than damaged-sort identity.
        "stratification_gap": (
            None if singletons["n"] == 0 or multi_book["n"] == 0 else
            round(multi_book["honest_recall_at_k"]
                  - singletons["honest_recall_at_k"], 3)
        ),
        "n_leaked": overall["n_leaked"],
        "leaked_books": leaked_books,
        "confidence_band_distribution": dict(band_counts),
        # Top-level aliases for backwards compatibility with --compare
        "raw_recall_at_k": overall["raw_recall_at_k"],
        "honest_recall_at_k": overall["honest_recall_at_k"],
        "raw_lift": round(overall["raw_recall_at_k"]
                          / max(chance, 1e-9), 2),
        "honest_lift": round(overall["honest_recall_at_k"]
                              / max(chance, 1e-9), 2),
        "median_rank": overall["median_rank"],
        "median_cosine_when_rank_1": overall["median_cos_top1"],
        "median_cosine_when_rank_not_1": overall["median_cos_other"],
    }

    if verbose:
        print(f"\n[eval] {ENCODER_TAG} encoder, "
              f"{overall['n']} books, {n_printers} printers, "
              f"chance@{k} = {chance:.1%}")
        print(f"\n  OVERALL:")
        print(f"    raw recall@{k}:    {overall['raw_recall_at_k']:.1%}")
        print(f"    honest recall@{k}: {overall['honest_recall_at_k']:.1%}  "
              f"(lift {summary['honest_lift']}x)")
        print(f"    leakage suspects:  {overall['n_leaked']}/{overall['n']}")
        print(f"    median rank:       {overall['median_rank']}")
        print(f"    cos median when rank=1: {overall['median_cos_top1']}, "
              f"rank>1: {overall['median_cos_other']}")

        print(f"\n  STRATIFIED (the key test for house-style vs damaged-sort):")
        print(f"    singleton-printer books   ({singletons['n']:2d} books, "
              f"printer has ONLY this book):")
        print(f"      honest recall@{k}: {singletons['honest_recall_at_k']:.1%}  "
              f"(leaked={singletons['n_leaked']})")
        print(f"      median rank:       {singletons['median_rank']}")
        print(f"    multi-book-printer books  ({multi_book['n']:2d} books, "
              f"printer has 2+ books):")
        print(f"      honest recall@{k}: {multi_book['honest_recall_at_k']:.1%}  "
              f"(leaked={multi_book['n_leaked']})")
        print(f"      median rank:       {multi_book['median_rank']}")
        gap = summary["stratification_gap"]
        if gap is not None:
            print(f"\n    Stratification gap: {gap:+.1%}")
            if abs(gap) < 0.10:
                interp = ("SMALL gap — recall is similar across both "
                          "strata, consistent with the encoder learning "
                          "damaged-sort identity that generalises beyond "
                          "printer house style.")
            elif gap > 0.20:
                interp = ("LARGE gap — multi-book books are attributed "
                          "much more reliably than singletons. This "
                          "strongly suggests the encoder is relying on "
                          "printer house style (other same-printer books "
                          "in the cluster space anchor the projection). "
                          "True damaged-sort identification would not "
                          "show this asymmetry.")
            else:
                interp = ("MODERATE gap — encoder is using some house-"
                          "style signal but also some sort-specific "
                          "signal. Both contribute.")
            print(f"    Interpretation: {interp}")
        print(f"\n    band distribution: "
              f"{summary['confidence_band_distribution']}")

    eval_path = RUN_DIR / f"recall_at_{k}_{ENCODER_TAG}.json"
    eval_path.write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"\n  saved: {eval_path}")
    return summary


def evaluate_printer_holdout(k=3, verbose=True):
    """For each printer, hold out ALL of their
    books (and glyphs) at once, rebuild clusters without any of them, then
    test whether each held-out book still attributes to something.

    When printer X is
    fully removed from the cluster space, the projection has NO
    same-printer anchor available. The true printer cannot rank #1 (it
    has been excluded), so what we measure is the TOP COSINE distribution:
      - if the encoder finds damage info, top cosines should
        remain elevated (book still binds to some other shop's clusters
        based on shared damage)

    We cache cluster rebuilds per printer so this is only ~30 calls to
    cluster_by_char instead of ~80.
    """
    estc_to_printer = {}
    for r in index.records:
        if r.estc:
            estc_to_printer[r.estc] = r.printer_slug

    estc_ids = sorted(estc_to_printer.keys())
    n_printers = len(set(printer_slugs))

    if verbose:
        print(f"\n[printer-holdout] running recall@{k} with full-printer "
              f"hold-out, encoder={ENCODER_TAG}")
        print(f"  for each book, ALL books by its catalogued printer are "
              f"removed from the cluster space")
        print(f"  ({len(estc_ids)} books across {n_printers} printers)")

    per_book = []
    cluster_cache = {}

    def _clusters_without_printer(slug):
        if slug in cluster_cache:
            return cluster_cache[slug]
        keep_idxs = [i for i, r in enumerate(index.records)
                     if r.printer_slug != slug]
        h_ass, h_sum = cluster_by_char(embeddings, chars,
                                        restrict_idxs=keep_idxs)
        cluster_cache[slug] = (keep_idxs, h_ass, h_sum)
        return cluster_cache[slug]

    for estc in estc_ids:
        true_printer = estc_to_printer[estc]
        book_glyph_idxs = [i for i, r in enumerate(index.records)
                           if r.estc == estc]
        if not book_glyph_idxs:
            continue

        keep_idxs, h_ass, h_sum = _clusters_without_printer(true_printer)
        if not h_sum:
            continue

        all_cids = sorted({s["cluster_id"] for s in h_sum})
        cid_to_col = {c: i for i, c in enumerate(all_cids)}
        keep_printers = sorted({printer_slugs[i] for i in keep_idxs})
        p2row = {p: i for i, p in enumerate(keep_printers)}

        M = np.zeros((len(keep_printers), len(all_cids)), dtype=np.float32)
        for i in keep_idxs:
            slug = printer_slugs[i]; cid = h_ass[i]
            if cid and cid in cid_to_col:
                M[p2row[slug], cid_to_col[cid]] += 1
        df = (M > 0).sum(axis=0)
        idf = np.log((M.shape[0] + 1) / (df + 1)) + 1.0
        tf = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
        tfidf = tf * idf
        fp = tfidf / np.maximum(np.linalg.norm(tfidf, axis=1, keepdims=True), 1e-9)

        centroids_by_char = defaultdict(list)
        for s in h_sum:
            centroids_by_char[s["char"]].append((s["cluster_id"], s["centroid"]))
        q_counts = np.zeros(len(all_cids), dtype=np.float32)
        for i in book_glyph_idxs:
            cands = centroids_by_char.get(chars[i], [])
            if not cands:
                continue
            best_cid, best_sim = None, -1.0
            for cid, cent in cands:
                sim = float(embeddings[i] @ cent)
                if sim > best_sim:
                    best_sim, best_cid = sim, cid
            if best_cid:
                q_counts[cid_to_col[best_cid]] += 1.0
        if q_counts.sum() < 1.0:
            continue
        q_vec = q_counts / q_counts.sum() * idf
        q_vec /= max(np.linalg.norm(q_vec), 1e-9)

        sims = fp @ q_vec
        order = np.argsort(-sims)
        ranking = [(keep_printers[int(i)], float(sims[int(i)]))
                   for i in order]
        top_match, top_cos = ranking[0]
        second_cos = ranking[1][1] if len(ranking) > 1 else None
        gap = (top_cos - second_cos) if second_cos else top_cos

        per_book.append({
            "estc": estc,
            "true_printer": true_printer,
            "top_match_when_excluded": top_match,
            "top_cosine": float(top_cos),
            "gap_to_second": float(gap),
            "n_glyphs": len(book_glyph_idxs),
        })

    if not per_book:
        print("[printer-holdout] no books evaluable"); return None

    top_cosines = [b["top_cosine"] for b in per_book]
    gaps = [b["gap_to_second"] for b in per_book]

    standard_path = RUN_DIR / f"recall_at_{k}_{ENCODER_TAG}.json"
    standard_top1_cos = None
    if standard_path.exists():
        try:
            d = json.loads(standard_path.read_text())
            standard_top1_cos = (d.get("overall", {}).get("median_cos_top1")
                                  or d.get("median_cosine_when_rank_1"))
        except Exception:
            pass

    summary = {
        "encoder": ENCODER_TAG,
        "k": k,
        "n_books_evaluated": len(per_book),
        "median_top_cosine_no_printer": round(float(np.median(top_cosines)), 3),
        "mean_top_cosine_no_printer":   round(float(np.mean(top_cosines)), 3),
        "min_top_cosine_no_printer":    round(float(np.min(top_cosines)), 3),
        "max_top_cosine_no_printer":    round(float(np.max(top_cosines)), 3),
        "median_gap_to_second":         round(float(np.median(gaps)), 3),
        "standard_eval_median_cos_when_rank1": standard_top1_cos,
        "per_book": per_book,
    }

    if verbose:
        print(f"\n[printer-holdout] {ENCODER_TAG} encoder, "
              f"{len(per_book)} books evaluated")
        print(f"  top cosines (printer fully excluded from cluster space):")
        print(f"    median: {summary['median_top_cosine_no_printer']}")
        print(f"    mean:   {summary['mean_top_cosine_no_printer']}")
        print(f"    range:  [{summary['min_top_cosine_no_printer']}, "
              f"{summary['max_top_cosine_no_printer']}]")
        print(f"  median gap to second: {summary['median_gap_to_second']}")
        if standard_top1_cos is not None:
            drop = standard_top1_cos - summary['median_top_cosine_no_printer']
            print(f"\n  Comparison with standard eval:")
            print(f"    median cos when rank=#1 (printer IN cluster space): "
                  f"{standard_top1_cos}")
            print(f"    median cos when printer FULLY EXCLUDED:           "
                  f"{summary['median_top_cosine_no_printer']}")
            print(f"    drop:  {drop:+.3f}")
            if drop > 0.20:
                interp = ("LARGE drop — removing the printer's other "
                          "books from the cluster space crashes the "
                          "top-match cosine, strongly suggesting the "
                          "encoder relies on PRINTER HOUSE STYLE.")
            elif drop > 0.10:
                interp = ("MODERATE drop — some of the info is "
                          "house-style-driven, some survives. The encoder "
                          "finds both kinds of evidence.")
            elif drop > 0.0:
                interp = ("SMALL drop — top cosines remain elevated even "
                          "with the printer excluded. Consistent with the "
                          "encoder finding cross-shop damaged-sort "
                          "info")
            else:
                interp = ("NO drop or top cosines INCREASED — surprising. "
                          "May indicate noise or a degenerate clustering.")
            print(f"    Interpretation: {interp}")

    out_path = RUN_DIR / f"printer_holdout_{ENCODER_TAG}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    if verbose:
        print(f"\n  saved: {out_path}")
    return summary


def compare_encoders_offline(k=3):
    """Compare saved recall@k results for DINOv2 vs contrastive (if both
    exist on disk). Run evaluate_encoder() under each encoder setting first."""
    out = {}
    for tag in ("dinov2", "contrastive"):
        path = OUT_DIR / "runs" / f"cdt_{tag}" / f"recall_at_{k}_{tag}.json"
        if path.exists():
            out[tag] = json.loads(path.read_text())
    if len(out) < 2:
        print("[compare] need both encoders' results on disk; "
              f"found: {list(out.keys())}")
        print("  Set USE_CONTRASTIVE_ENCODER=True, run evaluate_encoder(),")
        print("  then set it False, restart, run evaluate_encoder() again.")
        return out

    print(f"\n=== Encoder comparison @ k={k} ===")
    print(f"{'metric':<40} {'DINOv2':>10} {'Contrastive':>12}")
    print("-" * 65)
    print("OVERALL")
    for key in ("honest_recall_at_k", "raw_recall_at_k", "honest_lift",
                "median_rank", "median_cosine_when_rank_1",
                "median_cosine_when_rank_not_1", "n_leaked",
                "n_books_evaluated"):
        d_val = out["dinov2"].get(key)
        c_val = out["contrastive"].get(key)
        print(f"  {key:<38} {str(d_val):>10} {str(c_val):>12}")

    # Stratified breakdown (only if both encoders have the new format)
    def _strat(o, kind):
        try:
            return o["stratified"][kind]
        except (KeyError, TypeError):
            return None

    d_single = _strat(out["dinov2"], "singleton")
    c_single = _strat(out["contrastive"], "singleton")
    d_multi  = _strat(out["dinov2"], "multi_book")
    c_multi  = _strat(out["contrastive"], "multi_book")

    if d_single and c_single and d_multi and c_multi:
        print("\nSTRATIFIED (singleton-printer books — no anchor for house style)")
        print(f"  {'n':<38} {d_single['n']:>10} {c_single['n']:>12}")
        for key in ("honest_recall_at_k", "raw_recall_at_k",
                    "median_rank", "n_leaked"):
            d_val = d_single.get(key); c_val = c_single.get(key)
            print(f"  {key:<38} {str(d_val):>10} {str(c_val):>12}")

        print("\nSTRATIFIED (multi-book-printer books — house-style anchored)")
        print(f"  {'n':<38} {d_multi['n']:>10} {c_multi['n']:>12}")
        for key in ("honest_recall_at_k", "raw_recall_at_k",
                    "median_rank", "n_leaked"):
            d_val = d_multi.get(key); c_val = c_multi.get(key)
            print(f"  {key:<38} {str(d_val):>10} {str(c_val):>12}")

        d_gap = out["dinov2"].get("stratification_gap")
        c_gap = out["contrastive"].get("stratification_gap")
        print(f"\n  {'stratification_gap (multi - single)':<38} "
              f"{str(d_gap):>10} {str(c_gap):>12}")
        print(f"\n  Interpretation: a large positive gap means the encoder")
        print(f"  relies on having other same-printer books in the cluster")
        print(f"  space (house style). A small gap means damaged-sort")
        print(f"  identity is doing the work.")
    else:
        print("\n(Stratification breakdown not available -- rerun evaluate "
              "with the current code to add it.)")

    return out


TOOLS = {
    "find_similar_glyphs": find_similar_glyphs,
    "compare_printer_fingerprints": compare_printer_fingerprints,
    "audit_cluster": audit_cluster,
    "attribute_book": attribute_book,
    "attribute_book_multi_evidence": attribute_book_multi_evidence,
    "active_attribute_book": active_attribute_book,
    "find_books_by_printer": find_books_by_printer,
    "search_imprints": search_imprints,
    "lookup_estc": lookup_estc,
    "search_literature": search_literature,
    "list_printers": list_printers,
    "get_glyph_metadata": get_glyph_metadata,
    "get_empirical_baselines": get_empirical_baselines,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "find_similar_glyphs",
        "description": "Find glyphs visually similar to a given glyph (by "
                       "id) or sample from a character class.",
        "parameters": {"type": "object", "properties": {
            "glyph_id": {"type": "string"},
            "char": {"type": "string"},
            "top_k": {"type": "integer", "default": 10},
        }}}},
    {"type": "function", "function": {
        "name": "compare_printer_fingerprints",
        "description": "PRINTER-level held-out attribution. Hold out the "
                       "query printer entirely. More reliable than "
                       "attribute_book. Returns ranking + confidence_band.",
        "parameters": {"type": "object", "properties": {
            "query_printer": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        }, "required": ["query_printer"]}}},
    {"type": "function", "function": {
        "name": "audit_cluster",
        "description": "Check whether a cluster is genuine shared-sort "
                       "evidence or intra-shop variation.",
        "parameters": {"type": "object",
                       "properties": {"cluster_id": {"type": "string"}},
                       "required": ["cluster_id"]}}},
    {"type": "function", "function": {
        "name": "attribute_book",
        "description": "BOOK-level held-out attribution. The realistic "
                       "clandestine task. Returns ranking + leakage_suspect "
                       "+ confidence_band.",
        "parameters": {"type": "object", "properties": {
            "estc_id": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        }, "required": ["estc_id"]}}},
    {"type": "function", "function": {
        "name": "attribute_book_multi_evidence",
        "description": "Multi-evidence Bayesian attribution. Combines four "
                       "evidence streams (vision, imprint, temporal, "
                       "bookseller) into calibrated POSTERIOR PROBABILITIES "
                       "over candidate printers. Prefer this over "
                       "attribute_book when bibliographic context matters "
                       "(year, imprint, bookseller). Returns posterior "
                       "probabilities and per-stream log-LR breakdown so "
                       "you can see which evidence drove the ranking.",
        "parameters": {"type": "object", "properties": {
            "estc_id": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "use_streams": {"type": "array", "items": {"type": "string"},
                            "description": "Which streams to use. Defaults "
                                            "to all four. Valid: vision, "
                                            "imprint, temporal, bookseller."},
        }, "required": ["estc_id"]}}},
    {"type": "function", "function": {
        "name": "active_attribute_book",
        "description": "Active-inference attribution. Iteratively picks "
                       "the most informative evidence stream to run next, "
                       "stopping once posterior P(top printer) reaches the "
                       "threshold. Cheaper than running all streams when "
                       "easy attributions resolve quickly. Returns the "
                       "acquisition path (which streams ran in what order) "
                       "and cost savings. Prefer this when computational "
                       "cost matters or when you want to see the model's "
                       "evidence-gathering decisions. "
                       "Use the masking flags when the user asks about "
                       "clandestine or anonymous attribution: clandestine "
                       "hides the printer's name in the publisher field, "
                       "strict_clandestine also hides their known "
                       "bookseller partners, and strictest_clandestine "
                       "additionally suppresses the year — each "
                       "successively harder test.",
        "parameters": {"type": "object", "properties": {
            "estc_id": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "stop_threshold": {"type": "number",
                                "description": "Posterior P(top printer) "
                                                "at which to stop. "
                                                "Default 0.80."},
            "clandestine": {"type": "boolean",
                             "description": "If true, mask the "
                                            "printer's NAME from the "
                                            "publisher field so the "
                                            "imprint stream cannot read "
                                            "it. Use when the user asks "
                                            "for 'clandestine mode'."},
            "strict_clandestine": {"type": "boolean",
                                    "description": "If true, mask the "
                                                    "printer's name AND "
                                                    "their known "
                                                    "bookseller "
                                                    "partners. Use for "
                                                    "'strict clandestine "
                                                    "mode'."},
            "strictest_clandestine": {"type": "boolean",
                                       "description": "If true, ALSO "
                                                       "suppress the "
                                                       "year so the "
                                                       "temporal "
                                                       "stream goes "
                                                       "neutral. Only "
                                                       "vision evidence "
                                                       "carries signal. "
                                                       "The genuinely "
                                                       "worst-case "
                                                       "test."},
        }, "required": ["estc_id"]}}},
    {"type": "function", "function": {
        "name": "find_books_by_printer",
        "description": "List books attributed to a printer.",
        "parameters": {"type": "object", "properties": {
            "printer_slug": {"type": "string"},
            "year_min": {"type": "integer"},
            "year_max": {"type": "integer"},
        }, "required": ["printer_slug"]}}},
    {"type": "function", "function": {
        "name": "search_imprints",
        "description": "Hybrid dense+BM25 search over book titles, "
                       "imprints, and publishers.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 10},
            "printer_slug": {"type": "string"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_estc",
        "description": "Full bibliographic record for an ESTC ID.",
        "parameters": {"type": "object",
                       "properties": {"estc_id": {"type": "string"}},
                       "required": ["estc_id"]}}},
    {"type": "function", "function": {
        "name": "search_literature",
        "description": "Hybrid search over scholarly literature including "
                       "empirical baselines.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_printers",
        "description": "List printers in the corpus with stats.",
        "parameters": {"type": "object", "properties": {
            "min_glyphs": {"type": "integer", "default": 1},
        }}}},
    {"type": "function", "function": {
        "name": "get_glyph_metadata",
        "description": "Full record for one glyph.",
        "parameters": {"type": "object",
                       "properties": {"glyph_id": {"type": "string"}},
                       "required": ["glyph_id"]}}},
    {"type": "function", "function": {
        "name": "get_empirical_baselines",
        "description": "Validated performance baselines for this corpus.",
        "parameters": {"type": "object", "properties": {}}}},
]



# %% 11. AGENT — system prompt with aggressive tool chaining  ────────────────
SYSTEM_PROMPT = """You are an expert bibliographer investigating clandestine
and pseudonymous printers in Restoration England (1660-1700). Your job is to
attribute books with false imprints to their real shop using damaged-type
fingerprints from the Catalog of Distinctive Type (CDT).

You have three vector collections and 11 tools.

EMPIRICAL CALIBRATION (call get_empirical_baselines for details):
  - 81 books evaluated, honest recall@3 = 28%, chance = 16%, lift = 1.8x.
  - When the catalogued printer ranks #1, median cosine = 0.66.
  - When the catalogued printer does NOT rank #1, median cosine = 0.53.
  - A cosine of 0.6 has roughly 50/50 odds of being correct at book level.
  - Cosines >= 0.90 in book-level eval are usually LEAKAGE, not signal.

INVESTIGATIVE WORKFLOW — chain multiple tools before answering:

For BOOK-LEVEL attribution questions ("who really printed ESTC X?"):
  1. attribute_book(estc_id=X) — get ranking + confidence_band + leakage.
     The response includes a `cluster_ids_to_audit` field listing the top
     contributing cluster IDs verbatim (e.g. "A::0001", "C::0000").
  2. audit_cluster(cluster_id=Y) — call this 2-3 times, passing each cluster
     ID EXACTLY as it appeared in step 1's `cluster_ids_to_audit` field.
     NEVER invent a cluster ID. If you cannot find a cluster ID in a prior
     tool response, do not call audit_cluster.
  3. find_books_by_printer on the top candidate — does the candidate's
     known output match the book's year/subject?
  4. search_literature for the top candidate's known business relationships
  5. Synthesize a final answer that integrates ALL of these.

For PRINTER-LEVEL comparison questions ("are A and B related?"):
  1. compare_printer_fingerprints(A)
  2. compare_printer_fingerprints(B) if needed
  3. audit_cluster on shared clusters returned in step 1's response
  4. search_literature for known business overlaps
  5. Synthesize.

For CLUSTER questions ("is cluster X real evidence?"):
  1. audit_cluster(X) — primary
  2. find_similar_glyphs on a representative member (use a real glyph_id
     from the cluster's members, not a cluster_id)
  3. Synthesize.

CLUSTER ID FORMAT: cluster IDs always have the form '<CHARACTER>::<NUMBER>'
like 'A::0001', 'C::0000', 'W::0003'. They are NOT named 'C47' or 'C123'.
Only use cluster IDs that appeared verbatim in a prior tool response.

DO NOT stop after one tool call when the question deserves more investigation.
A single attribute_book call gives you a ranking but no validation of the
evidence. ALWAYS audit at least the top cluster contributor before issuing
a confident attribution.

CRITICAL CALIBRATION RULES (you MUST follow these):

1. When a tool returns confidence_band, report exactly that band. You may
   not upgrade or downgrade it based on how the evidence "feels."
2. When a tool returns leakage_suspect=true, you MUST note the attribution
   may be inflated and is not strong evidence.
3. Reserve "compelling," "robust," "strong evidence," "confirms" for
   confidence_band == "high" only. For moderate/low/low-to-moderate, use
   "consistent with," "plausible," "suggestive," "tentative."
4. When citing a cluster as evidence, prefer those that audit_cluster has
   verified as is_genuine_shared_sort=true. Note when a cluster is
   single-printer (NOT real evidence) or printer-dominated (weak evidence).

5. NEVER FABRICATE TOOL RESULTS. If you did not call search_literature in
   this conversation, do not write "the literature search reveals" or any
   similar phrase. If you did not call find_books_by_printer, do not
   describe the printer's "known output." Either call the tool first, or
   omit the claim. Inventing tool results to make an answer sound complete
   is a serious error.

6. Do not embellish title metadata. If a book title is truncated in the
   tool result, do not invent subject matter based on the prefix. State
   the title verbatim from the tool result.

FINAL ANSWER FORMAT:
  - Name the most likely real printer (or say "attribution inconclusive").
  - State the confidence_band from the tool result, verbatim.
  - Cite specific glyph IDs, ESTC numbers, and cluster IDs.
  - Note which clusters were audited and their verdicts.
  - Flag any leakage_suspect=true.
  - For moderate or below, name ONE concrete corroborating evidence type:
    imprint date, bookseller, subject matter, or woodcut ornaments.
  - When users ask point-attribution questions, remind them book-level
    recall@3 is only 28% — your answer is a narrowing, not a definitive
    identification.
"""


def _ollama_chat(messages, tools=None, timeout=600):
    body = {"model": AGENT_MODEL, "messages": messages,
            "stream": False, "options": {"temperature": AGENT_TEMPERATURE}}
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# %% 12. TRACE PERSISTENCE  ──────────────────────────────────────────────────
def _utc_timestamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _persist_trace(record):
    """Write one investigation to disk and update the running index."""
    ts = record["timestamp"]
    out_path = TRACE_DIR / f"{ts}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))
    # Update index
    idx_path = TRACE_DIR / "index.json"
    idx = []
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text())
        except Exception:
            idx = []
    idx.append({
        "timestamp": ts,
        "question": record["question"][:200],
        "n_steps": record["n_steps"],
        "n_tools_called": len(record["trace"]),
        "tools_used": sorted({t["tool"] for t in record["trace"]}),
        "encoder": ENCODER_TAG,
        "agent_model": AGENT_MODEL,
        "file": out_path.name,
    })
    idx_path.write_text(json.dumps(idx, indent=2))


def list_traces(limit=20):
    """List the most recent N traces."""
    idx_path = TRACE_DIR / "index.json"
    if not idx_path.exists():
        return []
    idx = json.loads(idx_path.read_text())
    return sorted(idx, key=lambda x: x["timestamp"], reverse=True)[:limit]


def load_trace(timestamp):
    """Reload a saved trace by timestamp."""
    p = TRACE_DIR / f"{timestamp}.json"
    return json.loads(p.read_text()) if p.exists() else None


def investigate(question: str, max_steps: int = AGENT_MAX_STEPS,
                 verbose: bool = True, persist: bool = True,
                 prior_messages: list = None):
    """Run the tool-using agent on a single question.

    If prior_messages is given, the investigation continues from that
    conversation history (so the agent has memory of previous turns).
    Otherwise starts fresh with [system_prompt, user_question].

    Writes the full trace to TRACE_DIR if persist=True. Returns a record
    dict including `messages_after` — the conversation state after this
    turn finishes, suitable for passing back as prior_messages on the
    next call."""
    timestamp = _utc_timestamp()
    if prior_messages:
        # Continue from existing conversation; append the new user turn
        messages = list(prior_messages)
        messages.append({"role": "user", "content": question})
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    trace = []
    start_time = time.time()
    verify_retries = 0  # how many times the verifier has asked for revisions

    for step in range(max_steps):
        if verbose:
            print(f"\n--- step {step + 1} ---")
        try:
            resp = _ollama_chat(messages, tools=TOOL_SCHEMAS)
        except Exception as e:
            answer = f"(agent error: {e})"
            record = {"timestamp": timestamp, "question": question,
                      "answer": answer, "trace": trace, "n_steps": step,
                      "elapsed_s": round(time.time() - start_time, 2),
                      "encoder": ENCODER_TAG, "agent_model": AGENT_MODEL,
                      "error": str(e), "messages_after": messages}
            if persist: _persist_trace(record)
            return record

        msg = resp.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        messages.append(msg)

        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception: args = {}
                if verbose:
                    print(f"  [tool] {name}({json.dumps(args)[:160]})")
                if name not in TOOLS:
                    result = {"error": f"unknown tool '{name}'"}
                else:
                    try:
                        result = TOOLS[name](**args)
                    except TypeError as e:
                        result = {"error": f"bad arguments: {e}"}
                    except Exception as e:
                        result = {"error": f"tool execution failed: {e}"}
                trace.append({"step": step + 1, "tool": name,
                              "args": args, "result": result})
                if verbose:
                    preview = json.dumps(result)[:280]
                    print(f"  [result] {preview}"
                          f"{'...' if len(preview) >= 280 else ''}")
                messages.append({"role": "tool",
                                 "content": json.dumps(result)[:16000]})
            continue

        if verbose:
            print(f"  [final answer]\n{content}")

        # Claim verification: detect tools the agent claims to have used
        # but didn't actually call. Common Qwen failure mode is asserting
        # "the literature search reveals..." when no search_literature
        # call appears in the trace.
        # Build the "prior messages haystack" for the verifiers — all
        # messages from earlier in THIS investigation (and the prior turns
        # of the conversation, if this investigate() call continued from
        # an earlier turn). The verifiers use these so they don't flag
        # entities/tools that came from earlier in the conversation.
        _verifier_prior = (prior_messages or [])
        unsupported_tools = _detect_unsupported_claims(
            content, trace, prior_messages=_verifier_prior)
        unsupported_entities = _detect_unsupported_entities(
            content, trace, prior_messages=_verifier_prior)
        needs_retry = (unsupported_tools or unsupported_entities)

        # Cap verification retries. If the agent has already been told
        # twice and hasn't fixed it, the verifier and the agent are
        # arguing about phrasing — accept the answer and move on. The
        # detected issues still get persisted to the trace for audit.
        MAX_VERIFY_RETRIES = 2

        if (needs_retry and step < max_steps - 1
                and verify_retries < MAX_VERIFY_RETRIES):
            verify_retries += 1
            if verbose:
                if unsupported_tools:
                    print(f"  [verify {verify_retries}/{MAX_VERIFY_RETRIES}] "
                          f"unsupported tool claims: {unsupported_tools}")
                if unsupported_entities:
                    print(f"  [verify {verify_retries}/{MAX_VERIFY_RETRIES}] "
                          f"unsupported entity references: "
                          f"{unsupported_entities}")
                print(f"  [verify] asking agent to revise...")
            tools_called = sorted({t["tool"] for t in trace})
            msg_parts = []
            if unsupported_tools:
                msg_parts.append(
                    f"Your answer references tools you did not actually "
                    f"call: {unsupported_tools}. The tools you actually "
                    f"called were: {tools_called}. Either call the missing "
                    f"tool now, or revise your answer to remove the "
                    f"unsupported claim."
                )
            if unsupported_entities:
                entity_summary = ", ".join(
                    f"{k}: {v}" for k, v in unsupported_entities.items())
                msg_parts.append(
                    f"Your answer cites entities (ESTC IDs / cluster IDs / "
                    f"glyph IDs / printer slugs) that DO NOT appear in any "
                    f"tool result: {entity_summary}. You may have invented "
                    f"these. Either call a tool to retrieve them (e.g. "
                    f"lookup_estc, get_glyph_metadata, search_imprints), "
                    f"or revise the answer to remove them. Do not cite "
                    f"entities you have not seen in a tool result."
                )
            messages.append({
                "role": "user",
                "content": " ".join(msg_parts) +
                           " Do not fabricate tool results or bibliographic "
                           "references.",
            })
            continue   # back to the agent loop for another try
        elif needs_retry and verbose:
            print(f"  [verify] giving up after {verify_retries} retries; "
                  f"accepting answer with flags: "
                  f"tools={unsupported_tools}, entities={unsupported_entities}")

        record = {"timestamp": timestamp, "question": question,
                  "answer": content, "trace": trace, "n_steps": step + 1,
                  "elapsed_s": round(time.time() - start_time, 2),
                  "encoder": ENCODER_TAG, "agent_model": AGENT_MODEL,
                  "unsupported_claims_detected": unsupported_tools,
                  "unsupported_entities_detected": unsupported_entities,
                  "messages_after": messages}
        if persist: _persist_trace(record)
        return record

    answer = "(max_steps reached without a final answer)"
    record = {"timestamp": timestamp, "question": question,
              "answer": answer, "trace": trace, "n_steps": max_steps,
              "elapsed_s": round(time.time() - start_time, 2),
              "encoder": ENCODER_TAG, "agent_model": AGENT_MODEL,
              "messages_after": messages}
    if persist: _persist_trace(record)
    return record


def _prune_messages(messages, max_keep_turns=3, max_chars_per_result=1500):
    """Truncate conversation history to keep context window manageable.

    Strategy: always keep the system prompt + the most recent N turns
    verbatim. For older turns, replace bulky tool results with a compact
    one-line summary. This preserves the conversational thread while
    shedding payload weight.

    A 'turn' here is a user→...→assistant cycle, possibly with tool calls
    in between.
    """
    if len(messages) <= 4:  # system + 1 turn, nothing to prune
        return messages

    # Identify turn boundaries by user role
    user_indices = [i for i, m in enumerate(messages)
                     if m.get("role") == "user"]
    if len(user_indices) <= max_keep_turns:
        return messages

    cutoff_idx = user_indices[-max_keep_turns]
    head = messages[:1]  # the system prompt
    older = messages[1:cutoff_idx]
    recent = messages[cutoff_idx:]

    # Compact the older turns: keep user questions and assistant answers
    # verbatim, but shrink tool messages.
    compacted = []
    for m in older:
        role = m.get("role")
        if role == "tool":
            content = m.get("content", "")
            if len(content) > max_chars_per_result:
                content = content[:max_chars_per_result] + "...(truncated)"
            compacted.append({"role": "tool",
                              "content": content,
                              "tool_call_id": m.get("tool_call_id", "")})
        else:
            compacted.append(m)
    return head + compacted + recent


def chat(initial_question=None, persist=True):
    """Interactive REPL for the agent. Multi-turn: each user question can
    reference previous tool results and answers.

    Type your question and hit enter. The agent will run an investigation
    (calling tools, verifying claims) and produce an answer. Then type
    your next question — the agent will have memory of what it already
    found.

    Special commands:
        /help              show command list
        /reset             clear conversation history and start fresh
        /history           show prior questions in this session
        /trace             show the full tool trace of the most recent turn
        /traces            list past investigations on disk
        /save <name>       save the current session to a named file
        /encoder <tag>     not supported mid-session — restart to switch
        /baselines         print loaded empirical baselines
        /quit, /exit, /q   end the session

    Use this for actual research: ask attribution questions, then follow
    up to drill into clusters, audit individual glyphs, search literature
    for context, and compare candidate printers. The conversation
    accumulates so the agent can build on what it already retrieved.
    """
    print("=" * 72)
    print(f"INTERACTIVE AGENT  (encoder={ENCODER_TAG}, model={AGENT_MODEL})")
    print("=" * 72)
    print(f"corpus: {len(index)} glyphs / "
          f"{len(set(printer_slugs))} printers / "
          f"{len(cluster_summary)} clusters")
    print(f"baseline: honest recall@3 = "
          f"{EMPIRICAL_BASELINES['honest_recall_at_3']:.0%}, "
          f"lift = {EMPIRICAL_BASELINES['lift_over_chance']}x")
    print(f"\nType your question to begin. Commands: /help /reset /trace "
          f"/history /save /baselines /quit")
    print(f"Tip: try follow-ups like 'audit the second cluster' or 'tell "
          f"me about the next-ranked printer' — the agent remembers context.")
    print()

    session = {
        "messages": None,         # conversation state across turns
        "last_record": None,      # most recent investigate() return
        "history": [],            # list of user questions
        "started_at": _utc_timestamp(),
    }

    def _handle_command(line):
        """Returns True if line was a command (and was handled)."""
        cmd = line.strip().lower().split()
        if not cmd:
            return True
        c = cmd[0]
        if c in ("/quit", "/exit", "/q"):
            print("Ending session. Saved trace files remain in", TRACE_DIR)
            return "exit"
        if c == "/help":
            print(chat.__doc__.split("Special commands:", 1)[1]
                                .split("Use this for")[0])
            return True
        if c == "/reset":
            session["messages"] = None
            session["last_record"] = None
            session["history"] = []
            print("[reset] conversation cleared.")
            return True
        if c == "/history":
            if not session["history"]:
                print("(no questions asked yet)")
            else:
                for i, q in enumerate(session["history"], 1):
                    print(f"  {i}. {q}")
            return True
        if c == "/trace":
            rec = session["last_record"]
            if not rec:
                print("(no investigation has run yet)")
                return True
            print(f"\nLast investigation: {rec['question']!r}")
            print(f"  {rec['n_steps']} steps, {len(rec['trace'])} tool calls, "
                  f"{rec['elapsed_s']}s")
            for i, t in enumerate(rec["trace"], 1):
                result_preview = str(t.get("result", ""))[:200]
                print(f"  [{i}] {t['tool']}({json.dumps(t.get('args', {}))})")
                print(f"      -> {result_preview}...")
            if rec.get("unsupported_claims_detected"):
                print(f"  flagged tool claims: "
                      f"{rec['unsupported_claims_detected']}")
            if rec.get("unsupported_entities_detected"):
                print(f"  flagged entities:    "
                      f"{rec['unsupported_entities_detected']}")
            return True
        if c == "/traces":
            traces = list_traces(limit=10)
            return True
        if c == "/save":
            name = cmd[1] if len(cmd) > 1 else f"session_{session['started_at']}"
            path = TRACE_DIR / f"{name}.json"
            path.write_text(json.dumps({
                "started_at": session["started_at"],
                "ended_at": _utc_timestamp(),
                "encoder": ENCODER_TAG,
                "history": session["history"],
                "last_messages": session["messages"],
                "last_record": session["last_record"],
            }, indent=2, default=str))
            print(f"[saved] {path}")
            return True
        if c == "/encoder":
            print("Encoder cannot be swapped mid-session — restart with "
                  "`python real_rag.py --encoder <tag> --chat`")
            return True
        if c == "/baselines":
            print(json.dumps(EMPIRICAL_BASELINES, indent=2, default=str))
            return True
        if c.startswith("/"):
            print(f"Unknown command: {c}. Try /help.")
            return True
        return False  # not a command — fall through to investigate

    # Optional first-question seed (for scripting; usually skipped)
    if initial_question:
        print(f"> {initial_question}")
        q = initial_question
    else:
        q = None

    while True:
        if q is None:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n(interrupted)")
                break
        if not q:
            q = None
            continue

        handled = _handle_command(q)
        if handled == "exit":
            break
        if handled:
            q = None
            continue

        # It's a real question — run an investigation
        session["history"].append(q)
        try:
            # Prune older context before invoking
            prior = (_prune_messages(session["messages"])
                     if session["messages"] else None)
            record = investigate(q, verbose=True, persist=persist,
                                  prior_messages=prior)
        except KeyboardInterrupt:
            print("\n(investigation interrupted)")
            q = None
            continue
        except Exception as e:
            print(f"\n[error] {e}")
            q = None
            continue

        # Carry the new conversation state forward for the next turn
        session["messages"] = record.get("messages_after")
        session["last_record"] = record

        print(f"\n{'─' * 72}")
        print(f"ANSWER:")
        print(record["answer"])
        print(f"\n({record['n_steps']} steps, {len(record['trace'])} "
              f"tool calls, {record['elapsed_s']}s — trace: "
              f"{record['timestamp']}.json)")
        # Reset for next iteration
        q = None

    return session


    """Detect language in the final answer that ASSERTS tool use without a
    matching entry in the trace.

    Only fires on assertive claims (e.g. "the literature search reveals X"),
    not on hedged/hypothetical phrasings (e.g. "we could cross-reference
    with known output"). This avoids the verifier arguing forever with the
    agent over hedged statements.

    Returns list of tool names the agent appears to have falsely claimed."""
    tools_called = {t["tool"] for t in trace}
    text_lower = answer_text.lower()

    # Assertive patterns: indicative tense, definite article, present-tense
    # verbs of finding/showing. Hedged forms ("could", "would", "can be",
    # "may help") are deliberately excluded.
    claim_patterns = {
        "search_literature": [
            "the literature search reveals", "literature search reveals",
            "the literature reveals", "scholarly literature shows",
            "according to the literature",
        ],
        "find_books_by_printer": [
            "the known output includes", "known output of",
            "their other works include", "the printer's other books",
            "books printed by this printer include",
        ],
        "search_imprints": [
            "the imprint search returned", "the search of imprints found",
        ],
        "lookup_estc": [
            "the estc record shows", "according to estc",
            "the estc record states",
        ],
    }

    unsupported = []
    for tool_name, patterns in claim_patterns.items():
        if tool_name in tools_called:
            continue
        for pat in patterns:
            if pat in text_lower:
                unsupported.append(tool_name)
                break
    return unsupported


def _detect_unsupported_claims(answer_text, trace, prior_messages=None):
    """Detect language in the final answer that ASSERTS tool use without a
    matching entry in the trace.

    In multi-turn conversations, also accepts the prior_messages list so
    that tools called in earlier turns count as "actually called" — a
    legitimate user follow-up that synthesises across turns should not
    be flagged just because no NEW tool was called this turn.

    Only fires on assertive claims (e.g. "the literature search reveals X"),
    not on hedged/hypothetical phrasings.

    Returns list of tool names the agent appears to have falsely claimed."""
    tools_called = {t["tool"] for t in trace}

    # In multi-turn: also count tools called in earlier turns. We find
    # them by scanning the prior_messages for assistant turns with
    # tool_calls attached.
    if prior_messages:
        for m in prior_messages:
            for call in m.get("tool_calls", []) or []:
                name = (call.get("function", {}).get("name")
                        if isinstance(call, dict) else None)
                if name:
                    tools_called.add(name)
    text_lower = answer_text.lower()

    claim_patterns = {
        "search_literature": [
            "the literature search reveals", "literature search reveals",
            "the literature reveals", "scholarly literature shows",
            "according to the literature",
        ],
        "find_books_by_printer": [
            "the known output includes", "known output of",
            "their other works include", "the printer's other books",
            "books printed by this printer include",
        ],
        "search_imprints": [
            "the imprint search returned", "the search of imprints found",
        ],
        "lookup_estc": [
            "the estc record shows", "according to estc",
            "the estc record states",
        ],
    }

    unsupported = []
    for tool_name, patterns in claim_patterns.items():
        if tool_name in tools_called:
            continue
        for pat in patterns:
            if pat in text_lower:
                unsupported.append(tool_name)
                break
    return unsupported


def _flatten_to_text(obj):
    """Recursively flatten a dict/list/scalar to one big text blob.
    Used to build a searchable haystack of everything the tools returned."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)) or obj is None:
        return str(obj)
    if isinstance(obj, dict):
        return " ".join(_flatten_to_text(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten_to_text(v) for v in obj)
    return str(obj)


# Entity patterns we expect agents to cite. Each regex extracts entities
# the agent might reference in its final answer; the verifier then checks
# whether each extracted entity actually appears in the trace's results.
_ENTITY_PATTERNS = {
    "estc_id": re.compile(r"\b[RT]\d{3,7}\b"),
    "cluster_id": re.compile(r"\b[A-Z]::\d{4}\b"),
    "glyph_id": re.compile(r"\b[A-Z][a-z]+_[A-Z]\d{3,5}\.\d{3}\b"),
    "printer_slug": re.compile(r"\b[a-z]+_[a-z]+\b"),
}

# Printer slugs we know exist in the corpus (populated lazily on first call).
_KNOWN_PRINTER_SLUGS = None


def _detect_unsupported_entities(answer_text, trace, prior_messages=None):
    """Scan the final answer for entity strings (ESTC IDs, cluster IDs,
    glyph IDs, printer slugs) and check that each one appears somewhere in
    the trace's tool results.

    In multi-turn conversations, also accepts prior_messages so entities
    that came from tools called in EARLIER turns still count as supported.
    Without this, a synthesis question in turn N would incorrectly flag
    every entity from turns 1..N-1.

    Returns a dict: {entity_type: [list of unsupported entities]}.
    Empty dict means everything in the answer is grounded.
    """
    global _KNOWN_PRINTER_SLUGS
    if _KNOWN_PRINTER_SLUGS is None:
        try:
            _KNOWN_PRINTER_SLUGS = set(printer_slugs)
        except NameError:
            _KNOWN_PRINTER_SLUGS = set()

    # Build haystack from CURRENT turn's tool calls
    haystack_parts = []
    for t in (trace or []):
        haystack_parts.append(_flatten_to_text(t.get("args", {})))
        haystack_parts.append(_flatten_to_text(t.get("result", {})))

    # Also include text from earlier turns of the conversation: tool
    # results, user questions (the user may have given a legit entity),
    # and prior assistant turns (the agent's own prior answers contain
    # entities it had supported then).
    if prior_messages:
        for m in prior_messages:
            content = m.get("content")
            if content:
                haystack_parts.append(str(content))
            # Tool call arguments
            for call in m.get("tool_calls", []) or []:
                if isinstance(call, dict):
                    fn = call.get("function", {})
                    args = fn.get("arguments")
                    if args:
                        haystack_parts.append(str(args))

    if not haystack_parts:
        return {}
    haystack = " ".join(haystack_parts).lower()

    unsupported = {}
    for kind, pattern in _ENTITY_PATTERNS.items():
        found = set(pattern.findall(answer_text))
        if not found:
            continue
        if kind == "printer_slug":
            found = {s for s in found if s in _KNOWN_PRINTER_SLUGS}
        missing = sorted(s for s in found if s.lower() not in haystack)
        if missing:
            unsupported[kind] = missing
    return unsupported


# %% 13. LITERATURE INGESTION  ────────────────────────────────────────────────
def add_literature(text: str, source: str, year: int = None):
    """Add a new scholarly excerpt to the literature collection."""
    vec = embed_text(text)[0].astype(np.float32)
    literature_tbl.add([{
        "vector": vec, "source": source,
        "year": str(year) if year else "", "text": text,
    }])
    print(f"[lit] added excerpt from '{source}' ({len(text)} chars)")
    try:
        literature_tbl.create_fts_index("text", replace=True)
    except Exception as e:
        print(f"  (FTS reindex failed: {e})")


# %% 14. DEMO  ────────────────────────────────────────────────────────────────
# Two demo modes:
#   default              -> one short attribution query (math-only path,
#                           does not exercise LanceDB)
#   --demo-suite         -> full suite that exercises all three vector
#                           collections: glyphs, books, literature
#
# Each suite query is labeled with which LanceDB collection it primarily
# exercises. Read the trace JSON afterwards to see exactly which tool calls
# ran. Some queries hit multiple collections; the label is the dominant one.

DEMO_SUITE = [
    {
        "label": "attribution (math-only — no LanceDB)",
        "exercises": ["attribute_book", "audit_cluster",
                       "find_books_by_printer"],
        "lancedb_collections_touched": [],
        "question": (
            "Who is the most likely real printer of ESTC R21865? Treat the "
            "imprint as unknown. Use book-level held-out attribution AND "
            "audit the top contributing clusters. Note any leakage or "
            "low-confidence flags."
        ),
    },
    {
        "label": "GLYPHS collection — vector search",
        "exercises": ["find_similar_glyphs", "get_glyph_metadata"],
        "lancedb_collections_touched": ["glyphs"],
        "question": (
            "Find the 10 glyphs most visually similar to glyph "
            "'ibbitson_robert_A_0001' (use find_similar_glyphs). Then look "
            "up each of the top 3 results with get_glyph_metadata to see "
            "which printers and books they come from. Report whether the "
            "similar glyphs cluster around a single shop or are spread "
            "across many."
        ),
    },
    {
        "label": "BOOKS collection — hybrid dense + BM25 search",
        "exercises": ["search_imprints", "lookup_estc"],
        "lancedb_collections_touched": ["books"],
        "question": (
            "Search the books collection for imprints that suggest "
            "nonconformist religious content (use search_imprints with a "
            "natural-language query). Pick the most promising result and "
            "use lookup_estc to retrieve its full bibliographic record. "
            "Do NOT run attribution analysis — this query is about "
            "metadata discovery."
        ),
    },
    {
        "label": "LITERATURE collection — semantic context lookup",
        "exercises": ["search_literature", "get_empirical_baselines"],
        "lancedb_collections_touched": ["literature"],
        "question": (
            "Search the scholarly literature for what is known about "
            "sort-sharing between Restoration printers. Also retrieve the "
            "empirical baselines for this corpus. Summarise what the "
            "literature says about how reliable damaged-sort attribution "
            "is in general. Do NOT attempt to attribute a specific book."
        ),
    },
    {
        "label": "mixed — glyphs + literature + math",
        "exercises": ["compare_printer_fingerprints", "audit_cluster",
                       "search_literature"],
        "lancedb_collections_touched": ["literature"],
        "question": (
            "Are Tyler and Simmons plausibly the same shop, or did they "
            "share type? Run compare_printer_fingerprints on tyler_evan, "
            "audit the top 2 contributing clusters, then search literature "
            "for what's known about their business relationship. "
            "Synthesise a calibrated answer."
        ),
    },
    {
        "label": "BOOKS collection — filtered search",
        "exercises": ["search_imprints", "list_printers"],
        "lancedb_collections_touched": ["books"],
        "question": (
            "List the printers in this corpus with at least 50 glyphs and "
            "their date ranges. Then search the books collection for any "
            "books printed around 1685 that mention astrology or almanacs "
            "in the title. Report which printers in the corpus produced "
            "books matching that description."
        ),
    },
]


def run_demo_suite():
    """Run all demo queries in sequence, showing which LanceDB collections
    each one exercises."""
    # Pick a real glyph ID for the GLYPHS demo so the agent has something
    # valid to query. Use the first glyph from the most populous printer.
    populous_printers = Counter(printer_slugs).most_common(1)
    if populous_printers:
        top_printer = populous_printers[0][0]
        real_glyph_id = next((r.id for r in index.records
                               if r.printer_slug == top_printer), None)
    else:
        real_glyph_id = index.records[0].id if index.records else "unknown"

    # Materialise the suite with the real glyph ID substituted in
    suite = []
    for demo in DEMO_SUITE:
        d = dict(demo)
        d["question"] = d["question"].replace(
            "ibbitson_robert_A_0001", real_glyph_id)
        suite.append(d)

    print("\n" + "=" * 72)
    print(f"DEMO SUITE: {len(suite)} queries exercising the full RAG stack")
    print(f"(using glyph_id={real_glyph_id} for the glyphs-collection demo)")
    print("=" * 72)
    summary = []
    for i, demo in enumerate(suite, 1):
        print(f"\n[{i}/{len(suite)}] {demo['label']}")
        if demo["lancedb_collections_touched"]:
            print(f"  LanceDB collections: "
                  f"{', '.join(demo['lancedb_collections_touched'])}")
        else:
            print(f"  LanceDB collections: (none — math-only path)")
        print(f"  Expected tools: {', '.join(demo['exercises'])}")
        print(f"\nQUESTION: {demo['question']}\n")

        out = investigate(demo["question"], verbose=True)

        # Audit: did the agent actually call the expected tools?
        called = sorted({t["tool"] for t in out["trace"]})
        expected = set(demo["exercises"])
        n_expected_called = len(set(called) & expected)
        coverage_frac = (n_expected_called / len(expected)
                         if expected else 1.0)
        unused_expected = expected - set(called)
        unexpected_called = set(called) - expected

        print(f"\nFINAL ANSWER:\n{out['answer']}")
        print(f"\n--- Audit ---")
        print(f"  tools called:        {called}")
        print(f"  expected tools used: {n_expected_called}/{len(expected)} "
              f"({coverage_frac:.0%})")
        if unused_expected:
            print(f"  expected but unused: {sorted(unused_expected)}")
        if unexpected_called:
            print(f"  called but unexpected: {sorted(unexpected_called)}")
        print(f"  steps: {out['n_steps']}, calls: {len(out['trace'])}, "
              f"time: {out['elapsed_s']}s")
        print(f"  trace: {TRACE_DIR}/{out['timestamp']}.json")
        print("=" * 72)

        summary.append({
            "label": demo["label"],
            "expected_tools": sorted(expected),
            "called_tools": called,
            "n_steps": out["n_steps"],
            "elapsed_s": out["elapsed_s"],
            "answer_preview": out["answer"][:200],
            "trace_file": f"{out['timestamp']}.json",
        })

    print("\n" + "=" * 72)
    print("SUITE SUMMARY")
    print("=" * 72)
    for s in summary:
        print(f"\n[{s['label']}]")
        print(f"  expected: {s['expected_tools']}")
        print(f"  called:   {s['called_tools']}")
        print(f"  {s['n_steps']} steps, {s['elapsed_s']}s, "
              f"trace={s['trace_file']}")
    print()
    print("All traces saved to:", TRACE_DIR)
    return summary


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("REAL RAG: tool-using bibliographic agent (calibrated)")
    print("=" * 72)
    print(f"corpus:   {len(index)} glyphs, "
          f"{len(set(printer_slugs))} printers, "
          f"{len(cluster_summary)} clusters")
    print(f"encoder:  {ENCODER_TAG}  "
          f"(embed_dim={embeddings.shape[1]})")
    print(f"agent:    {AGENT_MODEL} via Ollama")
    print(f"baseline: honest recall@3 = "
          f"{EMPIRICAL_BASELINES['honest_recall_at_3']:.0%}, "
          f"lift = {EMPIRICAL_BASELINES['lift_over_chance']}x  "
          f"[{EMPIRICAL_BASELINES.get('source', 'unknown')}]")
    print(f"traces:   {TRACE_DIR}")
    print()

    # CLI flags
    if "--evaluate" in sys.argv:
        # Run full recall@k validation across all books with the current
        # encoder. Compare DINOv2 vs contrastive results afterwards with
        # --compare. Takes ~3-5 min.
        evaluate_encoder(k=3, verbose=True)
    elif "--ablate-evidence" in sys.argv:
        # Module A ablation: run multi-evidence attribution under each
        # leave-one-stream-out config to measure each stream's contribution.
        # Takes ~5-10 min (calibration + 5 evaluation passes).
        ablate_evidence_streams(k=3, verbose=True)
    elif "--multi-eval" in sys.argv:
        # Single run of multi-evidence attribution (all 4 streams).
        evaluate_multi_evidence(k=3, verbose=True)
    elif "--active-eval" in sys.argv:
        # Module B: active evidence acquisition. Iteratively picks the
        # most informative stream to run next, stops at threshold.
        # Reports recall AND cost savings.
        #
        # --clandestine masks the true printer's name from the publisher
        # field before running attribution. Use this for the real test —
        # without it, the imprint stream trivially reads the answer off
        # the publisher field.
        thresh = 0.80
        for i, arg in enumerate(sys.argv):
            if arg == "--threshold" and i + 1 < len(sys.argv):
                try:
                    thresh = float(sys.argv[i + 1])
                except ValueError:
                    pass
        is_clandestine = "--clandestine" in sys.argv
        is_strict = "--strict-clandestine" in sys.argv
        is_strictest = "--strictest-clandestine" in sys.argv
        evaluate_active(k=3, stop_threshold=thresh,
                         clandestine=is_clandestine,
                         strict_clandestine=is_strict,
                         strictest_clandestine=is_strictest, verbose=True)
    elif "--active-report" in sys.argv:
        # Full honest report: open + clandestine + stratified diagnostic
        # in one run, with comparison table. This is the version to use
        # for any writeup. Takes ~15-20 min.
        thresh = 0.80
        for i, arg in enumerate(sys.argv):
            if arg == "--threshold" and i + 1 < len(sys.argv):
                try:
                    thresh = float(sys.argv[i + 1])
                except ValueError:
                    pass
        evaluate_active_report(k=3, stop_threshold=thresh, verbose=True)
    elif "--active-stratified" in sys.argv:
        # Just the imprint-leaks / imprint-clean split (no clandestine).
        # Faster than --active-report. ~5-10 min.
        thresh = 0.80
        for i, arg in enumerate(sys.argv):
            if arg == "--threshold" and i + 1 < len(sys.argv):
                try:
                    thresh = float(sys.argv[i + 1])
                except ValueError:
                    pass
        evaluate_active_stratified(k=3, stop_threshold=thresh, verbose=True)
    elif "--printer-holdout" in sys.argv:
        # House-style stress test: for each book, hold out ALL of its
        # printer's books from the cluster space, then rank what remains.
        # If top cosines collapse vs the standard eval, the encoder is
        # relying on printer house style. Takes ~3-5 min.
        evaluate_printer_holdout(k=3, verbose=True)
    elif "--compare" in sys.argv:
        # Print side-by-side comparison if both encoders have been evaluated
        compare_encoders_offline(k=3)
    elif "--demo-suite" in sys.argv:
        run_demo_suite()
    elif "--chat" in sys.argv:
        # Interactive multi-turn agent REPL. Each turn maintains
        # conversation state so follow-up questions can reference
        # earlier tool results.
        chat()
    else:
        demo = DEMO_SUITE[0]
        print(f"Running single demo: {demo['label']}")
        print(f"Available flags:")
        print(f"  --chat                interactive multi-turn REPL with "
              f"the agent")
        print(f"  --demo-suite          run all {len(DEMO_SUITE)} demos")
        print(f"  --evaluate            run recall@3 validation with "
              f"current encoder")
        print(f"  --multi-eval          Module A: multi-evidence Bayesian "
              f"attribution evaluation (all 4 streams)")
        print(f"  --active-eval         Module B: active evidence "
              f"acquisition — pick streams adaptively, stop at threshold")
        print(f"  --clandestine         (use with --active-eval) mask the "
              f"true printer's name from the publisher field — required "
              f"for an honest attribution test")
        print(f"  --strict-clandestine  (use with --active-eval) ALSO mask "
              f"the printer's known booksellers — the truly honest test")
        print(f"  --strictest-clandestine  also suppress the year — the "
              f"floor case (vision-only, false-imprint scenario)")
        print(f"  --active-stratified   stratified diagnostic: recall on "
              f"imprint-clean vs imprint-leaks subsets")
        print(f"  --active-report       full honest report: open + "
              f"clandestine + stratified, with comparison table")
        print(f"  --threshold <p>       stop threshold for --active-eval "
              f"(default 0.80)")
        print(f"  --clandestine         mask true printer from publisher "
              f"field (the real test for clandestine attribution)")
        print(f"  --ablate-evidence     Module A ablation: how much does "
              f"each evidence stream contribute?")
        print(f"  --printer-holdout     house-style stress test (hold out "
              f"entire printer, not just one book)")
        print(f"  --compare             compare DINOv2 vs contrastive "
              f"(needs both --evaluate runs)")
        print(f"  --encoder dinov2|contrastive   override the encoder "
              f"choice from config")
        print(f"  --retrain             force contrastive retraining "
              f"with current hyperparameters\n")
        print(f"QUESTION: {demo['question']}\n")
        out = investigate(demo["question"], verbose=True)
        print(f"\nFINAL ANSWER:\n{out['answer']}")
        print(f"\n({out['n_steps']} steps, {len(out['trace'])} tool calls, "
              f"{out['elapsed_s']}s, trace saved to "
              f"{TRACE_DIR}/{out['timestamp']}.json)")
        print("=" * 72)
