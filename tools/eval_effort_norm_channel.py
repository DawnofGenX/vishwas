#!/usr/bin/env python3
"""MEASURE all four (channel order x normalisation) combinations for EFFORT.

Two independent defects are suspected in the EFFORT path:

  1. CHANNEL ORDER. The video path does cv2.imread() -> BGR, and
     _img_resize_chw does NOT convert to RGB. EffortSpec.score() then
     ImageNet-normalises with RGB-ordered statistics. EFFORT is a CLIP
     ViT-L/14, trained on RGB. The sibling SPAI spec DOES swap
     ("SPAI was trained on PIL/RGB" -- model_archs/spai.py).

  2. NORMALISATION CONSTANTS. The official EFFORT implementation
     (DeepfakeBench/training/detectors/effort_detector.py) quotes
         mean: [0.48145466, 0.4578275, 0.40821073]
         std:  [0.26862954, 0.26130258, 0.27577711]
     which are CLIP's constants. The repo uses ImageNet's
         mean: [0.485, 0.456, 0.406]
         std:  [0.229, 0.224, 0.225]

This script scores every clip under all four combinations and reports the
AUC/EER for each, so the fix is chosen by MEASUREMENT rather than by
argument.

The scoring is done by calling the model DIRECTLY with the same tensor
shapes the production path uses, so only the two variables change.

Usage: python tools/eval_effort_norm_channel.py
"""
import json
import os
import pathlib
import shutil
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from vishwas.events import Artifact, JobContext
from vishwas.media_utils import extract_frames
from vishwas.model_adapters import resolve as _resolve_adapter
from vishwas.capabilities.deepfake_video import _read_bgr

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CORPUS = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_CORPUS",
    "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v2.jsonl"))
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_nc_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_norm_channel_rows.jsonl")


def auc(pos, neg):
    s = n = 0
    for p in pos:
        for q in neg:
            s += 1 if p > q else (0.5 if p == q else 0)
            n += 1
    return s / n


def eer(pos, neg):
    ths = sorted(set([round(v, 4) for v in pos + neg]))
    best = (1.0, None)
    for t in ths:
        miss = sum(1 for p in pos if p >= t) / len(pos)
        fa = sum(1 for q in neg if q < t) / len(neg)
        e = (miss + fa) / 2
        if e < best[0]:
            best = (e, t)
    return best


def resize_chw(arr, size=224):
    """Exactly the production _img_resize_chw geometry."""
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = arr[y0:y0 + s, x0:x0 + s]
    import cv2
    resized = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
    return np.transpose(resized, (2, 0, 1))


if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
print(f"corpus: {len(rows)} rows")

adapter = _resolve_adapter("VISHWAS_EFFORT_WEIGHTS")
model = adapter.load(os.environ["VISHWAS_EFFORT_WEIGHTS"])
net = model.model if hasattr(model, "model") else model
net.eval()
device = next(net.parameters()).device


def score_chw(chw, mean, std):
    arr = (chw - mean[None, :, None, None]) / std[None, :, None, None]
    xt = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    with torch.no_grad():
        logits = net(xt)
    return float(F.softmax(logits, dim=-1)[:, 1].mean().item())


out_rows = []
for i, r in enumerate(rows):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(r["path"])
    if not p.exists():
        continue
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"nc{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)

    buckets = {k: [] for k in ("bgr_imnet", "bgr_clip", "rgb_imnet", "rgb_clip")}
    for fp in frames[:8]:
        img = _read_bgr(fp)
        if img is None:
            continue
        bgr = resize_chw(img)
        rgb = resize_chw(img[:, :, ::-1].copy())
        buckets["bgr_imnet"].append(score_chw(bgr, IMAGENET_MEAN, IMAGENET_STD))
        buckets["bgr_clip"].append(score_chw(bgr, CLIP_MEAN, CLIP_STD))
        buckets["rgb_imnet"].append(score_chw(rgb, IMAGENET_MEAN, IMAGENET_STD))
        buckets["rgb_clip"].append(score_chw(rgb, CLIP_MEAN, CLIP_STD))

    rec = {"path": str(p), "label": r["label"],
           **{k: (st.median(v) if v else None) for k, v in buckets.items()}}
    out_rows.append(rec)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {p.name[:22]:<24s} lab={r['label']}  " +
          "  ".join(f"{k}={rec[k]:.3f}" for k in buckets))

print()
print("=" * 96)
print("EFFORT: all four (channel order x normalisation) combinations")
print("=" * 96)
results = []
for k in ("bgr_imnet", "bgr_clip", "rgb_imnet", "rgb_clip"):
    reals = [r[k] for r in out_rows if r["label"] == 0 and r[k] is not None]
    fakes = [r[k] for r in out_rows if r["label"] == 1 and r[k] is not None]
    a = auc(fakes, reals)
    e = eer(fakes, reals)
    results.append((a, k, e[0], st.median(reals), st.median(fakes)))
    tag = "  <- REPO TODAY" if k == "bgr_imnet" else (
        "  <- UPSTREAM (correct)" if k == "rgb_clip" else "")
    print(f"  {k:<11s} AUC={a:.4f}  EER={e[0]*100:5.2f}%  "
          f"real med={st.median(reals):.4f}  fake med={st.median(fakes):.4f}{tag}")

results.sort(reverse=True)
print()
print(f"  BEST: {results[0][1]}  AUC={results[0][0]:.4f}")
print(f"  repo today (bgr_imnet) AUC="
      f"{[r for r in results if r[1]=='bgr_imnet'][0][0]:.4f}")
print(f"  gain from fixing both: "
      f"{results[0][0] - [r for r in results if r[1]=='bgr_imnet'][0][0]:+.4f} AUC")
print()
print(f"  wrote {len(out_rows)} rows -> {OUT}")
