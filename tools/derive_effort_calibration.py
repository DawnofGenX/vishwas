#!/usr/bin/env python3
"""Re-derive the EFFORT affine calibration after the BGR->RGB fix.

WHY
---
The fusion layer's `_CALIBRATE["effort.prob"] = (1.0, -0.55)` was fitted
against the BROKEN score distribution (BGR input, ImageNet normalisation).
The fix moves the scores: on the leak-free corpus the real median fell from
0.6326 to 0.4762 and the fake median from 0.7707 to 0.6190.

A -0.55 offset fitted to the old distribution would now flatten the
separation to nothing (both classes pushed to ~0), so the calibration MUST
be re-derived -- but ON THE SAME PRINCIPLE as before: the offset exists so
that genuine media is not pushed into the DO_NOT_USE band, while a real
deepfake still clears it.

This script sweeps the offset and the weight and reports the resulting
verdict mix through the REAL FusionEngine, so the choice is measured.

The score used here is the PRODUCTION score (after the fix), recomputed
live on the leak-free corpus.

Usage: python tools/derive_effort_calibration.py
"""
import json
import os
import pathlib
import shutil
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict
from vishwas.events import Artifact, JobContext
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE
from vishwas.media_utils import extract_frames
from vishwas.model_adapters import resolve as _resolve_adapter
from vishwas.capabilities.deepfake_video import _read_bgr

CORPUS = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_CORPUS",
    "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v2.jsonl"))
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_cal_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_cal_rows.jsonl")


def auc(pos, neg):
    s = n = 0
    for p in pos:
        for q in neg:
            s += 1 if p > q else (0.5 if p == q else 0)
            n += 1
    return s / n


if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
adapter = _resolve_adapter("VISHWAS_EFFORT_WEIGHTS")
model = adapter.load(os.environ["VISHWAS_EFFORT_WEIGHTS"])
net = model.model if hasattr(model, "model") else model
net.eval()
device = next(net.parameters()).device

from vishwas.model_archs.effort import _IMAGENET_MEAN, _IMAGENET_STD


def score_chw_bgr(chw):
    """The PRODUCTION score: BGR in, RGB swap inside, CLIP-normalise.

    `chw` is (3, H, W) as `_img_resize_chw` produces; it is promoted to
    (1, 3, H, W) exactly as `EffortSpec.score` does.
    """
    a = np.asarray(chw, dtype=np.float32)
    if a.ndim == 3:
        a = a[None]
    a = a[:, ::-1, :, :].copy()
    a = (a - _IMAGENET_MEAN[None, :, None, None]) / _IMAGENET_STD[None, :, None, None]
    xt = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    with torch.no_grad():
        return float(F.softmax(net(xt), dim=-1)[:, 1].mean().item())


def resize_chw(arr, size=224):
    import cv2
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = arr[y0:y0 + s, x0:x0 + s]
    resized = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
    return np.transpose(resized, (2, 0, 1))


scored = []
for i, r in enumerate(rows):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(r["path"])
    if not p.exists():
        continue
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"cal{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)
    probs = []
    for fp in frames[:8]:
        img = _read_bgr(fp)
        if img is None:
            continue
        probs.append(score_chw_bgr(resize_chw(img)))
    if not probs:
        continue
    import statistics as st
    rec = {"path": str(p), "label": r["label"], "effort": st.median(probs)}
    scored.append(rec)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")

reals = [r["effort"] for r in scored if r["label"] == 0]
fakes = [r["effort"] for r in scored if r["label"] == 1]
print(f"scored {len(scored)} clips")
print(f"  NEW real median = {np.median(reals):.4f}   (was 0.6326 before the fix)")
print(f"  NEW fake median = {np.median(fakes):.4f}   (was 0.7707 before the fix)")
print(f"  AUC             = {auc(fakes, reals):.4f}  (was 0.6511)")
print()
print(f"  -> the OLD offset -0.55 now maps the real median to "
      f"{max(0.0, np.median(reals)-0.55):.4f} and the fake median to "
      f"{max(0.0, np.median(fakes)-0.55):.4f}")
print("     -> the separation is nearly gone. The offset MUST be re-derived.")
print()

# --- sweep the offset through the real fusion engine -----------------------
def evaluate(offset, weight=1.5):
    base_w = dict(WEIGHTS["deepfake_video"])
    base_c = dict(_CALIBRATE)
    WEIGHTS["deepfake_video"] = {**base_w, "effort.prob": weight}
    _CALIBRATE["effort.prob"] = (1.0, offset)
    try:
        fe = FusionEngine()
        res = []
        for r in scored:
            checks = [
                C("video_probe", "cheap", "ok", {"duration_s": 1.0}, "m"),
                C("usable_frames", "cheap", "ok", {"usable_frame_ratio": 1.0}, "m"),
                C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.15}, "m"),
                C("effort_face_forensics", "heavy", "ok",
                  {"prob_deepfake": r["effort"]}, "m"),
                C("cross_modal_av", "heavy", "ok",
                  {"av_risk_addition": 0.0, "alignment_class": "synced",
                   "av_correlation": 0.4, "av_synced_clean": 0.4}, "m"),
                C("havic_crossmodal_model", "heavy", "ok",
                  {"prob_inconsistent": 0.5}, "m"),
            ]
            d = fe.decide("deepfake_video", checks)
            res.append((r["label"], d.verdict.value))
    finally:
        WEIGHTS["deepfake_video"] = base_w
        _CALIBRATE.clear()
        _CALIBRATE.update(base_c)
    real = [v for l, v in res if l == 0]
    fake = [v for l, v in res if l == 1]
    return {
        "fp": sum(1 for v in real if v == "do_not_use"),
        "fn": sum(1 for v in fake if v == "trust"),
        "dnu_fake": sum(1 for v in fake if v == "do_not_use"),
        "trust_real": sum(1 for v in real if v == "trust"),
        "n_real": len(real), "n_fake": len(fake),
    }


print("=" * 92)
print("OFFSET SWEEP (effort weight 1.5) -- measured through the real FusionEngine")
print("=" * 92)
print(f"{'offset':>8} {'FP':>4} {'FN':>4} {'fakeDNU':>9} {'realTRUST':>10}")
for off in (-0.55, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0):
    m = evaluate(off)
    flag = ""
    if m["fp"] == 0 and m["fn"] == 0:
        flag = "  <-- no FP, no FN"
    print(f"{off:>8.2f} {m['fp']:>4} {m['fn']:>4} {m['dnu_fake']:>9} "
          f"{m['trust_real']:>10}{flag}")

print()
print("NOTE: this sweep holds the cross-modal signals SYNCHRONISED and benign")
print("(av=0.0, havic=0.5), so it isolates the effort contribution. The real")
print("corpus rows have their own av/havic values, so the production numbers")
print("differ; this is a CALIBRATION sweep, not a corpus evaluation.")
print()
print(f"wrote {len(scored)} scored rows -> {OUT}")
