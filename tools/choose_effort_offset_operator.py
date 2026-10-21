#!/usr/bin/env python3
"""Choose the EFFORT offset against the DESIGN INTENT, using the operator's
OWN media rather than a synthetic corpus.

WHY THIS TOOL EXISTS
---------------------
The BGR->RGB fix moves the EFFORT scores down by ~0.15 (real median
0.6326 -> 0.4762). The fusion offset -0.55 was fitted to the OLD
distribution, so after the fix it clips both classes to ~0 and 18 of 30
fakes read TRUST. The offset must be re-derived.

BUT the leak-free synthetic corpus is the WRONG instrument for choosing it:
it contains NO anti-correlated rows (every av is 0.0/0.1/0.45, because a
muxed audio track decorrelates but does not anti-correlate). The
operator's own AI videos all read av=0.5, and the design intent recorded
in the fusion layer is:

    "effort cal b -0.45 -> -0.55 ... maps that same real distribution to
     ~0.17-0.42 (CAUTION band) while the operator AI video (0.677 raw ->
     0.23 calibrated) still reaches DO_NOT_USE via the AV corroboration
     path (av_risk + havic), NOT effort alone."

So the offset's JOB is:
    (a) genuine media must NOT read DO_NOT_USE on effort alone;
    (b) the operator's AI videos must still read DO_NOT_USE, via the
        cross-modal corroboration rather than via effort.

This tool scores the operator's 4 genuine natural-audio clips and 3 AI
anchor videos with the FIXED model, sweeps the offset, and reports both
(a) and (b).

Usage: python tools/choose_effort_offset_operator.py
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
from vishwas.capabilities.deepfake_video import _read_bgr
from vishwas.events import Artifact, JobContext
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE
from vishwas.media_utils import extract_frames
from vishwas.model_adapters import resolve as _resolve_adapter
from vishwas.model_archs.effort import _IMAGENET_MEAN, _IMAGENET_STD

QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_op_q")
OPERATOR_REALS = sorted(pathlib.Path(
    "/home/hermes/fusion_av/trackA_operator_reals").glob("*.mp4"))
AI_ANCHORS = sorted(pathlib.Path(
    "/home/hermes/fusion_av/seeded").glob("ai_*.mp4"))
CONTROLS = sorted(pathlib.Path(
    "/home/hermes/fusion_av/seeded").glob("*real*.mp4")) + \
    sorted(pathlib.Path("/home/hermes/fusion_av/seeded").glob("testsrc*.mp4"))

if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)

adapter = _resolve_adapter("VISHWAS_EFFORT_WEIGHTS")
model = adapter.load(os.environ["VISHWAS_EFFORT_WEIGHTS"])
net = model.model if hasattr(model, "model") else model
net.eval()
device = next(net.parameters()).device


def resize_chw(arr, size=224):
    import cv2
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = arr[y0:y0 + s, x0:x0 + s]
    resized = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
    return np.transpose(resized, (2, 0, 1))


def score_bgr(chw):
    a = np.asarray(chw, dtype=np.float32)
    if a.ndim == 3:
        a = a[None]
    a = a[:, ::-1, :, :].copy()
    a = (a - _IMAGENET_MEAN[None, :, None, None]) / _IMAGENET_STD[None, :, None, None]
    xt = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    with torch.no_grad():
        return float(F.softmax(net(xt), dim=-1)[:, 1].mean().item())


def score_clip(p, i):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"op{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)
    probs = []
    for fp in frames[:8]:
        img = _read_bgr(fp)
        if img is None:
            continue
        probs.append(score_bgr(resize_chw(img)))
    return (sum(probs) / len(probs)) if probs else None


# --- score everything, and get each clip's REAL cross-modal av value ------
from vishwas.capabilities.cross_modal import CrossModalCapability
xcap = CrossModalCapability()


def av_for(p, i):
    q = QUARANTINE / f"xm_{i}"
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"xm{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    for c in xcap.analyze(art, ctx):
        if c.name == "cross_modal_av" and c.status == "ok":
            return c.signals.get("av_risk_addition"), c.signals.get("alignment_class")
    return None, None


items = []
for i, p in enumerate(OPERATOR_REALS):
    items.append(("REAL (operator)", p, score_clip(p, i), av_for(p, i)))
for i, p in enumerate(AI_ANCHORS, start=100):
    items.append(("AI (operator)", p, score_clip(p, i), av_for(p, i)))
for i, p in enumerate(CONTROLS, start=200):
    items.append(("CONTROL (synthetic)", p, score_clip(p, i), av_for(p, i)))

print(f"{'kind':<22} {'file':<34} {'effort':>8} {'av':>5}  align")
for kind, p, sc, (av, align) in items:
    print(f"{kind:<22} {p.name[:32]:<34} "
          f"{('%.4f' % sc) if sc is not None else 'n/a':>8} "
          f"{('%.2f' % av) if av is not None else 'n/a':>5}  {align or ''}")


def evaluate(offset):
    base_w = dict(WEIGHTS["deepfake_video"])
    base_c = dict(_CALIBRATE)
    WEIGHTS["deepfake_video"] = {**base_w, "effort.prob": 1.5}
    _CALIBRATE["effort.prob"] = (1.0, offset)
    try:
        fe = FusionEngine()
        out = []
        for kind, p, sc, (av, align) in items:
            checks = [
                C("video_probe", "cheap", "ok", {"duration_s": 1.0}, "m"),
                C("usable_frames", "cheap", "ok", {"usable_frame_ratio": 1.0}, "m"),
                C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.10}, "m"),
                C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": sc}, "m"),
                C("cross_modal_av", "heavy", "ok",
                  {"av_risk_addition": av if av is not None else 0.0,
                   "alignment_class": align or "synced",
                   "av_correlation": 0.4,
                   "av_synced_clean": 0.4 if (av == 0.0) else 0.0}, "m"),
                C("havic_crossmodal_model", "heavy", "ok",
                  {"prob_inconsistent": 0.99}, "m"),
            ]
            d = fe.decide("deepfake_video", checks)
            out.append((kind, d.verdict.value))
    finally:
        WEIGHTS["deepfake_video"] = base_w
        _CALIBRATE.clear()
        _CALIBRATE.update(base_c)
    return out


print()
print("=" * 100)
print("OFFSET SWEEP against the operator's own media")
print("=" * 100)
print(f"{'offset':>8} | {'REAL (4 operator clips)':<34} | {'AI (3 operator videos)':<32}")
print(f"{'':>8} | {'DNU must be 0':<34} | {'DNU should be 3':<32}")
for off in (-0.55, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, 0.0):
    res = evaluate(off)
    reals = [v for k, v in res if k.startswith("REAL")]
    ais = [v for k, v in res if k.startswith("AI")]
    r_dnu = sum(1 for v in reals if v == "do_not_use")
    a_dnu = sum(1 for v in ais if v == "do_not_use")
    a_trust = sum(1 for v in ais if v == "trust")
    ok = "  <-- MEETS THE INTENT" if r_dnu == 0 and a_dnu == 3 and a_trust == 0 else ""
    print(f"{off:>8.2f} | {str(Counter(reals)):.<34} | {str(Counter(ais)):<32}{ok}")
