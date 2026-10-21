#!/usr/bin/env python3
"""TEST: is EFFORT being fed BGR where it expects RGB?

THE SUSPICION
-------------
EFFORT-ffpp is an FF++-trained detector evaluated on FF++ data. It scores
AUC 0.6654 on the FF++ corpus, which is far below what a face-forensics
model trained on that data should achieve. The video path is:

    cv2.imread(...)        -> HWC uint8, BGR
    adapter.preprocess()  -> _img_resize_chw: centre-crop, resize, /255, transpose -> CHW
    EffortSpec.score()    -> ImageNet-normalise with RGB-ordered mean/std
                             -> softmax(logits)[..., 1] == fake posterior

`score()` never converts BGR to RGB, and `_img_resize_chw` never converts
either. So the model receives a BLUE-RED-SWAPPED image and normalises it
with RGB statistics. This is exactly the trap documented for SPAI in
model_archs/spai.py ("expects HWC BGR ... and does the BGR->RGB swap
ITSELF") -- SPAI's spec does the swap, EFFORT's does not.

THIS SCRIPT MEASURES IT: it scores every clip twice, once through the
production path and once with an explicit BGR->RGB conversion, and reports
the AUC/EER for each. If the RGB variant separates better, the production
path is feeding the model the wrong channel order.

Usage: python tools/eval_effort_channel_order.py
"""
import json
import os
import pathlib
import shutil
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from vishwas.events import Artifact, JobContext
from vishwas.media_utils import extract_frames
from vishwas.model_adapters import resolve as _resolve_adapter
from vishwas.capabilities.deepfake_video import _read_bgr, _infer

CORPUS = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_CORPUS",
    "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v2.jsonl"))
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_ch_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_channel_rows.jsonl")


def auc(pos, neg):
    s = 0
    n = 0
    for p in pos:
        for q in neg:
            s += 1 if p > q else (0.5 if p == q else 0)
            n += 1
    return s / n


def eer(pos, neg):
    ths = sorted(set([round(v, 4) for v in pos + neg]))
    best = (1.0, None, 0, 0)
    for t in ths:
        miss = sum(1 for p in pos if p >= t) / len(pos)
        fa = sum(1 for q in neg if q < t) / len(neg)
        e = (miss + fa) / 2
        if e < best[0]:
            best = (e, t, miss, fa)
    return best


if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
print(f"corpus: {len(rows)} rows")

env_name = "VISHWAS_EFFORT_WEIGHTS"
adapter = _resolve_adapter(env_name)
print(f"adapter: {adapter}")
if adapter is None:
    print("FATAL: the EFFORT adapter did not resolve")
    raise SystemExit(1)
model = adapter.load(os.environ[env_name])
print(f"model loaded: {type(model).__name__}")

out_rows = []
for i, r in enumerate(rows):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(r["path"])
    if not p.exists():
        continue
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"eff{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)

    bgr_probs, rgb_probs = [], []
    for fp in frames[:8]:
        img = _read_bgr(fp)
        if img is None:
            continue
        # A: the production path (BGR straight through)
        pa = _infer(adapter, model, img)
        if pa is not None:
            bgr_probs.append(pa)
        # B: the same frame with the channels swapped to RGB
        pb = _infer(adapter, model, img[:, :, ::-1].copy())
        if pb is not None:
            rgb_probs.append(pb)

    rec = {
        "path": str(p), "label": r["label"],
        "bgr": (st.median(bgr_probs) if bgr_probs else None),
        "rgb": (st.median(rgb_probs) if rgb_probs else None),
        "n": len(bgr_probs),
    }
    out_rows.append(rec)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {p.name[:26]:<28s} lab={r['label']}  "
          f"BGR(as shipped)={rec['bgr']}  RGB(swapped)={rec['rgb']}")

print()
print("=" * 92)
print("EFFORT: BGR (as shipped) vs RGB (channel-swapped) -- same frames, same model")
print("=" * 92)
for key in ("bgr", "rgb"):
    reals = [r[key] for r in out_rows if r["label"] == 0 and r[key] is not None]
    fakes = [r[key] for r in out_rows if r["label"] == 1 and r[key] is not None]
    if not reals or not fakes:
        print(f"  {key}: not enough data")
        continue
    a = auc(fakes, reals)
    e = eer(fakes, reals)
    print(f"  {key.upper():<4s} AUC={a:.4f}  EER={e[0]*100:5.2f}%  "
          f"real med={st.median(reals):.4f}  fake med={st.median(fakes):.4f}  "
          f"n={len(reals)+len(fakes)}")
print()
print(f"  wrote {len(out_rows)} rows -> {OUT}")
print()
print("  If RGB separates clearly better, the production path is feeding the")
print("  model blue-red-swapped frames normalised with RGB statistics.")
