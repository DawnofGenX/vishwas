#!/usr/bin/env python3
"""Evaluate the implementable face-crop variants for the EFFORT video path.

THE PROBLEM
-----------
The measurement (`tools/eval_effort_face_crop.py`) shows that cropping to
the YuNet face region raises the raw detector's separation (AUC 0.7189 ->
0.7711). But two things block a naive switch:

1. **A clip may have NO detectable face.** One of the operator's 4 genuine
   clips has faces=0 on every sampled frame. A face-crop path would have no
   signal for it, so the crop MUST fall back to the whole frame.
2. **The crop raises the score on SOME genuine clips** (ffc23 real 451:
   0.421 -> 0.775). Sharpening a face-forensics model onto a low-quality
   phone face can read as synthetic.

VARIANTS
--------
  full      the production path (whole frame, centre-cropped)
  face_only use the face crop when a face is found, else the whole frame
  max       per frame, take the MAXIMUM of the two readings
  avg       per frame, take the MEAN of the two readings
  face_only_the_frame_median
            use the face crop for the clip's median when ANY frame has a
            face, else the whole frame

Each is evaluated on the leak-free corpus (AUC) AND on the operator's own
media (the clean-side guard).

Usage: python tools/eval_effort_face_variants.py
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
import cv2

from vishwas.capabilities.deepfake_video import _read_bgr
from vishwas.events import Artifact, JobContext
from vishwas.media_utils import extract_frames
from vishwas.model_adapters import resolve as _resolve_adapter
from vishwas.model_archs.effort import _IMAGENET_MEAN, _IMAGENET_STD

FACE_MODEL = "/opt/vishwas/models/havic/face_detection_yunet_2023mar.onnx"
SCALE = 1.3
CORPUS = pathlib.Path("/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v2.jsonl")
OP_REALS = sorted(pathlib.Path("/home/hermes/fusion_av/trackA_operator_reals").glob("*.mp4"))
AI_ANCHORS = sorted(pathlib.Path("/home/hermes/fusion_av/seeded").glob("ai_*.mp4"))
Q = pathlib.Path("/home/hermes/.hermes/cache/scratch/facevar_q")


def auc(pos, neg):
    s = n = 0
    for p in pos:
        for q in neg:
            s += 1 if p > q else (0.5 if p == q else 0)
            n += 1
    return s / n


if Q.exists():
    shutil.rmtree(Q, ignore_errors=True)
Q.mkdir(parents=True, exist_ok=True)

adapter = _resolve_adapter("VISHWAS_EFFORT_WEIGHTS")
model = adapter.load(os.environ["VISHWAS_EFFORT_WEIGHTS"])
net = model.model if hasattr(model, "model") else model
net.eval()
device = next(net.parameters()).device


def score_bgr(chw):
    a = np.asarray(chw, dtype=np.float32)
    if a.ndim == 3:
        a = a[None]
    a = a[:, ::-1, :, :].copy()
    a = (a - _IMAGENET_MEAN[None, :, None, None]) / _IMAGENET_STD[None, :, None, None]
    xt = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    with torch.no_grad():
        return float(F.softmax(net(xt), dim=-1)[:, 1].mean().item())


def resize_chw(arr, size=224):
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = arr[y0:y0 + s, x0:x0 + s]
    r = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
    return np.transpose(r, (2, 0, 1))


def detect(img):
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN_create(FACE_MODEL, "", (w, h), 0.6, 0.3, 5000)
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    return tuple(float(v) for v in max(faces, key=lambda f: f[4])[:4])


def crop_face(img, box, scale):
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * scale
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(img.shape[1], x0 + side)), int(min(img.shape[0], y0 + side))
    return img[y0:y1, x0:x1] if (x1 - x0 >= 16 and y1 - y0 >= 16) else None


def variants_for(p, i):
    """Return a dict of variant -> clip-level score."""
    q = Q / f"j{i}"
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"v{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)
    full_s, face_s = [], []
    for fp in frames[:8]:
        img = _read_bgr(fp)
        if img is None:
            continue
        full_s.append(score_bgr(resize_chw(img)))
        box = detect(img)
        if box is None:
            continue
        c = crop_face(img, box, SCALE)
        if c is not None:
            face_s.append(score_bgr(resize_chw(c)))
    if not full_s:
        return None
    out = {"full": st.median(full_s)}
    out["face_only"] = st.median(face_s) if face_s else out["full"]
    out["max"] = st.median([max(f, c) for f, c in zip(full_s, face_s)]) if face_s else out["full"]
    out["avg"] = st.median([(f + c) / 2 for f, c in zip(full_s, face_s)]) if face_s else out["full"]
    return out


print("=== the leak-free corpus ===")
rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
corpus_out = []
for i, r in enumerate(rows):
    p = pathlib.Path(r["path"])
    if not p.exists():
        continue
    v = variants_for(p, i)
    if v:
        v["label"] = r["label"]
        corpus_out.append(v)
        print(f"  {p.name[:22]:<24s} lab={r['label']} " +
              " ".join(f"{k}={v[k]:.3f}" for k in ("full", "face_only", "max", "avg")))

print()
print("=" * 92)
print("VARIANT COMPARISON on the leak-free corpus")
print("=" * 92)
for key in ("full", "face_only", "max", "avg"):
    rv = [r[key] for r in corpus_out if r["label"] == 0]
    fv = [r[key] for r in corpus_out if r["label"] == 1]
    a = auc(fv, rv)
    print(f"  {key:<10s} AUC={a:.4f}  real med={st.median(rv):.4f}  "
          f"fake med={st.median(fv):.4f}  gap={st.median(fv)-st.median(rv):+.4f}")

print()
print("=" * 92)
print("the OPERATOR'S OWN media (the clean-side guard)")
print("=" * 92)
op_out = []
for i, p in enumerate(OP_REALS):
    v = variants_for(p, 1000 + i)
    if v:
        v["kind"] = "REAL"
        op_out.append(v)
        print(f"  REAL {p.name[:20]:<22s} " +
              " ".join(f"{k}={v[k]:.3f}" for k in ("full", "face_only", "max", "avg")))
for i, p in enumerate(AI_ANCHORS):
    v = variants_for(p, 2000 + i)
    if v:
        v["kind"] = "AI"
        op_out.append(v)
        print(f"  AI   {p.name[:20]:<22s} " +
              " ".join(f"{k}={v[k]:.3f}" for k in ("full", "face_only", "max", "avg")))

print()
print("  with the -0.30 offset, how many AI videos would the effort term")
print("  alone push into the DO_NOT_USE band (calibrated >= 0.40)?")
for key in ("full", "face_only", "max", "avg"):
    ai = [max(0.0, r[key] - 0.30) for r in op_out if r["kind"] == "AI"]
    real = [max(0.0, r[key] - 0.30) for r in op_out if r["kind"] == "REAL"]
    print(f"    {key:<10s} AI cal: {['%.3f' % v for v in ai]}  "
          f"-> {sum(1 for v in ai if v >= 0.40)}/3 flagged | "
          f"REAL cal: {['%.3f' % v for v in real]} -> "
          f"{sum(1 for v in real if v >= 0.40)}/4 flagged")
