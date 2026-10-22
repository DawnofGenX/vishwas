#!/usr/bin/env python3
"""MEASURE: does cropping to the FACE region improve EFFORT on video?

⚠️⚠️ THIS HARNESS MEASURES A **REJECTED** CROP — ITS NUMBERS ARE NOT THE
SHIPPED CONFIGURATION'S NUMBERS. (added 2026-10-22)
--------------------------------------------------------------------
This script has its OWN hand-rolled crop. It does NOT call the production
`deepfake_video._effort_face_crop`, and the two differ in two ways that
PR #12 fixed in production:

  1. FACE ORDERING — production picks the LARGEST-AREA box. This harness
     orders by YuNet's 5th column, which is a RAW detection score (it reads
     ~120 here, not 0..1), so that ordering is meaningless.
  2. MINIMUM FACE SIZE — production uses a FRACTION of the frame area
     (0.05%). This harness uses a 16-pixel floor, which rejects exactly the
     small distant faces that matter on the operator's wide street shots.

THE NUMBERS THIS PRINTS:
    face13  AUC 0.7711   <- THIS harness's crop (REJECTED)
    full    AUC 0.7189

THE SHIPPED NUMBERS (from tools/score_video_signals_crop.py, which uses the
PRODUCTION crop):
    crop    AUC 0.7789   <- the shipped configuration
    full    AUC 0.7189

BOTH ARE CORRECT — THEY MEASURE DIFFERENT CROPS. If you are checking the
shipped configuration, use `tools/score_video_signals_crop.py`, NOT this
script. The "full" row agrees (0.7189) because that path is identical.

WHY
---
EFFORT is a FACE-forensics model. The upstream paper (YZY-stack/Effort-
AIGI-Detection, ICML 2025 Oral) applies it to face regions, and the repo's
own model_archs/effort.py notes that the `chameleon` checkpoint is the
"face-region / spatial forensics" one while `ffpp` is "face forensics
(FF++ domain)".

But this pipeline feeds EFFORT the WHOLE FRAME, centre-cropped. The
`cross_modal` capability already loads a YuNet face detector
(face_detection_yunet_2023mar.onnx) for the HAVIC first-face crop, so the
tooling is on this host.

This script measures both crops on the leak-free corpus:
  A. the production path (whole frame, centre-cropped)
  B. the YuNet face region, expanded by FaceX-Zoo's 1.3 factor
  C. the face region WITHOUT expansion

It reports the AUC/EER for each, so the decision is made by measurement.

Usage: python tools/eval_effort_face_crop.py
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

from vishwas.capabilities.deepfake_video import _read_bgr
from vishwas.events import Artifact, JobContext
from vishwas.media_utils import extract_frames
from vishwas.model_adapters import resolve as _resolve_adapter
from vishwas.model_archs.effort import _IMAGENET_MEAN, _IMAGENET_STD

# The SAME detector the cross-modal capability already uses.
FACE_MODEL = "/opt/vishwas/models/havic/face_detection_yunet_2023mar.onnx"
FACE_SCALE = 1.3   # FaceX-Zoo FaceX_Zoo_FaceCropper, as used in cross_modal.py

CORPUS = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_CORPUS",
    "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v2.jsonl"))
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_face_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_face_rows.jsonl")


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


if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

import cv2

adapter = _resolve_adapter("VISHWAS_EFFORT_WEIGHTS")
model = adapter.load(os.environ["VISHWAS_EFFORT_WEIGHTS"])
net = model.model if hasattr(model, "model") else model
net.eval()
device = next(net.parameters()).device

detector = None
if os.path.exists(FACE_MODEL):
    try:
        detector = cv2.FaceDetectorYN_create(FACE_MODEL, "", (320, 320), 0.6, 0.3, 5000)
    except Exception as e:
        print(f"detector load failed: {e}")
if detector is None:
    print("FATAL: no face detector available; cannot measure the crop")
    raise SystemExit(1)
print(f"face detector: {FACE_MODEL}")


def score_bgr(chw):
    """The PRODUCTION score path: BGR in, RGB swap, CLIP normalise."""
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
    resized = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
    return np.transpose(resized, (2, 0, 1))


def detect_face(img):
    """Return the largest face box (x, y, w, h) or None."""
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN_create(FACE_MODEL, "", (w, h), 0.6, 0.3, 5000)
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    # faces: x, y, w, h, score...
    best = max(faces, key=lambda f: f[4] if len(f) > 4 else 0)
    return tuple(float(v) for v in best[:4])


def crop_face(img, box, scale):
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * scale
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(img.shape[1], x0 + side))
    y1 = int(min(img.shape[0], y0 + side))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return img[y0:y1, x0:x1]


rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
print(f"corpus: {len(rows)} rows\n")

out_rows = []
for i, r in enumerate(rows):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(r["path"])
    if not p.exists():
        continue
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"fc{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)

    full, face13, face10 = [], [], []
    n_faces = 0
    for fp in frames[:8]:
        img = _read_bgr(fp)
        if img is None:
            continue
        full.append(score_bgr(resize_chw(img)))
        box = detect_face(img)
        if box is None:
            continue
        n_faces += 1
        c13 = crop_face(img, box, FACE_SCALE)
        c10 = crop_face(img, box, 1.0)
        if c13 is not None:
            face13.append(score_bgr(resize_chw(c13)))
        if c10 is not None:
            face10.append(score_bgr(resize_chw(c10)))

    rec = {"path": str(p), "label": r["label"], "n_faces": n_faces,
           "full": (st.median(full) if full else None),
           "face13": (st.median(face13) if face13 else None),
           "face10": (st.median(face10) if face10 else None)}
    out_rows.append(rec)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {p.name[:24]:<26s} lab={r['label']} faces={n_faces} "
          f"full={rec['full']} face1.3={rec['face13']} face1.0={rec['face10']}")

print()
print("=" * 92)
print("EFFORT on the whole frame vs on the detected FACE region")
print("=" * 92)
for key in ("full", "face13", "face10"):
    reals = [r[key] for r in out_rows if r["label"] == 0 and r[key] is not None]
    fakes = [r[key] for r in out_rows if r["label"] == 1 and r[key] is not None]
    if not reals or not fakes:
        print(f"  {key}: insufficient data")
        continue
    a = auc(fakes, reals)
    e = eer(fakes, reals)
    tag = "  <- production path (whole frame)" if key == "full" else ""
    print(f"  {key:<8s} n={len(reals)+len(fakes):2d}  AUC={a:.4f}  EER={e[0]*100:5.2f}%  "
          f"real med={st.median(reals):.4f}  fake med={st.median(fakes):.4f}{tag}")
print()
print(f"  clips with a detected face: "
      f"{sum(1 for r in out_rows if r['n_faces'] > 0)}/{len(out_rows)}")
print(f"  wrote {len(out_rows)} rows -> {OUT}")
print()
print("!" * 92)
print("!! READ THIS BEFORE QUOTING THE NUMBERS ABOVE")
print("!!")
print("!!  The 'face13' and 'face10' rows measure THIS SCRIPT'S OWN crop, not")
print("!!  the shipped one. It differs from the production")
print("!!  deepfake_video._effort_face_crop in two ways PR #12 fixed:")
print("!!    - face ordering: production picks the LARGEST-AREA box; this")
print("!!      harness orders by YuNet's raw 5th column (reads ~120, not 0..1)")
print("!!    - minimum face size: production uses a FRACTION of the frame area")
print("!!      (0.05%); this harness uses a 16-pixel floor")
print("!!")
print("!!  THE SHIPPED NUMBERS are in tools/score_video_signals_crop.py, which")
print("!!  uses the PRODUCTION crop:")
print("!!      full 0.7189    crop 0.7789   <- the shipped configuration")
print("!!  This script prints 0.7711 for its own crop. BOTH ARE CORRECT --")
print("!!  THEY MEASURE DIFFERENT CROPS. Do not conclude the shipped crop is")
print("!!  worse, and do not treat the two as contradictory.")
print("!" * 92)
