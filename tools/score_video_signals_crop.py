#!/usr/bin/env python3
"""Re-measure the video signals with the PRODUCTION face crop in place.

The earlier measurement (tools/eval_effort_face_crop.py) used a hand-rolled
crop with a 16-pixel minimum face size and an ordering by YuNet's raw
detection score. The production helper
(deepfake_video._effort_face_crop) differs in two ways that matter:
  * it orders faces by AREA, not by the raw score column (YuNet's 5th
    column is not on a 0..1 scale -- it reads ~120 here);
  * its minimum face size is a FRACTION of the frame, because the
    operator's AI videos are wide street shots whose face is only 3-4% of
    the width -- a 16px floor rejected exactly the faces that matter.

So the earlier numbers do not describe the shipped path. This tool measures
what the production path actually produces.

Usage: python tools/score_video_signals_crop.py
"""
import json
import os
import pathlib
import shutil
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.cross_modal import CrossModalCapability
from vishwas.capabilities.deepfake_video import (
    DeepfakeVideoCapability, _effort_face_crop, _read_bgr)
from vishwas.events import Artifact, JobContext
from vishwas.fusion import FusionEngine
from vishwas.media_utils import extract_frames
from vishwas.model_adapters import resolve as _resolve_adapter

CORPUS = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_CORPUS",
    "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v2.jsonl"))
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/vidsig_crop_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/video_signal_rows_crop.jsonl")

if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
cap = DeepfakeVideoCapability()
xcap = CrossModalCapability()
fe = FusionEngine()
adapter = _resolve_adapter("VISHWAS_EFFORT_WEIGHTS")
model = adapter.load(os.environ["VISHWAS_EFFORT_WEIGHTS"])


def score_frames(frames, crop: bool):
    from vishwas.capabilities.deepfake_video import _infer
    probs = []
    for fp in frames[:8]:
        img = _read_bgr(fp)
        if img is None:
            continue
        if crop:
            c = _effort_face_crop(img)
            if c is not None:
                img = c
        try:
            p = _infer(adapter, model, img)
            if p is not None:
                probs.append(p)
        except Exception:
            continue
    return st.median(probs) if probs else None


def auc(pos, neg):
    s = n = 0
    for p in pos:
        for q in neg:
            s += 1 if p > q else (0.5 if p == q else 0)
            n += 1
    return s / n


out = []
for i, r in enumerate(rows):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(r["path"])
    if not p.exists():
        continue
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"cg{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)

    # the OTHER three signals, via the production capabilities
    heur = cap._spatial_heuristics(frames)
    by_h = {c.name: c for c in heur}
    fh = by_h["frame_heuristics"].signals.get("prob_deepfake") if "frame_heuristics" in by_h else None
    xchecks = list(xcap.analyze(art, ctx))
    by_x = {c.name: c for c in xchecks}
    cm = by_x.get("cross_modal_av")
    av = cm.signals.get("av_risk_addition") if cm and cm.status == "ok" else None
    hv_c = by_x.get("havic_crossmodal_model")
    hav = hv_c.signals.get("prob_inconsistent") if hv_c and hv_c.status == "ok" else None

    rec = {"path": str(p), "label": r["label"],
           "effort_full": score_frames(frames, crop=False),
           "effort_crop": score_frames(frames, crop=True),
           "frameheur": fh, "av": av, "havic": hav}
    out.append(rec)
    with OUT.open("a") as fh_:
        fh_.write(json.dumps(rec) + "\n")
    print(f"  {p.name[:22]:<24s} lab={r['label']} full={rec['effort_full']:.3f} "
          f"crop={rec['effort_crop']:.3f} fh={fh} av={av}")

print()
print("=" * 92)
print("PRODUCTION face crop vs whole frame (the real code path)")
print("=" * 92)
for key in ("effort_full", "effort_crop"):
    rv = [r[key] for r in out if r["label"] == 0 and r[key] is not None]
    fv = [r[key] for r in out if r["label"] == 1 and r[key] is not None]
    print(f"  {key:<14s} AUC={auc(fv, rv):.4f}  real med={st.median(rv):.4f}  "
          f"fake med={st.median(fv):.4f}  gap={st.median(fv)-st.median(rv):+.4f}")
print(f"\n  wrote {len(out)} rows -> {OUT}")
print()
print("  EXPECTED (measured 2026-10-21, reproduced 2026-10-22):")
print("    effort_full  AUC=0.7189  real med=0.4762  fake med=0.6190")
print("    effort_crop  AUC=0.7789  real med=0.4619  fake med=0.6881")
print()
print("  THIS is the authoritative harness for the shipped face crop.")
print("  tools/eval_effort_face_crop.py measures a DIFFERENT (rejected) crop")
print("  and prints 0.7711 for it -- both are correct, they are not the same")
print("  measurement. Do not mix the two.")
