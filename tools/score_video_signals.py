#!/usr/bin/env python3
"""Collect the four WEIGHTED deepfake_video signals with ground truth.

Why
---
The published calibration corpus (rows_video_v3.jsonl, 87 rows) has NO audio
stream on 83/87 rows, so `av_risk_addition` (w=3.5) and
`havic.prob_inconsistent` (w=0.8) were `known_gap` on 95% of the set the
weights were fitted on. The weights were fitted to a regime that does not
exist in production.

This harness runs the REAL production code paths - the same functions the
capability uses, in the same order - and records only the four signals that
actually enter the fusion weight map.

It SKIPS `transform_consistency` (the 6-variant ffmpeg battery). That check
is NOT in the weight map: it feeds `spread`/confidence only, and it is what
makes a full `analyze()` take ~2 minutes per clip. Skipping it does not
change any weighted value.

Everything else is production code:
  * `extract_frames`      (src/vishwas/media_utils.py)
  * `_spatial_heuristics` (src/vishwas/capabilities/deepfake_video.py)
  * `_effort`             (src/vishwas/capabilities/deepfake_video.py)
  * `CrossModalCapability.analyze` (src/vishwas/capabilities/cross_modal.py)

Output JSONL rows:
  {"path":..., "label":0|1, "effort":..., "frameheur":..., "av":..., "havic":...,
   "av_status":..., "has_audio":...}
"""
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.cross_modal import CrossModalCapability
from vishwas.capabilities.deepfake_video import DeepfakeVideoCapability
from vishwas.events import Artifact, JobContext
from vishwas.media_utils import extract_frames

CORPUS = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_CORPUS",
    "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v1.jsonl"))
OUT = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_OUT",
    "/home/hermes/.hermes/cache/scratch/video_signal_rows.jsonl"))
QUARANTINE = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_QUARANTINE",
    "/home/hermes/.hermes/cache/scratch/vidsig_q"))
LIMIT = int(os.environ.get("VISHWAS_VIDEO_LIMIT", "0"))

if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("")

rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
if LIMIT:
    rows = rows[:LIMIT]
n_real = sum(1 for r in rows if r["label"] == 0)
print(f"corpus: {len(rows)} rows ({n_real} real / {len(rows)-n_real} fake)", flush=True)

cap = DeepfakeVideoCapability()
xcap = CrossModalCapability()


def _get(by_name, name, key):
    c = by_name.get(name)
    if c is None:
        return None, "absent"
    if c.status == "ok":
        return c.signals.get(key), "ok"
    return None, ("known_gap" if c.status in ("skipped", "unavailable") else c.status)


out_rows = []
for i, r in enumerate(rows):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(r["path"])
    if not p.exists():
        print(f"  {p.name}: MISSING", flush=True)
        continue
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"vs_{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)

    # --- weighted signal 1 & 2: EFFORT + frame heuristics (production path) ---
    frames = extract_frames(art.path, q / "frames", n=8)
    heur = cap._spatial_heuristics(frames)
    by_h = {c.name: c for c in heur}
    fh, fh_st = _get(by_h, "frame_heuristics", "prob_deepfake")

    eff_checks = cap._effort(ctx, frames)
    by_e = {c.name: c for c in eff_checks}
    eff, eff_st = _get(by_e, "effort_face_forensics", "prob_deepfake")

    # --- weighted signal 3 & 4: cross-modal AV + HAVIC ---
    try:
        xchecks = list(xcap.analyze(art, ctx))
    except Exception as e:  # noqa: BLE001
        print(f"  {p.name}: xmodal {type(e).__name__} {e}", flush=True)
        xchecks = []
    by_x = {c.name: c for c in xchecks}
    av, av_st = _get(by_x, "cross_modal_av", "av_risk_addition")
    hav, hav_st = _get(by_x, "havic_crossmodal_model", "prob_inconsistent")

    rec = {
        "path": str(p), "label": r["label"],
        "effort": eff, "effort_status": eff_st,
        "frameheur": fh, "frameheur_status": fh_st,
        "av": av, "av_status": av_st,
        "havic": hav, "havic_status": hav_st,
        "has_audio": r.get("has_audio"),
        "n_frames": len(frames),
    }
    out_rows.append(rec)
    with OUT.open("a") as fh_:
        fh_.write(json.dumps(rec) + "\n")
    print(f"  {p.name:22s} lab={r['label']} eff={eff} ({eff_st}) "
          f"fh={fh} ({fh_st}) av={av} ({av_st}) hav={hav} ({hav_st})", flush=True)

print(f"\nwrote {len(out_rows)} rows -> {OUT}")
for key in ("effort", "frameheur", "av", "havic"):
    ok = [r for r in out_rows if r[f"{key}_status"] == "ok"]
    print(f"  {key:10s} produced a value on {len(ok)}/{len(out_rows)}")
