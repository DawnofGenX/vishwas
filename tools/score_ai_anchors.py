#!/usr/bin/env python3
"""Score the operator's AI ANCHOR videos (seeded/*.mp4) through the real
video capability.

WHY
---
The leak-free corpus proves the video channel no longer produces false
positives. But a rule that never fires is as useless as one that always
fires. These 3 clips are the operator's OWN generated deepfakes, so a correct
rule must flag them.

NOTE: these clips are SILENT (no audio stream), so the cross-modal gates are
a known_gap on them. That is precisely the regime the 87-row calibration
corpus was built from, and it is where the video channel must still work.

Usage: python tools/score_ai_anchors.py
"""
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.deepfake_video import DeepfakeVideoCapability
from vishwas.events import Artifact, JobContext
from vishwas.fusion import FusionEngine
from vishwas.media_utils import extract_frames

CLIPS = pathlib.Path(os.environ.get(
    "VISHWAS_ANCHORS", "/home/hermes/fusion_av/seeded"))
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/anchors_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/operator_ai_anchors_signals.jsonl")

if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)

# The seeded/ dir holds 5 clips: 3 AI-generated (ai_crf45, ai_fps12,
# ai_resize50) and 2 real CONTROLS (smptebars_real_like, testsrc_real_control)
# that are synthetic test patterns, not natural video. Only the 3 AI clips
# are the operator's deepfake anchors.
AI = ["ai_crf45.mp4", "ai_fps12.mp4", "ai_resize50.mp4"]

clips = [CLIPS / n for n in AI if (CLIPS / n).exists()]
print(f"AI anchor clips: {len(clips)}")
for c in clips:
    print(f"  {c.name} exists={c.exists()}")

cap = DeepfakeVideoCapability()
fe = FusionEngine()
rows = []
for i, p in enumerate(clips):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"an_{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)
    heur = cap._spatial_heuristics(frames)
    by_h = {c.name: c for c in heur}
    fh = by_h["frame_heuristics"].signals.get("prob_deepfake") if "frame_heuristics" in by_h else None
    eff_c = cap._effort(ctx, frames)
    by_e = {c.name: c for c in eff_c}
    eff = by_e["effort_face_forensics"].signals.get("prob_deepfake") if "effort_face_forensics" in by_e else None
    d = fe.decide("deepfake_video", list(heur) + list(eff_c))
    rec = {"path": str(p), "label": 1, "effort": eff, "frameheur": fh,
           "av": None, "havic": None,
           "verdict": d.verdict.value, "score": round(float(d.score), 4),
           "known_gaps": list(d.known_gaps)}
    rows.append(rec)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  {p.name:<22s} eff={eff} fh={fh} -> {d.verdict.value} ({d.score:.3f}) "
          f"gaps={d.known_gaps}")

print(f"\nwrote {len(rows)} rows -> {OUT}")
from collections import Counter
print("verdicts:", dict(Counter(r["verdict"] for r in rows)))
