#!/usr/bin/env python3
"""Score the operator's REAL clips (the only ones with NATURAL, synchronised
audio) through the real video capability.

WHY THIS MATTERS
----------------
`trackA_operator_reals/` holds 4 clips the operator actually sent. They are
the only media on this host with genuine lip-sync. Every synthetic corpus
muxes audio onto video, so it can only test that the cross-modal gates are
WIRED - never whether they are RIGHT.

These 4 are all REAL. So for any candidate configuration:
  * reading them DO_NOT_USE is a FALSE POSITIVE on genuine content;
  * reading them CAUTION or TRUST is correct behaviour;
  * and the CROSS-MODAL signals here are honest (real lip-sync -> "synced"),
    so `av_risk_addition` should be 0.0 and the clean-side bonus should fire.

This is a small sample (n=4) and it is clean-side only - it can disprove a
configuration but it cannot by itself prove one. It is used as a GUARD, not a
tuner.

Usage: python tools/score_operator_reals.py
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
from vishwas.fusion import FusionEngine
from vishwas.media_utils import extract_frames

CLIPS = pathlib.Path(os.environ.get(
    "VISHWAS_OPERATOR_REALS",
    "/home/hermes/fusion_av/trackA_operator_reals"))
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/opreals_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/operator_reals_signals.jsonl")

if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)

clips = sorted(p for p in CLIPS.glob("*.mp4"))
print(f"operator real clips: {len(clips)} (all REAL, natural synchronised audio)")

cap = DeepfakeVideoCapability()
xcap = CrossModalCapability()
fe = FusionEngine()

rows = []
for i, p in enumerate(clips):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"op_{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    frames = extract_frames(art.path, q / "frames", n=8)
    heur = cap._spatial_heuristics(frames)
    by_h = {c.name: c for c in heur}
    fh = by_h["frame_heuristics"].signals.get("prob_deepfake") if "frame_heuristics" in by_h else None
    eff_c = cap._effort(ctx, frames)
    by_e = {c.name: c for c in eff_c}
    eff = by_e["effort_face_forensics"].signals.get("prob_deepfake") if "effort_face_forensics" in by_e else None

    try:
        xchecks = list(xcap.analyze(art, ctx))
    except Exception as e:  # noqa: BLE001
        print(f"  {p.name}: xmodal {type(e).__name__} {e}")
        xchecks = []
    by_x = {c.name: c for c in xchecks}
    cm = by_x.get("cross_modal_av")
    av = cm.signals.get("av_risk_addition") if cm and cm.status == "ok" else None
    align = cm.signals.get("alignment_class") if cm and cm.status == "ok" else None
    corr = cm.signals.get("av_correlation") if cm and cm.status == "ok" else None
    hav_c = by_x.get("havic_crossmodal_model")
    hav = hav_c.signals.get("prob_inconsistent") if hav_c and hav_c.status == "ok" else None

    d = fe.decide("deepfake_video", list(heur) + list(eff_c) + list(xchecks))
    rec = {"path": str(p), "label": 0, "effort": eff, "frameheur": fh,
           "av": av, "av_alignment": align, "av_correlation": corr,
           "havic": hav, "verdict": d.verdict.value, "score": round(float(d.score), 4)}
    rows.append(rec)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  {p.name[:28]:<30s} eff={eff} fh={fh} av={av} align={align} "
          f"corr={corr} hav={hav} -> {d.verdict.value} ({d.score:.3f})")

print(f"\nwrote {len(rows)} rows -> {OUT}")
if rows:
    from collections import Counter
    print("verdicts:", dict(Counter(r["verdict"] for r in rows)))
    dnu = [r for r in rows if r["verdict"] == "do_not_use"]
    print(f"FALSE POSITIVES (real clip flagged DO_NOT_USE): {len(dnu)}/{len(rows)}")
    for r in dnu:
        print(f"  {r['path'].split('/')[-1]}: eff={r['effort']} fh={r['frameheur']} "
              f"av={r['av']} hav={r['havic']}")
    aligned = [r for r in rows if r["av_alignment"] == "synced"]
    print(f"clips whose audio actually reads as SYNCHRONISED: {len(aligned)}/{len(rows)} "
          "(the honest cross-modal case)")
