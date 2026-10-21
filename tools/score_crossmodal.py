#!/usr/bin/env python3
"""Score the cross-modal corpus through the FULL production video capability.

The published 87-row calibration corpus has no audio on 83 of 87 rows, so the
cross-modal gates (av_risk w=3.5, havic w=0.8) were known_gap on 95% of it.
This corpus (60 rows, 30 real / 30 fake, every row WITH audio) lets the
cross-modal gate actually engage, so we can see whether it helps or hurts.

It runs the real DeepfakeVideoCapability.analyze() over the real files, then
the real FusionEngine.decide() over the resulting checks — no replay, no
synthetic signal values.
"""
import collections
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, "/home/hermes/vishwas/src")

from vishwas.capabilities.deepfake_video import DeepfakeVideoCapability
from vishwas.capabilities.cross_modal import CrossModalCapability
from vishwas.events import Artifact, JobContext
from vishwas.fusion import FusionEngine

CORPUS = pathlib.Path(os.environ.get("VISHWAS_XM_OUT_JSONL",
                                   "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v1.jsonl"))
QUARANTINE = pathlib.Path(os.environ.get("VISHWAS_XM_QUARANTINE",
                                  "/home/hermes/.hermes/cache/scratch/xmodal_q"))
if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
print(f"corpus: {len(rows)} rows "
      f"({sum(1 for r in rows if r['label']==0)} real / {sum(1 for r in rows if r['label']==1)} fake)")

cap = DeepfakeVideoCapability()
xcap = CrossModalCapability()   # a SEPARATE capability the orchestrator runs
fe = FusionEngine()

results = []
for i, r in enumerate(rows):
    q = QUARANTINE / f"job_{i}"
    q.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(r["path"])
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"xm_{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0,
                     model_weights_available=True)
    try:
        checks = cap.analyze(art, ctx)
        # The orchestrator runs the cross-modal capability as its own stage;
        # deepfake_video does NOT include it.
        checks = list(checks) + list(xcap.analyze(art, ctx))
    except Exception as e:
        print(f"  row {i}: {type(e).__name__} {e}")
        continue
    d = fe.decide("deepfake_video", checks)
    av = next((c for c in checks if c.name == "cross_modal_av"), None)
    eff = next((c for c in checks if c.name == "effort_face_forensics"), None)
    rec = {
        "label": r["label"], "verdict": d.verdict.value, "score": round(float(d.score), 4),
        "disagreement": round(float(d.disagreement), 3),
        "av_status": (av.status if av else "absent"),
        "av_value": (av.signals.get("av_risk_addition") if av else None),
        "av_alignment": (av.signals.get("alignment_class") if av else None),
        "effort": (eff.signals.get("prob_deepfake") if eff else None),
    }
    results.append(rec)
    print(f"  {p.name:16s} label={r['label']} -> {d.verdict.value:20s} "
          f"score={d.score:.3f} dis={d.disagreement:.3f} "
          f"av={rec['av_status']}/{rec['av_value']} eff={rec['effort']}")

pathlib.Path("/tmp/xmodal_results.json").write_text(json.dumps(results))

if results:
    print("\n=== SUMMARY ===")
    print("  verdicts:", dict(collections.Counter(r["verdict"] for r in results)))
    for lab in (0, 1):
        sub = [r for r in results if r["label"] == lab]
        if sub:
            caught = sum(1 for r in sub if r["verdict"] == "do_not_use")
            clean = sum(1 for r in sub if r["verdict"] in ("trust", "low"))
            print(f"  label={lab} n={len(sub)}  DNU={caught}  clean={clean}  "
                  f"caution={sum(1 for r in sub if r['verdict']=='caution')}")
    av_used = sum(1 for r in results if r["av_status"] == "ok")
    av_valued = sum(1 for r in results if r["av_value"] is not None)
    print(f"  cross_modal_av RAN on {av_used}/{len(results)} rows, produced a VALUE on {av_valued}")
    print("  (it was known_gap on 83/87 of the old 87-row corpus)")
