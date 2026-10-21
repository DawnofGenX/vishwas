#!/usr/bin/env python3
"""Compare candidate deepfake_video configurations against BOTH corpora:
the leak-free synthetic corpus (60 rows) and the operator's natural-audio
real clips (4 rows).

The operator clips are the GUARD: they are genuine, synchronised media, so
any configuration that reads them DO_NOT_USE is producing a false positive on
real content, no matter what it does on the synthetic corpus.
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE

V2 = pathlib.Path("/home/hermes/.hermes/cache/scratch/video_signal_rows_v2.jsonl")
OP = pathlib.Path("/home/hermes/.hermes/cache/scratch/operator_reals_signals.jsonl")

v2 = [json.loads(l) for l in V2.read_text().splitlines() if l.strip()]
op = [json.loads(l) for l in OP.read_text().splitlines() if l.strip()]
base_w = dict(WEIGHTS["deepfake_video"])
base_c = {k: v for k, v in _CALIBRATE.items()}


def checks(r, use_xm):
    out = [
        CheckResult("effort_face_forensics", "heavy", "ok",
                    {"prob_deepfake": r["effort"]}, "m"),
        CheckResult("frame_heuristics", "mid", "ok",
                    {"prob_deepfake": r["frameheur"]}, "m"),
    ]
    if use_xm:
        out.append(CheckResult("cross_modal_av", "heavy", "ok",
                                {"av_risk_addition": r["av"],
                                 "alignment_class": r.get("av_alignment", "m"),
                                 "av_synced_clean": 0.0}, "m"))
        out.append(CheckResult("havic_crossmodal_model", "heavy", "ok",
                                {"prob_inconsistent": r["havic"]}, "m"))
    else:
        out.append(CheckResult("cross_modal_av", "heavy", "skipped", {}, "m"))
        out.append(CheckResult("havic_crossmodal_model", "heavy", "skipped", {}, "m"))
    return out


def ev(rows, W, C, xm):
    WEIGHTS["deepfake_video"] = W
    _CALIBRATE.update(C)
    fe = FusionEngine()
    res = []
    for r in rows:
        d = fe.decide("deepfake_video", checks(r, xm))
        res.append((r["label"], d.verdict.value))
    real = [v for l, v in res if l == 0]
    fake = [v for l, v in res if l == 1]
    return {"fp": sum(1 for v in real if v == "do_not_use"),
            "fn": sum(1 for v in fake if v == "trust"),
            "dnu_fake": sum(1 for v in fake if v == "do_not_use"),
            "trust_real": sum(1 for v in real if v == "trust"),
            "n_real": len(real), "n_fake": len(fake),
            "real": Counter(real), "fake": Counter(fake)}


def restore():
    WEIGHTS["deepfake_video"] = base_w
    _CALIBRATE.clear()
    _CALIBRATE.update(base_c)


XCAL = {"effort.prob": (1.0, -0.55), "frameheur.prob": (1.97, 0.05),
        "av_risk_addition": (1.0, 0.0), "havic.prob_inconsistent": (0.2, 0.0)}

CANDS = [
    ("CURRENT (shipping)", base_w, XCAL, True),
    ("A  video-only, fh_a=0, b=-0.55",
     {"effort.prob": 1.5, "frameheur.prob": 1.0},
     {"effort.prob": (1.0, -0.55), "frameheur.prob": (0.0, 0.0)}, False),
    ("A2 video-only, fh_a=0, b=-0.60",
     {"effort.prob": 1.5, "frameheur.prob": 1.0},
     {"effort.prob": (1.0, -0.60), "frameheur.prob": (0.0, 0.0)}, False),
    ("B  keep xm, fh_a=0, b=-0.55, av_w=0",
     {"effort.prob": 1.0, "frameheur.prob": 1.0, "av_risk_addition": 0.0,
      "havic.prob_inconsistent": 0.0},
     {"effort.prob": (1.0, -0.55), "frameheur.prob": (0.0, 0.0),
      "av_risk_addition": (1.0, 0.0), "havic.prob_inconsistent": (0.2, 0.0)}, True),
    ("C  keep xm, fh_a=1.0 (no amp), b=-0.55",
     {"effort.prob": 1.5, "frameheur.prob": 1.0, "av_risk_addition": 3.5,
      "havic.prob_inconsistent": 0.8},
     {"effort.prob": (1.0, -0.55), "frameheur.prob": (1.0, 0.0),
      "av_risk_addition": (1.0, 0.0), "havic.prob_inconsistent": (0.2, 0.0)}, True),
]

print("=" * 104)
print("CANDIDATE COMPARISON")
print(f"  leak-free synthetic corpus: {len(v2)} rows   "
      f"operator natural-audio reals: {len(op)} rows")
print("=" * 104)
print(f"{'candidate':<42} {'v2FP':>5} {'v2FN':>5} {'fakeDNU':>8} {'realTRUST':>10} "
      f"{'OPFP':>5} {'OPTRUST':>8}")
for name, W, C, xm in CANDS:
    m = ev(v2, W, C, xm); restore()
    mo = ev(op, W, C, xm); restore()
    print(f"{name:<42} {m['fp']:>5} {m['fn']:>5} {m['dnu_fake']:>8} "
          f"{m['trust_real']:>10} {mo['fp']:>5} {mo['trust_real']:>8}")
    print(f"{'':<42} v2 real={dict(m['real'])}")
    print(f"{'':<42} v2 fake={dict(m['fake'])}")
    print(f"{'':<42} operator_reals={dict(mo['real'])}")
    print()
