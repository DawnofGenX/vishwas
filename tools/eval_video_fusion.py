#!/usr/bin/env python3
"""Evaluate deepfake_video fusion verdicts on a leak-free labelled corpus and
search for a defensible recalibration.

WHAT THIS DOES
--------------
Replays the MEASURED per-signal values (from tools/score_video_signals.py) through
the REAL FusionEngine.decide("deepfake_video", checks) using real CheckResult
objects - no synthetic signals, no invented scores. Then it evaluates
candidate weight/calibration sets against the ground-truth labels.

It reports:
  * the CURRENT configuration's confusion matrix;
  * every candidate's accuracy, and specifically the two errors that matter:
      - FALSE POSITIVE  (a REAL clip flagged as fake)  <- a security tool that
        cries wolf on genuine content is worse than one that stays silent;
      - FALSE NEGATIVE  (a FAKE clip reads clean)       <- the overclaim that
        matters most.

Usage:
  VISHWAS_VIDEO_ROWS=/path/rows.jsonl python tools/eval_video_fusion.py
"""
import itertools
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE

ROWS = pathlib.Path(os.environ.get(
    "VISHWAS_VIDEO_ROWS",
    "/home/hermes/.hermes/cache/scratch/video_signal_rows_v2.jsonl"))

rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
print(f"rows: {len(rows)} "
      f"({sum(1 for r in rows if r['label']==0)} real / "
      f"{sum(1 for r in rows if r['label']==1)} fake)")
print(f"source: {ROWS}\n")


def build_checks(r):
    """Real CheckResult objects carrying the measured values."""
    out = []
    eff = r.get("effort")
    out.append(CheckResult(
        "effort_face_forensics", "heavy",
        "ok" if eff is not None else "unavailable",
        {"prob_deepfake": eff} if eff is not None else {},
        "measured"))
    fh = r.get("frameheur")
    out.append(CheckResult(
        "frame_heuristics", "mid",
        "ok" if fh is not None else "unavailable",
        {"prob_deepfake": fh} if fh is not None else {},
        "measured"))
    av = r.get("av")
    out.append(CheckResult(
        "cross_modal_av", "heavy",
        "ok" if av is not None else "skipped",
        {"av_risk_addition": av, "alignment_class": "measured",
         "av_synced_clean": 0.0} if av is not None else {},
        "measured"))
    hav = r.get("havic")
    out.append(CheckResult(
        "havic_crossmodal_model", "heavy",
        "ok" if hav is not None else "skipped",
        {"prob_inconsistent": hav} if hav is not None else {},
        "measured"))
    return out


def evaluate(weights=None, calibrate=None, quiet=True):
    """Return a confusion summary for a candidate configuration."""
    if weights is not None:
        WEIGHTS["deepfake_video"] = weights
    if calibrate is not None:
        _CALIBRATE.update(calibrate)
    fe = FusionEngine()
    res = []
    for r in rows:
        d = fe.decide("deepfake_video", build_checks(r))
        res.append((r["label"], d.verdict.value, float(d.score)))
    real = [v for lab, v, s in res if lab == 0]
    fake = [v for lab, v, s in res if lab == 1]
    fp = sum(1 for v in real if v == "do_not_use")
    fn = sum(1 for v in fake if v in ("trust",))
    dnu_fake = sum(1 for v in fake if v == "do_not_use")
    trust_real = sum(1 for v in real if v == "trust")
    return {
        "real": real, "fake": fake, "res": res,
        "fp": fp, "fn": fn, "dnu_fake": dnu_fake, "trust_real": trust_real,
        "n_real": len(real), "n_fake": len(fake),
    }


def show(name, m):
    print(f"{name:<46s} FP(real->DNU)={m['fp']:2d}/{m['n_real']}  "
          f"FN(fake->TRUST)={m['fn']:2d}/{m['n_fake']}  "
          f"fakeDNU={m['dnu_fake']:2d}  realTRUST={m['trust_real']:2d}")


BASE_W = dict(WEIGHTS["deepfake_video"])
BASE_C = {k: v for k, v in _CALIBRATE.items()}

print("=" * 78)
print("CURRENT CONFIGURATION")
print("=" * 78)
print(f"weights:    {BASE_W}")
print(f"calibrate:  { {k: BASE_C[k] for k in ('effort.prob','frameheur.prob','av_risk_addition','havic.prob_inconsistent') if k in BASE_C} }")
cur = evaluate()
show("CURRENT", cur)
from collections import Counter
print(f"  real verdicts:  {dict(Counter(cur['real']))}")
print(f"  fake verdicts:  {dict(Counter(cur['fake']))}")

print()
print("=" * 78)
print("CANDIDATE A: drop the leaked cross-modal gates from the weight map")
print("=" * 78)
print("The cross-modal gates cannot be trusted from a synthetic corpus, and the")
print("only corpus with natural AV audio is 4 operator clips. Dropping them")
print("leaves effort + frameheur, the two VIDEO-ONLY signals that CAN be")
print("measured honestly. A known_gap then contributes its weight at zero, so")
print("the denominator stays honest.")
W_A = {"effort.prob": 1.5, "frameheur.prob": 1.0}
a = evaluate(weights=W_A, calibrate={"effort.prob": (1.0, 0.0), "frameheur.prob": (1.0, 0.0)})
show("A: effort+frameheur only, no affine", a)
print(f"  real verdicts:  {dict(Counter(a['real']))}")
print(f"  fake verdicts:  {dict(Counter(a['fake']))}")

print()
print("=" * 78)
print("CANDIDATE B: A + the clean-side override, calibrated on the real medians")
print("=" * 78)
for bshift in (-0.30, -0.35, -0.40, -0.45):
    for fh_a in (0.0, 0.5, 1.0):
        W_B = {"effort.prob": 1.5, "frameheur.prob": 1.0}
        C_B = {"effort.prob": (1.0, bshift), "frameheur.prob": (fh_a, 0.0)}
        b = evaluate(weights=W_B, calibrate=C_B)
        # keep only candidates with no false positives on reals
        if b["fp"] == 0:
            print(f"  effort b={bshift:+.2f} fh_a={fh_a:.1f} -> "
                  f"FP={b['fp']} FN={b['fn']} fakeDNU={b['dnu_fake']} realTRUST={b['trust_real']}  "
                  f"real={dict(Counter(b['real']))} fake={dict(Counter(b['fake']))}")

print()
print("=" * 78)
print("CANDIDATE C: grid search over effort b, frameheur a, weights")
print("=" * 78)
best = []
for we in (1.0, 1.5, 2.0, 2.5):
    for wf in (0.0, 0.25, 0.5, 1.0):
        for b in (-0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20):
            for fa in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
                W_C = {"effort.prob": we, "frameheur.prob": wf}
                C_C = {"effort.prob": (1.0, b), "frameheur.prob": (fa, 0.0)}
                m = evaluate(weights=W_C, calibrate=C_C)
                if m["fp"] == 0:
                    best.append((m["dnu_fake"], m["trust_real"], we, wf, b, fa,
                                 dict(Counter(m["real"])), dict(Counter(m["fake"]))))
best.sort(key=lambda t: (-t[0], -t[1]))
print(f"candidates with ZERO false positives on reals: {len(best)}")
for t in best[:12]:
    print(f"  fakeDNU={t[0]:2d} realTRUST={t[1]:2d} | w_eff={t[2]} w_fh={t[3]} "
          f"b_eff={t[4]:+.2f} a_fh={t[5]:.1f} | real={t[6]} fake={t[7]}")

# restore the module state so nothing downstream sees the experiment
WEIGHTS["deepfake_video"] = BASE_W
_CCAL = BASE_C
_CALIBRATE.clear(); _CALIBRATE.update(BASE_C)
print("\n(restored the module weight/calibration state)")
