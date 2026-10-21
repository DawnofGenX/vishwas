#!/usr/bin/env python3
"""Test a CORROBORATION requirement for the video channel's DO_NOT_USE verdict.

Hypothesis under test
---------------------
Only ONE of the four weighted video signals has measured separation
(EFFORT, AUC 0.6511). frameheur is noise (0.5350), av_risk is 0.4744, and
HAVIC is 0.6272 while being near-degenerate (it reads ~0.95 on genuine
clips). When a weighted MEAN combines one informative signal with three
uninformative ones, the verdict is driven by whichever uninformative signal
happens to fire hardest on that clip — which is why a genuine clip with
EFFORT 0.924 and a mux-artefact decorrelation still crossed the DNU
threshold.

A corroboration rule ("DO_NOT_USE requires at least two INDEPENDENT signal
families to be elevated") is the standard remedy and is the same lesson the
repo already learned from the 2026-08-26 chameleon overfit: ONE untrusted
gate must not drive DNU.

This script measures the rule on:
  * the leak-free corpus (60 rows), and
  * the operator's genuine natural-audio clips (4 rows, clean-side guard).

Usage: python tools/eval_corroboration_rule.py
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE

V2 = pathlib.Path("/home/hermes/.hermes/cache/scratch/video_signal_rows_v2.jsonl")
OP = pathlib.Path("/home/hermes/.hermes/cache/scratch/operator_reals_signals.jsonl")

v2 = [json.loads(l) for l in V2.read_text().splitlines() if l.strip()]
op = [json.loads(l) for l in OP.read_text().splitlines() if l.strip()]

# The four signal families, with the calibrated value each contributes.
FAMILIES = ("effort", "frameheur", "av", "havic")


def cal(key, v):
    a, b = _CALIBRATE.get(key, (1.0, 0.0))
    return min(1.0, max(0.0, a * v + b))


def calibrated(r):
    """The calibrated value of each family, as the fusion layer sees it."""
    return {
        "effort": cal("effort.prob", r["effort"]),
        "frameheur": cal("frameheur.prob", r["frameheur"]),
        "av": cal("av_risk_addition", r["av"]),
        "havic": cal("havic.prob_inconsistent", r["havic"]),
    }


def elevated_families(r, thr):
    """How many DISTINCT families are elevated above `thr`."""
    c = calibrated(r)
    return sum(1 for k in FAMILIES if c[k] >= thr), c


def evaluate(rows, thr, min_families, weight_map=None):
    """Score the corpus under a hypothetical corroboration rule:
    the fusion verdict, but DO_NOT_USE only if >= min_families are elevated."""
    base_w = dict(WEIGHTS["deepfake_video"])
    if weight_map:
        WEIGHTS["deepfake_video"] = weight_map
    fe = FusionEngine()
    res = []
    for r in rows:
        checks = [
            C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": r["effort"]}, "m"),
            C("frame_heuristics", "mid", "ok", {"prob_deepfake": r["frameheur"]}, "m"),
            C("cross_modal_av", "heavy", "ok",
              {"av_risk_addition": r["av"],
               "alignment_class": r.get("av_alignment", "weakly_synced"),
               "av_correlation": r.get("av_correlation", 0.0),
               "av_synced_clean": 0.0}, "m"),
            C("havic_crossmodal_model", "heavy", "ok",
              {"prob_inconsistent": r["havic"]}, "m"),
        ]
        d = fe.decide("deepfake_video", checks)
        v = d.verdict.value
        if v == "do_not_use":
            n_elev, cal_vals = elevated_families(r, thr)
            if n_elev < min_families:
                v = "caution"   # demoted: not corroborated
        res.append((r["label"], v))
    WEIGHTS["deepfake_video"] = base_w
    real = [v for l, v in res if l == 0]
    fake = [v for l, v in res if l == 1]
    return {"fp": sum(1 for v in real if v == "do_not_use"),
            "fn": sum(1 for v in fake if v == "trust"),
            "dnu_fake": sum(1 for v in fake if v == "do_not_use"),
            "trust_real": sum(1 for v in real if v == "trust"),
            "n_real": len(real), "n_fake": len(fake),
            "real": Counter(real), "fake": Counter(fake)}


print("=" * 92)
print("CORROBORATION RULE SWEEP  (threshold x minimum elevated families)")
print(f"  leak-free corpus: {len(v2)} rows | operator natural-audio reals: {len(op)} rows")
print("=" * 92)
print(f"{'thr':>6} {'minFam':>7} | {'v2 FP':>6} {'v2 FN':>6} {'fakeDNU':>8} {'realTRUST':>10} "
      f"| {'OPFP':>5} {'OPTRUST':>8}   note")
best = []
for thr in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
    for mf in (1, 2, 3):
        m = evaluate(v2, thr, mf)
        mo = evaluate(op, thr, mf)
        ok = m["fp"] == 0 and m["fn"] == 0 and mo["fp"] == 0
        note = "CLEAN" if ok else ""
        print(f"{thr:>6.2f} {mf:>7} | {m['fp']:>6} {m['fn']:>6} {m['dnu_fake']:>8} "
              f"{m['trust_real']:>10} | {mo['fp']:>5} {mo['trust_real']:>8}   {note}")
        if ok:
            best.append((thr, mf, m["dnu_fake"], m["trust_real"],
                         dict(m["real"]), dict(m["fake"])))

print()
print("=" * 92)
print("CLEAN CONFIGS (0 false positives AND 0 false negatives on the leak-free corpus)")
print("=" * 92)
if not best:
    print("  none in the swept grid")
for thr, mf, dnu, tr, rv, fv in best:
    print(f"  thr={thr:.2f} min_families={mf} -> fakeDNU={dnu} realTRUST={tr}")
    print(f"      real={rv}")
    print(f"      fake={fv}")

print()
print("=" * 92)
print("WHY: how many families are elevated, per row? (thr=0.30)")
print("=" * 92)
tally = Counter()
for r in v2:
    n, c = elevated_families(r, 0.30)
    tally[(r["label"], n)] += 1
for lab in (0, 1):
    for n in range(0, 5):
        if tally[(lab, n)]:
            tag = "REAL" if lab == 0 else "FAKE"
            print(f"  {tag}: {n} elevated families -> {tally[(lab, n)]} rows")
