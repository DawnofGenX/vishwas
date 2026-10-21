#!/usr/bin/env python3
"""MEASURE the effect of freqband's sign on the image channel, end to end.

The measurement says freqband is ANTI-correlated: real photos score HIGHER
(median 0.4625) than AI images (median 0.3990). AUC(fake=high) = 0.1222.
Yet it carries a POSITIVE weight (0.5) in the image_facecheck weight map, so
it pushes genuine photos toward MORE risk.

This script answers the question that matters: does FLIPPING the sign
actually improve the verdicts, or is the effect negligible?

It replays the measured per-image values through the real FusionEngine
under three configurations:
  A. as shipped        (freqband weight +0.5, no calibration)
  B. sign flipped      (freqband weight -0.5)
  C. freqband removed  (the honest move if it carries no information)

and reports the confusion matrix for each.

The signal is emitted by the capability as `prob_deepfake`; flipping the
SIGN is a fusion-layer concern only (the capability is untouched), so this
is modelled by changing the weight and/or adding a calibration offset.

Usage: python tools/eval_freqband_sign.py
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE

ROWS = pathlib.Path("/home/hermes/.hermes/cache/scratch/img_rows.json")
# The file is a single-line JSON ARRAY (not JSONL), so it must be parsed as
# one document rather than line by line.
rows = json.loads(ROWS.read_text())
base_w = dict(WEIGHTS["image_facecheck"])
base_c = {k: v for k, v in _CALIBRATE.items()}


def checks_for(r):
    """Real CheckResult objects, carrying the measured values."""
    return [
        C("image_integrity", "cheap", "ok", {"format": "jpeg"}, "m"),
        C("frequency_band_analysis", "mid", "ok",
          {"prob_deepfake": r["freqband"], "band_anomaly": r["freqband"]}, "m"),
        C("image_face_forensics", "heavy", "ok",
          {"prob_deepfake": r["spai"]}, "m"),
        C("nyuad_image_detector", "heavy", "ok",
          {"prob_deepfake": r["nyuad"]}, "m"),
    ]


def evaluate(weights, calib):
    WEIGHTS["image_facecheck"] = weights
    _CALIBRATE.update(calib)
    fe = FusionEngine()
    res = []
    for r in rows:
        d = fe.decide("image_facecheck", checks_for(r))
        res.append((r["label"], d.verdict.value))
    real = [v for l, v in res if l == 0]
    fake = [v for l, v in res if l == 1]
    return {
        "real_trust": sum(1 for v in real if v == "trust"),
        "real_caution": sum(1 for v in real if v == "caution"),
        "real_dnu": sum(1 for v in real if v == "do_not_use"),
        "fake_trust": sum(1 for v in fake if v == "trust"),
        "fake_caution": sum(1 for v in fake if v == "caution"),
        "fake_dnu": sum(1 for v in fake if v == "do_not_use"),
        "n_real": len(real), "n_fake": len(fake),
        "real": Counter(real), "fake": Counter(fake),
    }


def restore():
    WEIGHTS["image_facecheck"] = base_w
    _CALIBRATE.clear()
    _CALIBRATE.update(base_c)


def show(name, m):
    # The image channel CAPS at CAUTION by design, so the meaningful metrics
    # are: a fake reading TRUST (the false-clean -- the dangerous direction)
    # and a real reading CAUTION/DNU (the false-alarm).
    fc = m["fake_trust"]
    fa = m["real_caution"] + m["real_dnu"]
    print(f"{name:<34s} fakeTRUST={fc:2d}/{m['n_fake']}  realFALSEALARM={fa:2d}/{m['n_real']}  "
          f"fakeCaught={m['n_fake']-fc:2d}")
    print(f"{'':<34s} real={dict(m['real'])}")
    print(f"{'':<34s} fake={dict(m['fake'])}")


print("=" * 92)
print("FREQ BAND SIGN — measured end-to-end effect on the image channel")
print(f"  corpus: {len(rows)} images (30 AI / 30 real), production paths")
print("=" * 92)

# A. as shipped
m = evaluate(base_w, base_c); restore()
show("A. as shipped (w=+0.5)", m)

# B. sign flipped: the SAME signal, negated.
# A negative weight in a positive-weight map flips its contribution, and the
# calibration is identity so the raw value is used as-is.
B_W = {"freqband.prob": -0.5, "faceforensics.prob": 1.0}
m = evaluate(B_W, base_c); restore()
show("B. sign flipped (w=-0.5)", m)

# B2. flipped via calibration instead of a negative weight, so the weight map
# stays structurally uniform (all weights positive) and the SIGN lives in
# the affine table where the other calibrations live.
B2_C = dict(base_c)
B2_C["freqband.prob"] = (-1.0, 1.0)   # 1 - fb
B2_W = {"freqband.prob": 0.5, "faceforensics.prob": 1.0}
m = evaluate(B2_W, B2_C); restore()
show("B2. flipped via calibration 1-fb", m)

# C. removed entirely
C_W = {"faceforensics.prob": 1.0}
m = evaluate(C_W, base_c); restore()
show("C. freqband REMOVED", m)

print()
print("=" * 92)
print("READING THE TABLE")
print("=" * 92)
print("The image channel caps at CAUTION, so the two numbers that matter are:")
print("  fakeTRUST      = a fake that reads CLEAN (the false-clean, the")
print("                   operator's declared worst bug)")
print("  realFALSEALARM = a genuine photo that does not read LOW")
print()
print("If flipping the sign does not improve BOTH, the honest move is to")
print("remove freqband rather than to keep a signal whose sign is wrong.")
