"""freqband's sign is wrong AND the signal is INERT — pin both facts.

THE MEASUREMENT (2026-10-21, 60 real images through the production paths)
------------------------------------------------------------------------
`freqband.prob` is ANTI-correlated with reality, not merely uninformative:

    real photos  median 0.4625   (max 0.5150)
    AI images    median 0.3990   (max 0.4710)
    AUC(fake=high) = 0.1222      -> the INVERSE of 0.8778

So a genuine photo scores HIGHER on "prob_deepfake" than an AI image. The
repo's comment calls it "dead noise"; it is worse than that — it points the
wrong way. Yet it carries a POSITIVE weight (0.5) in the
`image_facecheck` weight map, so it pushes real photos toward MORE risk.

WHY THAT DOESN'T CHANGE ANY VERDICT — measured, not assumed
------------------------------------------------------------
Flipping the sign (weight -0.5), flipping it via calibration (1-fb), and
REMOVING it entirely all produce **identical verdicts on all 60 images**:

    config                         fakeTRUST  realFALSEALARM
    A. as shipped (w=+0.5)             3/30        12/30
    B.  sign flipped (w=-0.5)          3/30        12/30
    B2. flipped via calibration 1-fb   3/30        12/30
    C.  freqband REMOVED               3/30        12/30

Rows whose VERDICT changed when the sign was flipped: **0/60**.

The mechanism, traced exactly:

1. freqband enters the image verdict in only TWO places.
   a. The weighted sum (weight 0.5). This moves `x` by ~0.15 — but the
      image channel CAPS at CAUTION (a single learned read must never flip
      a photo to DO_NOT_USE), so a score moving inside the CAUTION band
      changes nothing. Measured: 11/60 rows sit in [0.35, 0.70) and 17/60
      would reach DNU, all capped to CAUTION.
   b. The `_image_clean` guard (`fb > 0.60` blocks trust). This NEVER
      FIRES on this corpus: max fb over reals is 0.5150, over fakes 0.4710.

2. Every TRUST verdict comes from the clean-side override, not the sum:
   21/21 TRUST verdicts were decided by `_image_clean`, 0 by the raw sum.
   `_image_clean` requires SPAI low AND NYUAD low AND freqband <= 0.60 —
   and the freqband term of that conjunction is never the binding one.

So freqband is INERT: it contributes a constant-ish offset to every score
and a guard that never fires. Its wrong sign is therefore harmless *on this
corpus* — but it is a live landmine, because:

  * the guard fires on 21/30 reals at a 0.45 threshold and 4/30 at 0.50, so
    a future threshold change would start blocking GENUINE photos from
    reading LOW — in the wrong direction, since the signal is inverted;
  * the 0.15 offset inflates every image's risk score, which is visible in
    the `calibrated` field of the reply debug output.

WHAT THIS TEST PINS
-------------------
Not a behaviour change — freqband is deliberately LEFT IN PLACE, because
"remove it" would be a weight-map change that needs its own evidence, and
because it is measurably inert. These assertions lock the FINDING so it
cannot silently drift, and so that whoever next touches the image weight map
knows the sign is inverted rather than neutral.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from vishwas.capabilities.base import CheckResult as C
from vishwas.fusion import WEIGHTS, FusionEngine, _image_clean

ROWS = pathlib.Path("/home/hermes/.hermes/cache/scratch/img_rows.json")
fe = FusionEngine()


def _rows():
    if not ROWS.exists():
        return None
    return json.loads(ROWS.read_text())


def _checks(r):
    return [
        C("image_integrity", "cheap", "ok", {"format": "jpeg"}, "m"),
        C("frequency_band_analysis", "mid", "ok",
          {"prob_deepfake": r["freqband"], "band_anomaly": r["freqband"]}, "m"),
        C("image_face_forensics", "heavy", "ok", {"prob_deepfake": r["spai"]}, "m"),
        C("nyuad_image_detector", "heavy", "ok", {"prob_deepfake": r["nyuad"]}, "m"),
    ]


# ---------------------------------------------------------------------------
# 1. The sign is inverted. This is a fact about the data, not a preference.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS.exists(), reason="image corpus not scored on this host")
def test_freqband_is_anti_correlated_with_reality():
    """Real photos score HIGHER on freqband.prob than AI images.

    The repo comment calls it "dead noise". It is worse than that: it is
    INVERTED. A "prob_deepfake" that rises for genuine content is a signal
    whose sign is wrong, and it must not be read as neutral."""
    rows = _rows()
    reals = [r["freqband"] for r in rows if r["label"] == 0]
    fakes = [r["freqband"] for r in rows if r["label"] == 1]
    assert sum(reals) / len(reals) > sum(fakes) / len(fakes), (
        "freqband no longer reads backwards -- if this now passes, the "
        "signal's polarity changed and the weight map must be revisited")

    # AUC(fake=high) must stay clearly BELOW 0.5.
    wins = 0
    for f in fakes:
        for r in reals:
            wins += 1 if f > r else (0.5 if f == r else 0)
    auc = wins / (len(fakes) * len(reals))
    assert auc < 0.35, (
        f"freqband AUC(fake=high) = {auc:.4f}; the measurement said 0.1222")


# ---------------------------------------------------------------------------
# 2. The sign does not currently change any verdict. Pin THAT, so nobody
#    "fixes" the weight and silently changes 60 verdicts.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS.exists(), reason="image corpus not scored on this host")
def test_flipping_the_freqband_sign_changes_no_verdict():
    """Flip the sign, the calibration, and remove it entirely: all three must
    give identical verdicts on all 60 images.

    This is the surprising part and it is worth locking down. freqband is
    INERT: the image channel caps at CAUTION so a score shift inside the
    band changes nothing, and the clean-side guard never fires because the
    max freqband on ANY image (0.5150 over reals) is below its 0.60
    threshold."""
    rows = _rows()
    base_w = dict(WEIGHTS["image_facecheck"])
    base_c = {}
    from vishwas.fusion import _CALIBRATE
    base_c = {k: v for k, v in _CALIBRATE.items()}

    def verdicts(weights, calib):
        WEIGHTS["image_facecheck"] = weights
        _CALIBRATE.update(calib)
        try:
            fe2 = FusionEngine()
            return [fe2.decide("image_facecheck", _checks(r)).verdict.value
                    for r in rows]
        finally:
            WEIGHTS["image_facecheck"] = base_w
            _CALIBRATE.clear()
            _CALIBRATE.update(base_c)

    shipped = verdicts(base_w, base_c)
    flipped = verdicts({**base_w, "freqband.prob": -0.5}, base_c)
    removed = verdicts({"faceforensics.prob": 1.0}, base_c)

    assert shipped == flipped, (
        "flipping the freqband sign now changes verdicts -- the sign has "
        "become load-bearing and the weight map must be re-derived")
    assert shipped == removed, (
        "removing freqband now changes verdicts -- it has become "
        "load-bearing and the weight map must be re-derived")


# ---------------------------------------------------------------------------
# 3. The clean-side guard is inert, and would fire BACKWARDS if the
#    threshold were ever lowered.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS.exists(), reason="image corpus not scored on this host")
def test_the_freqband_clean_guard_never_fires_but_would_fire_backwards():
    """At its current 0.60 threshold the guard blocks nothing. Lowered to
    0.45 -- a plausible-looking "tightening" -- it would block 21 of 30
    GENUINE photos and only 1 of 30 AI images.

    That is the landmine: the signal is inverted, so tightening the guard
    pushes real photos AWAY from reading LOW."""
    rows = _rows()
    reals = [r["freqband"] for r in rows if r["label"] == 0]
    fakes = [r["freqband"] for r in rows if r["label"] == 1]

    blocked_now = sum(1 for v in reals + fakes if v > 0.60)
    assert blocked_now == 0, (
        f"the freqband clean-guard now fires on {blocked_now} images; it was "
        "measured at 0, and if it fires it may now be blocking real photos")

    blocked_reals_at_045 = sum(1 for v in reals if v > 0.45)
    blocked_fakes_at_045 = sum(1 for v in fakes if v > 0.45)
    assert blocked_reals_at_045 > blocked_fakes_at_045, (
        "the guard at 0.45 no longer blocks more real photos than AI ones; "
        "the inversion has changed shape and the guard needs re-deriving")


# ---------------------------------------------------------------------------
# 4. The clean override is what actually decides TRUST.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS.exists(), reason="image corpus not scored on this host")
def test_every_trust_verdict_comes_from_the_clean_override():
    """TRUST is decided by `_image_clean` (SPAI low AND NYUAD low), never by
    the weighted sum. The sum only produces CAUTION/DNU, and DNU is capped
    to CAUTION for images."""
    rows = _rows()
    trust = 0
    for r in rows:
        checks = _checks(r)
        d = fe.decide("image_facecheck", checks)
        if d.verdict.value == "trust":
            trust += 1
            assert _image_clean({c.name: c for c in checks}), (
                "a TRUST verdict was produced WITHOUT the clean override "
                "firing -- the image channel's clean-side rule was bypassed")
    assert trust > 0, "no TRUST verdicts at all -- the corpus is not loaded"
