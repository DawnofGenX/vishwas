"""The image channel's arbitration: the shipped policy is the best of the
five MEASURED, and the reason is structural.

THE CLAIM THAT WAS WRONG
-------------------------
The 2026-10-21 measurement note said:

> If the two detectors disagree, the honest reading is "uncertain", not
> "risky" -- the current rule treats NYUAD's high as decisive even when
> SPAI says clean. [...] That would drop the 5 NYUAD-only false alarms
> without touching the 8 catches, since those 8 have BOTH detectors
> agreeing (SPAI ~0 and NYUAD ~0.99 -- they agree the image is synthetic).

That was ASSERTED and never measured. It is false, and measurably so.

THE STRUCTURAL REASON
----------------------
The 5 real photos that NYUAD alone flags and the 8 AI images that only
NYUAD catches have the SAME (SPAI, NYUAD) signature:

    real (5): SPAI 0.0000-0.0075, NYUAD 0.6233-0.9993
    AI   (8): SPAI 0.0000-0.3040, NYUAD 0.9830-0.9995

SPAI low + NYUAD high. No rule over those two numbers can keep the 8 and
drop the 5. The only third number available, freqband, is INVERTED (real
0.4660 vs AI 0.3850 median) so it points the wrong way.

MEASURED — five policies, real FusionEngine, 60 images
--------------------------------------------------------
    policy                                    fakeTRUST  realFALSEALARM
    0. as shipped (SPAI low AND NYUAD low)         3/30        12/30
    1. agreement-only (both must say real)         3/30        12/30
    2. SPAI-only veto (ignore NYUAD)              11/30         7/30
    3. NYUAD-only veto (ignore SPAI)              15/30         5/30
    4. either-low (a veto needs BOTH to flag)     23/30         0/30

Every loosening that drops the false alarms EXPLODES the false-cleans --
by 8, 12 and 20 images respectively. The false-clean is the dangerous
direction: it tells the user "this AI image is genuine". Trading 8 real
catches for 5 fewer false alarms is a good trade; trading 20 catches for
5 is not.

The shipped policy (0) and "agreement-only" (1) are equivalent, because
"both must say real" IS "SPAI low AND NYUAD low" -- the AND of two
low-readings. The note mistook it for something weaker.

WHAT WOULD ACTUALLY SEPARATE THEM
----------------------------------
A THIRD signal, not a re-arbitration of these two:
  * a different AI-image model whose errors do not overlap NYUAD's
    (gigagan is NYUAD's blind spot: 7/8 read below 0.5);
  * a provenance check (EXIF/metadata), which is a different KIND of
    evidence rather than another reading of the pixels;
  * a content check (does the image contain a face at all -- NYUAD is a
    face-centric model and may be unreliable on scenes without one).

This test pins the MEASURED operating point so that any future change to
the image arbitration has to beat these numbers, not just feel better.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import vishwas.fusion as F
from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine

ROWS = pathlib.Path("/home/hermes/.hermes/cache/scratch/img_rows.json")
fe = FusionEngine()
_SHIPPED_IMAGE_CLEAN = F._image_clean


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


def _fb_ok(by_name):
    return F._probe(by_name, "frequency_band_analysis", "prob_deepfake", 0.0) <= 0.60


def _evaluate(policy):
    original = F._image_clean
    F._image_clean = policy
    try:
        eng = FusionEngine()
        res = [(r["label"], eng.decide("image_facecheck", _checks(r)).verdict.value)
               for r in _rows()]
    finally:
        F._image_clean = original
    real = [v for l, v in res if l == 0]
    fake = [v for l, v in res if l == 1]
    return {
        "fake_trust": sum(1 for v in fake if v == "trust"),
        "real_caution": sum(1 for v in real if v == "caution"),
        "n_real": len(real), "n_fake": len(fake),
    }


# ---------------------------------------------------------------------------
# 1. The two groups are structurally indistinguishable.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS.exists(), reason="image corpus not scored on this host")
def test_the_nyuad_only_groups_are_indistinguishable():
    """The 5 real photos NYUAD alone flags and the 8 AI images only NYUAD
    catches share the same (SPAI low, NYUAD high) signature. This is WHY no
    re-arbitration of these two detectors can help -- and it is the fact
    the 2026-10-21 note got wrong."""
    rows = _rows()
    fa = [r for r in rows if r["label"] == 0 and r["verdict"] != "trust"
          and r["spai"] <= 0.45 and r["nyuad"] >= 0.50]
    ca = [r for r in rows if r["label"] == 1 and r["spai"] < 0.45 and r["nyuad"] >= 0.50]
    assert len(fa) == 5, f"expected 5 NYUAD-only false alarms, got {len(fa)}"
    assert len(ca) == 8, f"expected 8 NYUAD-only catches, got {len(ca)}"
    for r in fa + ca:
        assert r["spai"] <= 0.45 and r["nyuad"] >= 0.50, (
            f"{r['name']} does not have the shared (SPAI low, NYUAD high) "
            "signature -- the groups have become separable and the "
            "indistinguishability argument no longer holds")


# ---------------------------------------------------------------------------
# 2. The measured policy table. Pin the numbers, not just the shape.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS.exists(), reason="image corpus not scored on this host")
def test_the_shipped_arbitration_is_the_best_measured_policy():
    """Every loosening of the arbitration that reduces the false alarms
    costs far more false-cleans than it saves. Pin the whole table so a
    future change has to beat it."""
    rows = _rows()

    def p_shipped(by_name):
        return _SHIPPED_IMAGE_CLEAN(by_name)

    def p_spai_only(by_name):
        return (F._probe(by_name, "image_face_forensics", "prob_deepfake", 1.0)
                <= 0.45 and _fb_ok(by_name))

    def p_nyuad_only(by_name):
        return (F._probe(by_name, "nyuad_image_detector", "prob_deepfake", 1.0)
                < 0.50 and _fb_ok(by_name))

    def p_either_low(by_name):
        spai = F._probe(by_name, "image_face_forensics", "prob_deepfake", 1.0)
        ny = F._probe(by_name, "nyuad_image_detector", "prob_deepfake", 1.0)
        return (spai <= 0.45 or ny < 0.50) and _fb_ok(by_name)

    shipped = _evaluate(p_shipped)
    spai_only = _evaluate(p_spai_only)
    nyuad_only = _evaluate(p_nyuad_only)
    either_low = _evaluate(p_either_low)

    # The shipped policy's measured operating point.
    assert shipped["fake_trust"] == 3, shipped
    assert shipped["real_caution"] == 12, shipped

    # Every loosening trades a small fall in false alarms for a MUCH larger
    # rise in false-cleans. The false-clean is the dangerous direction.
    for name, m in (("SPAI-only", spai_only), ("NYUAD-only", nyuad_only),
                    ("either-low", either_low)):
        assert m["fake_trust"] > shipped["fake_trust"] + 3, (
            f"{name} unexpectedly did not cost false-cleans: {m}")
        assert m["real_caution"] <= shipped["real_caution"], (
            f"{name} unexpectedly improved the false alarms: {m}")


# ---------------------------------------------------------------------------
# 3. The shipped policy must actually veto from EITHER detector.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS.exists(), reason="image corpus not scored on this host")
def test_either_detector_can_veto_trust():
    """A lone SPAI-high real photo and a lone NYUAD-high real photo must BOTH
    be denied TRUST. This is the behaviour the note proposed to weaken, and
    it is the behaviour that keeps the 8 NYUAD-only catches caught."""
    rows = _rows()
    spai_veto = [r for r in rows if r["label"] == 0 and r["spai"] > 0.45]
    nyuad_veto = [r for r in rows if r["label"] == 0 and r["spai"] <= 0.45
                  and r["nyuad"] >= 0.50]
    assert spai_veto and nyuad_veto, "expected both veto populations"
    for r in spai_veto + nyuad_veto:
        d = fe.decide("image_facecheck", _checks(r))
        assert d.verdict is not Verdict.TRUST, (
            f"{r['name']} (spai={r['spai']:.4f} nyuad={r['nyuad']:.4f}) was "
            "allowed to read TRUST despite a detector flagging it")
