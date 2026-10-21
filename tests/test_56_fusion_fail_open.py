"""FAIL-OPEN PATHS IN THE FUSION LAYER (2026-10-21).

A verification tool must never answer "safe" when it could not check. These
three defects each made a DEAD or SILENT detector read as a CLEAN verdict.

C-1  _image_clean returned True when SPAI was ABSENT.
     The docstring says "Any detector missing/flagged -> NOT clean
     (conservative)" but the code did the opposite:
         if eff_c is None or not eff_c.usable():
             return True   # SPAI absent, everything else clean -> trust
     `usable()` is False for status "unavailable" (weights not provisioned),
     "failed" (inference error) AND "degraded" -- so a DEAD SPAI authorised
     TRUST. Measured before the fix: SPAI unavailable + clean integrity +
     freqband 0.44 -> verdict=TRUST score=0.64.
     The only guard was the frequency-band heuristic, which the codebase
     itself documents as dead noise (real 0.463 vs AI 0.399, overlapping).

C-2  _file_clean counted a DEGRADED VirusTotal as a decisive clean scan.
     `CheckResult.usable()` returns True for status "degraded" (base.py:30),
     and a "no VT record for this hash" result is emitted as status
     "degraded" with positives_ratio 0.0 (capabilities/malware_file.py).
     So a file nobody ever looked up was treated as VT-CLEARED.
     Measured before the fix: VT degraded + clean clamscan + clean yara ->
     verdict=TRUST.

C-3  _to_float mapped a NaN posterior to 1.0 -- a CONFIDENT FAKE.
     `input=nan -> _to_float=1.0` (model_adapters.py). A numerically
     saturated detector therefore produced prob_deepfake=1.0 with
     status="ok". A NaN is not evidence of anything.

C-4  _file_clean ignored the RISK-bearing signals it was supposed to gate.
     It only inspected clamscan / yara_x / vt_reputation, but the
     malicious_file weight map also carries quark.score_norm (2.0),
     mobsf.risk_high_plus, apk.prob_malicious, pe.prob_malicious,
     pe.packed_flag and sandbox.malicious. A file that Quark/MoBFS/APK
     statics flagged at 0.95 still reached TRUST.
     Measured before the fix: quark 0.95 -> TRUST; apk 0.95 -> TRUST.

Each test below is written to FAIL against the pre-fix code.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "/home/hermes/vishwas/src")
from vishwas import fusion as F
from vishwas.capabilities.base import CheckResult
from vishwas.fusion import FusionEngine, Verdict


def C(name, status="ok", sig=None, cost="light"):
    return CheckResult(name=name, cost=cost, status=status, signals=sig or {})


@pytest.fixture()
def fe():
    return FusionEngine()


# --------------------------------------------------------------------------
# C-1: a dead SPAI must not authorise TRUST
# --------------------------------------------------------------------------

def test_image_dead_spai_does_not_read_trust(fe):
    """SPAI unavailable (the DEFAULT deployment) must NOT produce TRUST."""
    checks = [
        C("image_integrity", sig={"ok": True}),
        C("frequency_band_analysis", sig={"prob_deepfake": 0.44}),
        C("image_face_forensics", status="unavailable", sig={}, cost="heavy"),
    ]
    d = fe.decide("image_facecheck", checks)
    assert d.verdict is not Verdict.TRUST, (
        f"a DEAD image detector produced TRUST (score {d.score:.4f}) -- the "
        f"image channel fails OPEN"
    )


def test_image_failed_spai_does_not_read_trust(fe):
    """SPAI that RAN and errored must not produce TRUST either."""
    checks = [
        C("image_integrity", sig={"ok": True}),
        C("frequency_band_analysis", sig={"prob_deepfake": 0.40}),
        C("image_face_forensics", status="failed", sig={}, cost="heavy"),
    ]
    d = fe.decide("image_facecheck", checks)
    assert d.verdict is not Verdict.TRUST


def test_image_degraded_spai_does_not_read_trust(fe):
    """SPAI degraded (partial signal) must not produce TRUST."""
    checks = [
        C("image_integrity", sig={"ok": True}),
        C("frequency_band_analysis", sig={"prob_deepfake": 0.40}),
        C("image_face_forensics", status="degraded", sig={"prob_deepfake": 0.1}, cost="heavy"),
    ]
    d = fe.decide("image_facecheck", checks)
    assert d.verdict is not Verdict.TRUST


def test_image_healthy_low_spai_still_reads_trust(fe):
    """The fix must not break the healthy path: a real photo with a working
    low-SPAI detector and an available second signal still reads TRUST."""
    checks = [
        C("image_integrity", sig={"ok": True}),
        C("frequency_band_analysis", sig={"prob_deepfake": 0.40}),
        C("image_face_forensics", sig={"prob_deepfake": 0.01}, cost="heavy"),
        C("nyuad_image_detector", sig={"prob_deepfake": 0.02}, cost="heavy"),
    ]
    d = fe.decide("image_facecheck", checks)
    assert d.verdict is Verdict.TRUST, f"healthy clean image should read TRUST, got {d.verdict}"


# --------------------------------------------------------------------------
# C-2: a degraded VT is "not checked", not "clean"
# --------------------------------------------------------------------------

def test_file_degraded_vt_is_not_a_decisive_clean_scan(fe):
    """VT 'no record for this hash' (status=degraded) must not authorise TRUST."""
    checks = [
        C("clamscan", sig={"detected": False}),
        C("yara_x", sig={"hits_norm": 0.0}),
        C("vt_reputation", status="degraded", sig={"positives_ratio": 0.0}),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict is not Verdict.TRUST, (
        f"a DEGRADED VT (never looked up) produced TRUST (score {d.score:.4f})"
    )


def test_file_ok_vt_clean_still_reads_trust(fe):
    """A VT that RAN and found nothing still authorises TRUST."""
    checks = [
        C("clamscan", sig={"detected": False}),
        C("yara_x", sig={"hits_norm": 0.0}),
        C("vt_reputation", status="ok", sig={"positives_ratio": 0.0}),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict is Verdict.TRUST, f"a scanned-clean file should read TRUST, got {d.verdict}"


# --------------------------------------------------------------------------
# C-3: NaN posteriors must not become a confident FAKE
# --------------------------------------------------------------------------

def test_to_float_nan_is_not_one():
    """A NaN posterior must not be read as prob_deepfake=1.0."""
    from vishwas.model_adapters import _to_float
    v = _to_float(float("nan"))
    assert v is None or not (v == 1.0), f"NaN posterior became {v}"


def test_to_float_infinity_is_not_a_confident_verdict():
    """+/-inf must not become a confident 0.0/1.0."""
    from vishwas.model_adapters import _to_float
    for val in (float("inf"), float("-inf")):
        v = _to_float(val)
        assert v is None or not (v == 0.0 or v == 1.0), f"inf posterior became {v}"


def test_to_float_keeps_a_valid_probability():
    """The fix must not break the normal path."""
    from vishwas.model_adapters import _to_float
    assert _to_float(0.25) == pytest.approx(0.25)
    assert _to_float(0.0) == pytest.approx(0.0)
    assert _to_float(1.0) == pytest.approx(1.0)


def test_to_float_still_sigmaps_a_real_logit():
    """A genuine out-of-range LOGIT must still be mapped, not rejected."""
    from vishwas.model_adapters import _to_float
    v = _to_float(2.5)
    assert v is not None and 0.5 < v < 1.0, f"logit 2.5 -> {v}"


# --------------------------------------------------------------------------
# C-4: a risk-bearing signal that fires must veto TRUST
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,signals", [
    ("quark_engine", {"score_norm": 0.95}),
    ("apk_statics", {"prob_malicious": 0.95}),
    ("pe_statics", {"prob_malicious": 0.95}),
    ("mobsf_apk", {"mobsf_risk_level": "critical"}),
])
def test_file_risk_signal_vetoes_trust(fe, name, signals):
    """A file that a RISK-bearing detector flags must not read TRUST, even
    when ClamAV / YARA / VT all cleared it."""
    checks = [
        C("clamscan", sig={"detected": False}),
        C("yara_x", sig={"hits_norm": 0.0}),
        C("vt_reputation", status="ok", sig={"positives_ratio": 0.0}),
        C(name, sig=signals),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict is not Verdict.TRUST, (
        f"{name} flagged {signals} but the file still read TRUST "
        f"(score {d.score:.4f})"
    )


def test_file_packed_pe_vetoes_trust(fe):
    """A packed PE is a risk signal and must veto TRUST."""
    checks = [
        C("clamscan", sig={"detected": False}),
        C("yara_x", sig={"hits_norm": 0.0}),
        C("vt_reputation", status="ok", sig={"positives_ratio": 0.0}),
        C("pe_statics", sig={"packed": True, "prob_malicious": 0.0}),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict is not Verdict.TRUST
