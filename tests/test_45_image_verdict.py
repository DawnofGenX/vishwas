"""Image verdict fixes: clean real photos read LOW/trust; flagged AI images read
CAUTION/HIGH (never UNVERIFIED). Corroboration-gated so a lone SPAI read can't
false-trust and can't false-doom.
"""
from __future__ import annotations

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine


def _ck(n, s, status="ok"): return CheckResult(n, "mid", status, s, "")


def _decide(checks):
    return FusionEngine().decide(target="image_facecheck", checks=checks)


def test_clean_real_photo_reads_low():
    # integrity ok + BOTH detectors low => genuine clean photo -> TRUST/LOW
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.30}),
        _ck("image_face_forensics", {"prob_deepfake": 0.10}),
    ])
    assert d.verdict == Verdict.TRUST, d.reasons


def test_clean_real_photo_spai_absent_still_trusts_if_others_clean():
    # SPAI heavy may be unavailable; integrity + freqband clean is enough to trust
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.32}),
    ])
    assert d.verdict == Verdict.TRUST, d.reasons


def test_lone_spai_high_reads_caution_not_unverified():
    # SPAI flags a real photo (0.999) but second signal clean => keep CAUTION/MEDIUM,
    # do NOT demote to UNVERIFIED (that's the reported bug).
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.49}),
        _ck("image_face_forensics", {"prob_deepfake": 0.999}),
    ])
    assert d.verdict == Verdict.CAUTION, d.reasons
    assert d.verdict is not Verdict.UNABLE_TO_VERIFY


def test_ai_image_corroborated_reads_high_or_caution():
    # SPAI high AND freqband high (two independent signals) => strong fake signal
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.72}),
        _ck("image_face_forensics", {"prob_deepfake": 0.90}),
    ])
    # image caps at CAUTION absent a second *trusted* signal; freqband is documented
    # dead-noise so CAUTION is the honest max today, NOT UNVERIFIED.
    assert d.verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE), d.reasons
    assert d.verdict is not Verdict.UNABLE_TO_VERIFY


def test_low_spai_but_high_freqband_reads_caution_not_trust():
    # one detector moderately high -> not clean (no false-trust)
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.65}),
        _ck("image_face_forensics", {"prob_deepfake": 0.20}),
    ])
    assert d.verdict is not Verdict.TRUST, d.reasons


def test_integrity_fail_never_trusts():
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64, "tampered": True}, status="failed"),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.10}),
        _ck("image_face_forensics", {"prob_deepfake": 0.10}),
    ])
    assert d.verdict is not Verdict.TRUST, d.reasons


def test_flux_ai_spai_zero_nyuad_high_not_low():
    # THE fix: a flux AI image reads SPAI ~0.0 (the false-clean). NYUAD (second
    # independent detector) catches it -> must NOT read LOW/trust.
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.38}),
        _ck("image_face_forensics", {"prob_deepfake": 0.0}),        # SPAI misses
        _ck("nyuad_image_detector", {"prob_deepfake": 0.98, "source": "nyuad.subprocess(.venv-ambient)"}),
    ])
    assert d.verdict is not Verdict.TRUST, d.reasons


def test_clean_real_both_low_reads_low():
    # real photo: SPAI low AND NYUAD low -> still LOW
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.40}),
        _ck("image_face_forensics", {"prob_deepfake": 0.03}),
        _ck("nyuad_image_detector", {"prob_deepfake": 0.02, "source": "nyuad.subprocess(.venv-ambient)"}),
    ])
    assert d.verdict == Verdict.TRUST, d.reasons


def test_nyuad_unavailable_still_trusts_on_spai():
    # when the second detector is unavailable (no .venv-ambient), a clean SPAI read
    # still trusts (don't hard-block the whole channel on a missing detector)
    d = _decide([
        _ck("image_integrity", {"sha256": "a" * 64}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.38}),
        _ck("image_face_forensics", {"prob_deepfake": 0.10}),
        _ck("nyuad_image_detector", {}, status="unavailable"),
    ])
    assert d.verdict == Verdict.TRUST, d.reasons