"""Concern-bullet extraction for the richer user-facing reply.

The ⚠️ bullets in the reply come from the individual CheckResult signals
(prob_deepfake audio/video, prob_inconsistent, positives_ratio, host_string_score,
prob_forged), NOT from the machine-token `reasons` list.
"""
from __future__ import annotations

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.report import concerns_for, _VERDICT_TILE, _recommend_line


def _ck(name, signals, status="ok"):
    return CheckResult(name, "mid", status, signals, "")


# --------------------------------------------------------------- concerns_for

def test_audio_high_yields_voice_bullet():
    checks = [
        _ck("aasist_detector", {"prob_deepfake": 0.97}),
        _ck("effort_face_forensics", {"prob_deepfake": 0.1}),
    ]
    got = concerns_for(checks, "deepfake_video", Verdict.DO_NOT_USE, "en")
    assert "concern_audio_ai" in got
    assert "concern_video_face" not in got


def test_video_face_high_yields_face_bullet():
    checks = [_ck("effort_face_forensics", {"prob_deepfake": 0.83})]
    got = concerns_for(checks, "deepfake_video", Verdict.DO_NOT_USE, "en")
    assert "concern_video_face" in got


def test_av_sync_high_yields_sync_bullet():
    checks = [_ck("cross_modal_av", {"prob_inconsistent": 0.71})]
    got = concerns_for(checks, "deepfake_video", Verdict.DO_NOT_USE, "en")
    assert "concern_av_sync" in got


def test_url_phish_scanner_fires():
    checks = [
        _ck("url_phish_scanner", {"risk_score_norm": 0.7, "is_phishing": True}),
    ]
    got = concerns_for(checks, "url_phishing", Verdict.DO_NOT_USE, "en")
    assert "concern_url_flag" in got
    assert "concern_url_typo" in got


def test_no_fire_returns_empty():
    # CAUTION with no credible signal fires no bullets (the DO_NOT_USE-unverified
    # append is tested separately).
    checks = [_ck("aasist_detector", {"prob_deepfake": 0.1})]
    got = concerns_for(checks, "deepfake_video", Verdict.CAUTION, "en")
    assert got == []


def test_do_not_use_always_adds_unverified_source():
    checks = []
    got = concerns_for(checks, "deepfake_video", Verdict.DO_NOT_USE, "en")
    assert "concern_unverified_source" in got


def test_caution_does_not_add_unverified_source():
    checks = []
    got = concerns_for(checks, "deepfake_video", Verdict.CAUTION, "en")
    assert got == []


def test_cap_at_three_and_deterministic_order():
    checks = [
        _ck("aasist_detector", {"prob_deepfake": 0.97}),
        _ck("effort_face_forensics", {"prob_deepfake": 0.83}),
        _ck("cross_modal_av", {"prob_inconsistent": 0.71}),
        _ck("vt_url_reputation", {"positives_ratio": 0.2}),
    ]
    got = concerns_for(checks, "deepfake_video", Verdict.DO_NOT_USE, "en")
    assert len(got) <= 3
    assert got[0] == "concern_audio_ai"  # priority order


def test_trust_and_unable_have_no_concerns():
    for v in (Verdict.TRUST, Verdict.UNABLE_TO_VERIFY):
        got = concerns_for([_ck("aasist_detector", {"prob_deepfake": 0.97})],
                           "deepfake_video", v, "en")
        assert got == []


# -------------------------------------------------------- verdict tile / recommendation

def test_verdict_tile_labels():
    assert _VERDICT_TILE[Verdict.DO_NOT_USE][0] == "LIKELY FAKE"
    assert _VERDICT_TILE[Verdict.CAUTION][0] == "SUSPICIOUS"
    assert _VERDICT_TILE[Verdict.TRUST][0] == "LIKELY GENUINE"
    assert _VERDICT_TILE[Verdict.UNABLE_TO_VERIFY][0] == "UNVERIFIED"


def test_recommend_line_do_not_use():
    line = _recommend_line(Verdict.DO_NOT_USE, "en")
    assert "Don't forward" in line


def test_risk_line_still_first_for_all_verdicts():
    from vishwas.report import ReportBuilder
    rb = ReportBuilder()
    for verdict, label in _VERDICT_TILE.items():
        r = rb.build(target="deepfake_video", verdict=verdict, confidence=0.8,
                     reasons=[], checks=[_ck("aasist_detector", {"prob_deepfake": 0.9})],
                     lang="en")
        assert str(r).splitlines()[0].startswith("RISK LEVEL: "), verdict


def test_build_renders_localized_bullets_not_keys():
    from vishwas.report import ReportBuilder
    rb = ReportBuilder()
    checks = [
        _ck("aasist_detector", {"prob_deepfake": 0.97}),
        _ck("effort_face_forensics", {"prob_deepfake": 0.83}),
    ]
    r = rb.build(target="deepfake_video", verdict=Verdict.DO_NOT_USE, confidence=0.8,
                 reasons=[], checks=checks, lang="en")
    text = str(r)
    assert "the voice shows signs of AI manipulation" in text  # localized bullet
    assert "concern_audio_ai" not in text  # no raw key leaked
    assert "Don't forward. Verify with a trusted source." in text


def test_tile_risk_all_caps_matches_risk_line():
    from vishwas.report import ReportBuilder
    rb = ReportBuilder()
    r = rb.build(target="deepfake_video", verdict=Verdict.DO_NOT_USE, confidence=0.8,
                 reasons=[], checks=[], lang="en")
    text = str(r)
    assert "LIKELY FAKE · HIGH RISK" in text  # all-caps risk matches RISK LEVEL styling
    assert "High Risk" not in text and "high Risk" not in text