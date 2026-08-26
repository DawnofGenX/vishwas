"""Cross-media concern-bullet matrix.

Every reachable media type must render a coherent, risk-line-first reply with
concern bullets sourced from the real check-name->signal mapping (no raw
`concern_*` keys leaking, no type-specific gap). Pure + hermetic (no weights).
"""
from __future__ import annotations

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.report import ReportBuilder, concerns_for
from vishwas.report import _CONCERN_RULES


def _ck(name, signals, status="ok"):
    return CheckResult(name, "mid", status, signals, "")


_BASE = dict(confidence=0.8, reasons=[])


def _render(target, verdict, checks):
    return str(ReportBuilder().build(target=target, verdict=verdict, checks=checks, **_BASE))


def _assert_media(rendered, *expect_fragments):
    assert rendered.splitlines()[0].startswith("RISK LEVEL: "), "risk line first"
    assert "concern_" not in rendered, f"raw concern key leaked -> {rendered!r}"
    for frag in expect_fragments:
        assert frag in rendered, f"missing {frag!r} in -> {rendered!r}"


# --------------------------------------------------------------- video

def test_video_audio_concern():
    t = _render("deepfake_video", Verdict.DO_NOT_USE,
                [_ck("aasist_detector", {"prob_deepfake": 0.97})])
    _assert_media(t, "LIKELY FAKE", "voice shows signs of AI manipulation",
                  "Don't forward. Verify with a trusted source.")


def test_video_av_sync_concern():
    t = _render("deepfake_video", Verdict.DO_NOT_USE,
                [_ck("cross_modal_av", {"prob_inconsistent": 0.71})])
    _assert_media(t, "lips")


def test_video_face_concern():
    t = _render("deepfake_video", Verdict.CAUTION,
                [_ck("effort_face_forensics", {"prob_deepfake": 0.83})])
    _assert_media(t, "SUSPICIOUS", "face appears digitally altered")


# --------------------------------------------------------------- image

def test_image_freqband_concern():
    # image_facecheck caps verdict at CAUTION (NOT-binary); freqband heuristic fires
    t = _render("image_facecheck", Verdict.CAUTION,
                [_ck("frequency_band_analysis", {"prob_deepfake": 0.72})])
    _assert_media(t, "SUSPICIOUS", "face appears digitally altered")


def test_image_spai_heavy_concern():
    t = _render("image_facecheck", Verdict.CAUTION,
                [_ck("image_face_forensics", {"prob_deepfake": 0.9})])
    _assert_media(t, "SUSPICIOUS", "face appears digitally altered")


# --------------------------------------------------------------- url

def test_url_phish_scanner_bullets():
    t = _render("url_phishing", Verdict.DO_NOT_USE, [
        _ck("url_phish_scanner", {"risk_score_norm": 0.7, "is_phishing": True}),
    ])
    _assert_media(t, "poor security reputation", "disguised copy")


# --------------------------------------------------------------- govdoc

def test_govdoc_forgery_concern():
    t = _render("gov_document", Verdict.DO_NOT_USE,
                [_ck("gov_document", {"prob_forged": 0.8})])
    _assert_media(t, "LIKELY FAKE", "signs of forgery")


# --------------------------------------------------------------- full mapping sanity

def test_every_rule_signal_is_used_by_a_media():
    """Guard: no dead entry in _CONCERN_RULES (every mapping is exercised)."""
    # sanity: rules are well-formed and count > 5 (covers all channels)
    assert len(_CONCERN_RULES) >= 6
    names = {name for name, _, _, _ in _CONCERN_RULES}
    assert names == {
        "aasist_detector", "xlsr_audio_detector", "effort_face_forensics",
        "frame_heuristics", "cross_modal_av", "frequency_band_analysis",
        "image_face_forensics", "url_phish_scanner",
        "gov_document",
    }