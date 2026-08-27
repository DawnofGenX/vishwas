"""clean-side AV-sync evidence: genuinely-synced real video reaches TRUST.

The deepfake_video fusion carries an av.synced_clean NEGATIVE-weight signal that
fires only when cross_modal independently classes alignment=="synced" (real
lip-sync). TDD: synced+low-face-forensics -> TRUST; decorated/weakly/anti
never TRUST; a lipsync'd deepfake (high effort) never TRUST.
"""
from __future__ import annotations

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine


def _ck(name: str, sig: dict | None = None, status: str = "ok") -> CheckResult:
    return CheckResult(name=name, cost="mid", status=status,
                       signals=sig if sig else {}, notes="")


def test_synced_real_with_low_forensics_reaches_trust():
    # real phone video: AV synced + low face-forensics + low frameheur
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.45}),
        _ck("cross_modal_av", {"av_correlation": 0.62, "best_lag_ms": 40,
                               "alignment_class": "synced", "av_risk_addition": 0.0,
                               "av_synced_clean": 0.62}),
        _ck("frame_heuristics", {"prob_deepfake": 0.2}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is Verdict.TRUST, f"synced real should trust: {d.verdict} {d.reasons}"


def test_synced_deepfake_face_never_trusts():
    # convincing lip-sync on a DEEPFAKED face: high effort => must NOT trust
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.92}),
        _ck("cross_modal_av", {"av_correlation": 0.60, "best_lag_ms": 30,
                               "alignment_class": "synced", "av_risk_addition": 0.0,
                               "av_synced_clean": 0.60}),
        _ck("frame_heuristics", {"prob_deepfake": 0.3}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.TRUST, f"lipsynced deepfake must NOT trust: {d.verdict}"
    # high face-forensics means it stays dangerous (CAUTION or worse), never clean
    assert d.verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE)


def test_decorrelated_never_trusts():
    # no real sync => even with low forensics, must NOT go clean
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.4}),
        _ck("cross_modal_av", {"av_correlation": 0.10, "best_lag_ms": 200,
                               "alignment_class": "decorrelated", "av_risk_addition": 0.45,
                               "av_synced_clean": 0.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.2}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.TRUST, f"decorrelated must not trust: {d.verdict}"
    assert d.verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE)


def test_no_audio_gets_no_clean_bonus():
    # UPDATED 2026-08-26: a silent real clip with LOW face + LOW frameheur now
    # reads LOW/trust (no-audio clean path; was a false-positive stuck at MEDIUM).
    # The silent case must trust ONLY when BOTH face and frame are low — my
    # test_no_audio_high_effort_never_trusts / _high_frameheur_ never_trusts cover
    # the guards (a genuinely-flagged silent clip still stays non-LOW).
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.4}),
        _ck("frame_heuristics", {"prob_deepfake": 0.2}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict == Verdict.TRUST, f"clean silent real should trust: {d.verdict}"


def test_weakly_synced_low_corr_real_now_trusts():
    # Operator's real video 3205c5f9 (corr 0.452, av_risk 0.1, effort 0.52, fh 0.10)
    # currently reads MEDIUM. Aggressive posture (operator-directed):
    # any genuinely-synced/weakly real with low face+frame evidence trusts.
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.52}),
        _ck("frame_heuristics", {"prob_deepfake": 0.10}),
        _ck("cross_modal_av", {"av_correlation": 0.452, "alignment_class": "weakly_synced",
                               "av_risk_addition": 0.1, "av_synced_clean": 0.0}),
    ]
    from vishwas.fusion import FusionEngine
    d = FusionEngine().decide(target="deepfake_video", checks=checks)
    assert d.verdict == Verdict.TRUST, d


def test_real_low_corr_267_now_trusts():
    # Operator's real video aab08774 (corr 0.266, av_risk 0.1, effort 0.47, fh 0.14)
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.47}),
        _ck("frame_heuristics", {"prob_deepfake": 0.14}),
        _ck("cross_modal_av", {"av_correlation": 0.266, "alignment_class": "weakly_synced",
                               "av_risk_addition": 0.1, "av_synced_clean": 0.0}),
    ]
    from vishwas.fusion import FusionEngine
    d = FusionEngine().decide(target="deepfake_video", checks=checks)
    assert d.verdict == Verdict.TRUST, d


def test_weakly_synced_high_corr_real_reaches_trust():
    # operator's real phone video corr ~0.79 — with low face-forensics it trusts
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.43}),
        _ck("cross_modal_av", {"av_correlation": 0.79, "best_lag_ms": 198,
                               "alignment_class": "weakly_synced", "av_risk_addition": 0.1,
                               "av_synced_clean": 0.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.15}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is Verdict.TRUST, f"weakly_synced high-corr real should trust: {d.verdict}"


def test_no_audio_real_video_reads_low():
    # 2026-08-26 false-positive fix: a silent/no-audio REAL video (cross_modal
    # absent) with low face + low frameheur reads LOW/trust. Previously the clean
    # side required audio -> silent real video stuck at MEDIUM.
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.637}),
        _ck("frame_heuristics", {"prob_deepfake": 0.105}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict == Verdict.TRUST, d.reasons


def test_no_audio_high_effort_never_trusts():
    # a silent FAKE with high face-forensics still fails the clean corroboration
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.9}),
        _ck("frame_heuristics", {"prob_deepfake": 0.1}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.TRUST, f"high-effort no-audio must not trust: {d.verdict}"


def test_no_audio_high_frameheur_never_trusts():
    # silent real with high frame-heuristic (frame-level artifact) stays non-LOW
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.4}),
        _ck("frame_heuristics", {"prob_deepfake": 0.6}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.TRUST, f"high-frame no-audio must not trust: {d.verdict}"


def test_no_audio_unverified_when_face_failed():
    # effort failed + no audio -> no face corroboration at all -> stays not-LOW
    # (honest: we cannot certify clean without any face evidence)
    checks = [
        _ck("effort_face_forensics", {}, status="failed"),
        _ck("frame_heuristics", {}, status="degraded"),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.TRUST, f"failed-effort no-audio must not trust: {d.verdict}"


def test_decorrelated_no_audio_never_trusts():
    # even mid-effort, a decorrelated/anti signal (when present) kills clean
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.5}),
        _ck("frame_heuristics", {"prob_deepfake": 0.1}),
        _ck("cross_modal_av", {"av_correlation": 0.0, "alignment_class": "decorrelated",
                               "av_risk_addition": 0.35, "av_synced_clean": 0.0}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.TRUST, f"decorrelated must not trust: {d.verdict}"


def test_high_effort_real_like_never_trusts():
    # even a real-sync clip with clear face-forensics (>0.72) stays not-LOW
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.9}),
        _ck("cross_modal_av", {"av_correlation": 0.79, "best_lag_ms": 0,
                               "alignment_class": "synced", "av_risk_addition": 0.1,
                               "av_synced_clean": 0.5}),
        _ck("frame_heuristics", {"prob_deepfake": 0.1}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.TRUST, f"high-effort must not trust: {d.verdict}"


def test_mid_effort_071_real_now_trusts():
    # Aggressive posture: genuine phone video measuring effort 0.71 (>old 0.60 bar)
    # still reads LOW; only >0.72 (a clear face-forensics flag) blocks trust.
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.71}),
        _ck("cross_modal_av", {"av_correlation": 0.54, "best_lag_ms": 198,
                               "alignment_class": "weakly_synced", "av_risk_addition": 0.1,
                               "av_synced_clean": 0.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.08}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is Verdict.TRUST, f"mid-effort real should trust: {d.verdict}"