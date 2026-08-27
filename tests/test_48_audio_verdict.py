"""Audio verdict fixes: aasist3 (proven AUC 0.9967) clean read -> LOW/TRUST;
strong aasist spoof read -> HIGH (never UNVERIFIED via disagreement-abort).
Corroborated on the trusted aasist gate so a spoof is never cleaned.
"""
from __future__ import annotations

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine


def _ck(n, s, status="ok"): return CheckResult(n, "mid", status, s, "")


def _decide(checks): return FusionEngine().decide(target="deepfake_audio", checks=checks)


def test_clean_audio_reads_low():
    # aasist3 low (bonafide) -> TRUST/LOW
    d = _decide([
        _ck("aasist_detector", {"prob_deepfake": 0.001}),
        _ck("audio_offline_features", {"prob_deepfake": 0.3}),
    ])
    assert d.verdict == Verdict.TRUST, d.reasons


def test_spoof_audio_reads_high_not_unverified():
    # aasist3 strong spoof (0.998) even with weak offline disagreement -> HIGH,
    # NOT UNVERIFIED (audio is exempt from the disagreement-abort: aasist trusted)
    d = _decide([
        _ck("aasist_detector", {"prob_deepfake": 0.998}),
        _ck("audio_offline_features", {"prob_deepfake": 0.1}),  # disagreeing weak gate
        _ck("xlsr_audio_detector", {"prob_deepfake": 0.05}),
    ])
    assert d.verdict == Verdict.DO_NOT_USE, d.reasons
    assert d.verdict is not Verdict.UNABLE_TO_VERIFY


def test_missing_aasist_no_false_clean():
    # no aasist evidence -> never trust (can't clean without the trusted gate)
    d = _decide([
        _ck("audio_offline_features", {"prob_deepfake": 0.2}),
    ])
    assert d.verdict is not Verdict.TRUST, d.reasons


def test_mid_aasist_not_clean():
    # aasist 0.5 (ambiguous) -> not clean (only <0.30 trusts)
    d = _decide([
        _ck("aasist_detector", {"prob_deepfake": 0.5}),
        _ck("audio_offline_features", {"prob_deepfake": 0.3}),
    ])
    assert d.verdict is not Verdict.TRUST, d.reasons