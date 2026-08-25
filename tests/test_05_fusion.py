"""Fusion determinism + reliability gating.

Pins: (a) feature-vector layout is train/serving-consistent, (b) verdict
threshold bands, (c) selective prediction (abstain on weak/absent evidence),
(d) detector-disagreement penalty, (e) ReliabilityGate conflict rules.
Signal names below are taken verbatim from fusion.WEIGHTS / _SIGNAL_SOURCES.
"""
from __future__ import annotations

import pathlib

import pytest

from vishwas.events import Verdict, InputType
from vishwas.fusion import FusionEngine, ReliabilityGate
from vishwas.capabilities.base import CheckResult
from vishwas.events import JobContext, Artifact


def cr(name, status, **signals):
    return CheckResult(name=name, cost="mid", status=status, signals=signals)


# ---------------------------------------------------------- feature vector --
def test_feature_vector_layout_pairs_value_with_gap_flag():
    engine = FusionEngine()
    checks = [cr("phish_heuristics", "ok", score_norm=0.4, young_domain=False)]
    vec = FusionEngine.feature_vector("url_phishing", checks)
    # one (value, gap) pair per weight key
    from vishwas import fusion as _fm
    n_weights = len(_fm.WEIGHTS["url_phishing"])
    assert len(vec) == 2 * n_weights
    assert all(isinstance(v, float) for v in vec)
    # phish_heuristics present & usable -> value written, gap flag 0
    phish_key = list(_fm.WEIGHTS["url_phishing"].keys()).index("phish.heuristic_score")
    idx = 2 * phish_key
    assert vec[idx] == pytest.approx(0.4)
    assert vec[idx + 1] == 0.0


def test_feature_vector_unknown_target_is_empty_not_crash():
    vec = FusionEngine.feature_vector("does_not_exist", [cr("x", "ok", y=1.0)])
    assert isinstance(vec, list)


# ------------------------------------------------------------------- decide --
def test_no_usable_checks_abstains():
    d = FusionEngine().decide("url_phishing", [cr("vt_url_reputation", "unavailable")])
    assert d.verdict is Verdict.UNABLE_TO_VERIFY
    assert "no_usable_signals" in d.reasons
    assert d.confidence == 0.0


def test_strong_phish_signals_do_not_use():
    checks = [
        cr("phish_heuristics", "ok", score_norm=1.0, young_domain=True),
        cr("vt_url_reputation", "ok", positives_ratio=1.0),
        cr("ssrf_guard", "ok", blocked=True),   # bool->float coerced by _extract? keep numeric form
    ]
    d = FusionEngine().decide("url_phishing", checks)
    assert d.verdict is Verdict.DO_NOT_USE
    assert d.score >= 0.70


def test_benign_low_signal_caution_band():
    checks = [cr("phish_heuristics", "ok", score_norm=0.02, young_domain=False)]
    d = FusionEngine().decide("url_phishing", checks)
    # single cheap signal cannot certify trust -> caution band, abstaining from TRUST
    assert d.verdict in (Verdict.CAUTION, Verdict.UNABLE_TO_VERIFY)


def test_detector_disagreement_penalised_and_reported():
    checks = [
        cr("fakemamba_detector", "ok", prob_deepfake=0.1),
        cr("aasist_detector", "ok", prob_deepfake=0.9),
    ]
    d = FusionEngine().decide("deepfake_audio", checks)
    assert d.disagreement == pytest.approx(0.8)
    assert any("disagreement" in r for r in d.reasons)
    assert d.confidence <= 0.4, "strong disagreement must cap confidence"


def test_all_probes_present_high_prob_do_not_use():
    checks = [
        cr("fakemamba_detector", "ok", prob_deepfake=0.95),
        cr("aasist_detector", "ok", prob_deepfake=0.9),
        cr("ssl_audio_detector", "ok", prob_deepfake=0.92),
        cr("audio_offline_features", "ok", prob_deepfake=0.88),
    ]
    d = FusionEngine().decide("deepfake_audio", checks)
    assert d.verdict is Verdict.DO_NOT_USE
    assert d.disagreement < 0.2  # agreement across detectors


# ------------------------------------------------------------ calibration ----
def test_calibration_applied_changes_raw_probability():
    # identity (no stack, no calib) baseline
    plain = FusionEngine().decide("deepfake_audio", [cr("fakemamba_detector", "ok", prob_deepfake=0.9)])
    # hot temperature t=3 flattens logits toward 0.5 -> lower than uncalibrated 0.9-based squash
    calib = FusionEngine(calibration={"deepfake_audio": {"t": 3.0, "b": 0.0}})
    hot = calib.decide("deepfake_audio", [cr("fakemamba_detector", "ok", prob_deepfake=0.9)])
    assert plain.score != hot.score


# ------------------------------------------------------------- reliability ---
@pytest.fixture
def ctx(tmp_path):
    art = Artifact(path=tmp_path / "a.bin", original_filename="a.bin", declared_type=InputType.FILE)
    return JobContext(job_id="j1", artifact=art, quarantine_root=tmp_path)


def test_gate_zero_usable_signals_fails(ctx):
    g = ReliabilityGate()
    from vishwas.fusion import FusionDecision
    fused = FusionDecision(verdict=Verdict.CAUTION, score=0.5, raw=0.5, disagreement=0.0)
    ok, notes = g.evaluate(fused, [cr("x", "unavailable")], ctx)
    assert ok is False
    assert "zero_usable_signals" in notes


def test_gate_passes_on_clean_consistent_evidence(ctx):
    from vishwas.fusion import FusionDecision
    fused = FusionDecision(verdict=Verdict.CAUTION, score=0.5, raw=0.5, disagreement=0.0)
    ok, notes = ReliabilityGate().evaluate(
        fused, [cr("phish_heuristics", "ok", score_norm=0.5)], ctx)
    assert ok is True and notes == []


def test_gate_blocks_excessive_disagreement(ctx):
    from vishwas.fusion import FusionDecision
    fused = FusionDecision(verdict=Verdict.DO_NOT_USE, score=0.8, raw=0.8, disagreement=0.9)
    ok, notes = ReliabilityGate().evaluate(
        fused, [cr("fakemamba_detector", "ok", prob_deepfake=0.1),
                cr("aasist_detector", "ok", prob_deepfake=0.9)], ctx)
    assert ok is False
    assert any(n.startswith("disagreement=") for n in notes)


def test_gate_flags_authoritative_conflict(ctx):
    """valid signature object but DigiLocker says NOT verified -> conflict."""
    from vishwas.fusion import FusionDecision
    fused = FusionDecision(verdict=Verdict.TRUST, score=0.2, raw=0.2, disagreement=0.0)
    checks = [
        cr("digital_signature", "ok", has_sig_object=True),
        cr("digilocker_verify", "degraded", dl_verified=False),
    ]
    ok, notes = ReliabilityGate().evaluate(fused, checks, ctx)
    assert ok is False
    assert any("conflict" in n for n in notes)
    assert "suppressed TRUST verdict" in notes
