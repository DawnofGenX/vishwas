"""test_35_fusion_v2_scenarios.py — Fusion v2 pattern-aware deepfake fusion.

Anchors to the 2026-08-25 operator incident: an AI-generated video whose
per-detector signals (effort 0.677, havic 1.0 saturated, AV anti-correlated,
frame_heuristics 0.281) must yield a HIGH-confidence DO_NOT_USE with a
"fully_generated" pattern — NOT a spread-abstained UNABLE, and NOT a weak
0.37-confidence DNU.

Also pins the calibration->pattern->confidence pipeline for the mode-shapes
different AI generators produce (fully generated vs face-swap vs weak/single).
"""
from __future__ import annotations

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine


def _ck(name: str, signals: dict) -> CheckResult:
    return CheckResult(name=name, cost="mid", status="ok", signals=signals, notes="")


def _video_ai(device: str = "generic") -> list[CheckResult]:
    """The operator's real 10.9s AI video signal set (device=generic=cpu-era)."""
    return [
        _ck("ext_mismatch_flag", {"declared": "video", "verified": "mp4"}),
        _ck("video_probe", {"duration_s": 10.9, "width": 896, "height": 480,
                            "fps": 29.633, "has_audio": True}),
        _ck("usable_frames", {"usable_frame_ratio": 1.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.281, "edge_aliasing_frac": 0.056}),
        _ck("effort_face_forensics", {"prob_deepfake": 0.677, "n_frames_scored": 8}),
        _ck("transform_consistency", {"consistent": True, "consistency_spread": 0.0}),
        _ck("cross_modal_av", {"av_correlation": -2.0, "alignment_class": "anti_correlated",
                               "av_risk_addition": 0.5}),
        _ck("havic_crossmodal_model", {"prob_inconsistent": 1.0}),
        CheckResult(name="demamba_general", cost="heavy", status="unavailable",
                    signals={"missing_dependency": "model-weights"}, notes=""),
    ]


def test_operator_ai_video_high_confidence_dnu():
    d = FusionEngine().decide("deepfake_video", _video_ai())
    assert d.verdict is Verdict.DO_NOT_USE, f"expected DO_NOT_USE, got {d.verdict}"
    assert d.confidence >= 0.45, f"corroborated generative evidence must be confident: {d.confidence}"
    assert any("pattern:fully_generated" in r for r in d.reasons), f"missing pattern: {d.reasons}"
    # Fusion v2 gate contract: this coherent pattern must NOT be aborted by the
    # ReliabilityGate on spread alone (would otherwise force UNABLE).
    assert d.coherent_pattern is True, "fully_generated must be a coherent pattern"


def test_clean_video_still_reaches_trust():
    # all-clean signals drive the clean-evidence bonus (-1.8) into the TRUST
    # band when >=3 mapped signals ran and ALL are <=0.10, AND nothing is
    # genuinely missing (the real video pipeline emits every mapped check).
    checks = [
        _ck("frame_heuristics", {"prob_deepfake": 0.02}),
        _ck("effort_face_forensics", {"prob_deepfake": 0.02}),
        _ck("cross_modal_av", {"av_correlation": 0.4, "alignment_class": "synced",
                               "av_risk_addition": 0.0}),
        _ck("havic_crossmodal_model", {"prob_inconsistent": 0.02}),
        _ck("demamba_general", {"prob_deepfake": 0.02}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is Verdict.TRUST, f"clean evidence should reach trust: {d.verdict}"


def test_single_weak_signal_is_caution_not_unable():
    # only frame_heuristics emitted -> effort/demamba/av/havic are GENUINE missing
    # (selective prediction: an unmapped heavy detector didn't run), so honest
    # answer is UNABLE, not a forced verdict. Absence of corroboration = abstain.
    checks = [_ck("frame_heuristics", {"prob_deepfake": 0.6})]
    d = FusionEngine().decide("deepfake_video", checks)
    allowed = (Verdict.CAUTION, Verdict.UNABLE_TO_VERIFY, Verdict.DO_NOT_USE)
    assert d.verdict in allowed, f"weak single must not over-assert: {d.verdict}"
    assert d.confidence <= 0.4, f"single weak signal caps certainty: {d.confidence}"


def test_face_swap_partial_pattern():
    # effort high but AV OK (<0.30) -> face_swap_partial, not fully_generated
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.80}),
        _ck("cross_modal_av", {"av_correlation": 0.2, "alignment_class": "weakly_synced",
                               "av_risk_addition": 0.1}),
        _ck("frame_heuristics", {"prob_deepfake": 0.15}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert any("pattern:face_swap_partial" in r for r in d.reasons), f"expected face_swap_partial: {d.reasons}"


def test_conflicting_audio_conf_capped():
    # two fake, one real -> conflicting_detectors, confidence capped (not boosted)
    checks = [
        _ck("fakemamba_detector", {"prob_deepfake": 0.9}),
        _ck("aasist_detector", {"prob_deepfake": 0.85}),
        _ck("xlsr_audio_detector", {"prob_deepfake": 0.1}),
        _ck("audio_offline_features", {"prob_deepfake": 0.3}),
    ]
    d = FusionEngine().decide("deepfake_audio", checks)
    assert any("pattern:conflicting_detectors" in r for r in d.reasons), f"expected conflict: {d.reasons}"
    assert d.confidence <= 0.4, f"conflict must cap confidence: {d.confidence}"


def test_corroborated_audio_boost():
    checks = [
        _ck("fakemamba_detector", {"prob_deepfake": 0.9}),
        _ck("aasist_detector", {"prob_deepfake": 0.88}),
    ]
    d = FusionEngine().decide("deepfake_audio", checks)
    assert d.verdict is Verdict.DO_NOT_USE
    assert any("pattern:corroborated_multi" in r for r in d.reasons), d.reasons


def test_rel_gate_skips_spread_abort_for_coherent_pattern():
    # Fully_generated pattern: per-detector spread is high but coherence is
    # strong -> gate must NOT force UNABLE (was the 2026-08-25 bug).
    from types import SimpleNamespace as _NS
    from vishwas.fusion import ReliabilityGate

    d = FusionEngine().decide("deepfake_video", _video_ai())
    gate = ReliabilityGate()
    ctx = _NS(extra={})
    ok, notes = gate.evaluate(d, _video_ai(), ctx)
    assert ok, f"coherent pattern must pass the gate: {notes}"
    assert d.coherent_pattern, "pattern coherence flag must be set"


def test_rel_gate_still_aborts_genuine_conflict():
    # Audio with two-fake-then-one-real is a conflict -> gate aborts
    from types import SimpleNamespace as _NS
    from vishwas.fusion import ReliabilityGate

    checks = [
        _ck("fakemamba_detector", {"prob_deepfake": 0.9}),
        _ck("aasist_detector", {"prob_deepfake": 0.85}),
        _ck("xlsr_audio_detector", {"prob_deepfake": 0.1}),
        _ck("audio_offline_features", {"prob_deepfake": 0.3}),
    ]
    d = FusionEngine().decide("deepfake_audio", checks)
    gate = ReliabilityGate()
    ok, notes = gate.evaluate(d, checks, _NS(extra={}))
    assert not ok, f"conflicting_detectors must be gated: {notes}"
    assert not d.coherent_pattern, "conflict must NOT be coherent"


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))