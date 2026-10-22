"""test_35_fusion_v2_scenarios.py — Fusion v2 pattern-aware deepfake fusion.

Anchors to the 2026-08-25 operator incident: an AI-generated video whose
per-detector signals (effort 0.677, havic 1.0 saturated, AV anti-correlated,
frame_heuristics 0.281) must yield a HIGH-confidence DO_NOT_USE with a
"fully_generated" pattern — NOT a spread-abstained UNABLE, and NOT a weak
0.37-confidence DNU.

Also pins the calibration->pattern->confidence pipeline for the mode-shapes
different AI generators produce (fully generated vs face-swap vs weak/single).

2026-10-22 — THE AV FIXTURE HERE USED A MEASUREMENT ARTEFACT.
--------------------------------------------------------------------
The fixture's `av_correlation: -2.0` was the SENTINEL the correlation
function returned when it could not compute a correlation (a constant
envelope). The operator's AI videos have a COMPLETELY SILENT audio track
(RMS 0.0), so their correlation was undefined and the pipeline reported
them as "anti-correlated" -- which the fusion layer then read as the
strongest possible deepfake evidence.

So the fixture asserted a verdict that DEPENDED ON A BUG. It is now a
REAL anti-correlation (r = -0.45, inside the -0.25..-0.6 band the pipeline
actually produces), and the guarantee is kept where it belongs: a fake
whose video-side signals AGREE with an elevated cross-modal reading must
still read DO_NOT_USE.

The honest outcome for a SILENT AI video is covered separately, in
test_66_silence_not_anticorrelated.py: without a measurable cross-modal
signal, and with the effort alone below the threshold, the video reads
CAUTION -- which is the truthful answer for a detector that cannot see
the evidence.
"""
from __future__ import annotations

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine


def _ck(name: str, signals: dict, status: str = "ok") -> CheckResult:
    return CheckResult(name=name, cost="mid", status=status, signals=signals, notes="")


#: A REAL anti-correlation, inside the -0.25..-0.6 band the pipeline
#: actually produces (the operator's AI videos measured -0.299 and -0.443).
_REAL_ANTI_CORRELATION = -0.45


def _video_ai(device: str = "generic") -> list[CheckResult]:
    """The operator's real 10.9s AI video signal set (device=generic=cpu-era).

    2026-10-22: the AV entry carries a REAL anti-correlation, not the old
    -2.0 sentinel (see the module docstring), and the video-side signals
    are STRONG so that they CORROBORATE the cross-modal reading rather
    than being propped up by it.

    The effort value is 0.844, which calibrates to 0.444 — ABOVE the 0.40
    corroboration-gate threshold, so the elevated cross-modal reading is
    treated as agreeing evidence instead of an uncorroborated one."""
    return [
        _ck("ext_mismatch_flag", {"declared": "video", "verified": "mp4"}),
        _ck("video_probe", {"duration_s": 10.9, "width": 896, "height": 480,
                            "fps": 29.633, "has_audio": True}),
        _ck("usable_frames", {"usable_frame_ratio": 1.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.281, "edge_aliasing_frac": 0.056}),
        _ck("effort_face_forensics", {"prob_deepfake": 0.844, "n_frames_scored": 8}),
        _ck("transform_consistency", {"consistent": True, "consistency_spread": 0.0}),
        _ck("cross_modal_av", {"av_correlation": _REAL_ANTI_CORRELATION,
                               "alignment_class": "anti_correlated",
                               "av_risk_addition": 0.5}),
        _ck("havic_crossmodal_model", {"prob_inconsistent": 1.0}),
        CheckResult(name="demamba_general", cost="heavy", status="unavailable",
                    signals={"missing_dependency": "model-weights"}, notes=""),
    ]


def test_operator_ai_video_high_confidence_dnu():
    """A CORROBORATED fake reads DO_NOT_USE with high confidence.

    2026-10-22 — WHAT THIS TEST NOW MEANS.

    The ORIGINAL version of this test asserted the operator's real AI video
    reads DO_NOT_USE. That assertion was TRUE, but for a WRONG reason: the
    fixture's `av_correlation: -2.0` was the correlation function's
    UNDEFINED sentinel, and the operator's AI videos have a completely
    SILENT audio track (RMS 0.0). The fusion layer read that sentinel as
    "anti-correlated" and trusted it as the strongest deepfake evidence
    available — so the verdict was carried by a MEASUREMENT ARTEFACT.

    Measured on the operator's real WhatsApp traffic, that artefact was
    indistinguishable from genuine media: 12% of genuine WhatsApp video
    (9 of 74 clips) has the same all-zero audio track and was reported the
    same way. And HAVIC, the other cross-modal signal, is saturated on
    genuine traffic (>= 0.95 on 42 of 63 genuine clips, median 0.980)
    versus 0.995 on the AI videos, so it does not discriminate either.

    The fixture now carries a REAL anti-correlation (r = -0.45, inside the
    -0.25..-0.6 band the pipeline actually produces) and STRONG video-side
    signals, so the video-side signals CORROBORATE the cross-modal reading
    rather than being propped up by it. That is the guarantee worth having:
    a fake that trips multiple independent signals is still flagged.

    The companion tests below pin the honest outcome for the cases where
    the evidence is NOT there: a silent AI video, and a genuine muted clip.
    """
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
    # effort high (raw 0.98 -> calibrated 1.0*0.98-0.55=0.43 > 0.40 threshold)
    # but AV OK (<0.30) -> face_swap_partial, not fully_generated. Post-2026-08-26
    # the effort threshold maps to calibrated >0.40 on the -0.55 scale.
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.98}),
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


def test_real_talkinghead_not_do_not_use():
    # 2026-08-25 real-video calibration regression (Task 1 / gbdt_report §8):
    # a typical FF++-style REAL talking-head clip — effort mid-band, frameheur
    # low, NO audio track (cross-modal checks unavailable = known gaps) —
    # previously fused to DO_NOT_USE because only present signals divided the
    # weighted sum and effort's old calibration pinned reals at ~0.9 risk.
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.70}),
        _ck("frame_heuristics", {"prob_deepfake": 0.18}),
        _ck("demamba_general", {}, status="unavailable"),
        _ck("cross_modal_av", {}, status="unavailable"),
        _ck("havic_crossmodal_model", {}, status="unavailable"),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.DO_NOT_USE, (
        f"real talking-head must NOT be DNU (got {d.verdict} raw={d.score})")


def test_ai_with_audio_still_dnu():
    # The other half of Task 1's contract: a fully-covered AI video (audio
    # present, av_risk anti-correlated 0.5, havic inconsistent) stays DO_NOT_USE
    # under the recalibrated scale.
    #
    # 2026-10-22: the AV entry used the -2.0 SENTINEL (an unmeasurable
    # correlation, i.e. a silent audio track). It now carries a REAL
    # anti-correlation and STRONG video-side signals (effort 0.844
    # calibrates to 0.444, above the 0.40 corroboration threshold), so this
    # test pins the guarantee that actually matters: a fake whose video-side
    # signals AGREE with the cross-modal reading is still flagged.
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.844}),
        _ck("frame_heuristics", {"prob_deepfake": 0.281}),
        _ck("cross_modal_av", {"av_correlation": _REAL_ANTI_CORRELATION,
                               "alignment_class": "anti_correlated",
                               "av_risk_addition": 0.5}),
        _ck("havic_crossmodal_model", {"prob_inconsistent": 1.0}),
        _ck("demamba_general", {}, status="unavailable"),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is Verdict.DO_NOT_USE, f"AI video must stay DNU: {d.verdict}"
    assert any("pattern:fully_generated" in r for r in d.reasons), d.reasons


def test_a_silent_ai_video_is_judged_on_its_video_side_signals_alone():
    """What an UNMEASURABLE cross-modal check must and must not do.

    The operator's AI videos have a completely SILENT audio track (RMS 0.0),
    so the cross-modal correlation is UNDEFINED. The pipeline reports that
    as `unknown` with a neutral av_risk_addition of 0.0 — it does not claim
    "anti-correlated".

    So the verdict must be carried ENTIRELY by the video-side signals. This
    test pins that in two ways:

      1. The av contribution must be ZERO. The `fully_generated` pattern
         (which fires on av_risk >= 0.45) must NOT appear — that pattern is
         the cross-modal one, and it must not fire on an unmeasurable
         reading. Instead the `face_swap_partial` pattern fires, which is
         driven by the calibrated effort, i.e. by evidence that exists.
      2. The verdict must be the SAME as if the cross-modal check had
         reported a neutral "synced" reading. An unmeasurable check must
         not move the verdict in either direction.

    Measured: raw 0.7156 -> DO_NOT_USE, from effort_cal 0.444 and
    frameheur 0.281 alone. The av term contributes 0.0000 to the weighted
    sum.

    The companion guarantee — a fake whose video-side signals AGREE with
    the cross-modal reading still reads DO_NOT_USE — is pinned by
    test_operator_ai_video_high_confidence_dnu above.
    """
    def _checks(av_signals: dict) -> list[CheckResult]:
        return [
            _ck("ext_mismatch_flag", {"declared": "video", "verified": "mp4"}),
            _ck("video_probe", {"duration_s": 10.9, "width": 896, "height": 480,
                                "fps": 29.633, "has_audio": True}),
            _ck("usable_frames", {"usable_frame_ratio": 1.0}),
            _ck("frame_heuristics", {"prob_deepfake": 0.281, "edge_aliasing_frac": 0.056}),
            _ck("effort_face_forensics", {"prob_deepfake": 0.844, "n_frames_scored": 8}),
            _ck("transform_consistency", {"consistent": True, "consistency_spread": 0.0}),
            _ck("cross_modal_av", av_signals),
            _ck("havic_crossmodal_model", {"prob_inconsistent": 1.0}),
            CheckResult(name="demamba_general", cost="heavy", status="unavailable",
                        signals={"missing_dependency": "model-weights"}, notes=""),
        ]

    unknown_av = {"av_correlation": float("nan"),
                  "alignment_class": "unknown",
                  "av_risk_addition": 0.0}
    neutral_av = {"av_correlation": 0.40,
                  "alignment_class": "synced",
                  "av_risk_addition": 0.0,
                  "av_synced_clean": 0.40}

    d_unknown = FusionEngine().decide("deepfake_video", _checks(unknown_av))
    d_neutral = FusionEngine().decide("deepfake_video", _checks(neutral_av))

    # (1) The cross-modal pattern must not fire on an unmeasurable reading.
    assert not any("pattern:fully_generated" in r for r in d_unknown.reasons), (
        f"the fully_generated (cross-modal) pattern fired on an UNMEASURABLE "
        f"cross-modal reading: {d_unknown.reasons}")

    # (2) An unmeasurable check must not move the verdict in either
    #     direction -- it must read the same as a neutral one.
    assert d_unknown.verdict is d_neutral.verdict, (
        f"an unmeasurable cross-modal check changed the verdict: "
        f"{d_unknown.verdict.value} vs a neutral reading's {d_neutral.verdict.value}")
    assert abs(d_unknown.score - d_neutral.score) < 1e-9, (
        f"an unmeasurable cross-modal check changed the score: "
        f"{d_unknown.score:.6f} vs {d_neutral.score:.6f}")


def test_a_silent_ai_video_with_weak_video_signals_reads_caution():
    """The other half of the honest outcome: when the video-side signals are
    ALSO weak, the verdict is CAUTION — not a fabricated DO_NOT_USE.

    This is the real situation for the operator's AI videos as they measure
    today: their face crop is small and distant, so the calibrated effort is
    0.14-0.25, and their cross-modal check cannot measure anything. There is
    no honest evidence for a DO_NOT_USE, so the system says CAUTION and
    names the gap.
    """
    checks = [
        _ck("ext_mismatch_flag", {"declared": "video", "verified": "mp4"}),
        _ck("video_probe", {"duration_s": 10.9, "width": 896, "height": 480,
                            "fps": 29.633, "has_audio": True}),
        _ck("usable_frames", {"usable_frame_ratio": 1.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.281, "edge_aliasing_frac": 0.056}),
        # the measured face-crop value for the operator's AI videos
        _ck("effort_face_forensics", {"prob_deepfake": 0.647, "n_frames_scored": 8}),
        _ck("transform_consistency", {"consistent": True, "consistency_spread": 0.0}),
        _ck("cross_modal_av", {"av_correlation": float("nan"),
                               "alignment_class": "unknown",
                               "av_risk_addition": 0.0}),
        _ck("havic_crossmodal_model", {"prob_inconsistent": 1.0}),
        CheckResult(name="demamba_general", cost="heavy", status="unavailable",
                    signals={"missing_dependency": "model-weights"}, notes=""),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is Verdict.CAUTION, (
        f"a silent AI video with weak video-side signals must read CAUTION "
        f"(there is no honest evidence for DO_NOT_USE), got {d.verdict} "
        f"raw={d.score}")
    assert not any("pattern:fully_generated" in r for r in d.reasons), (
        f"the fully_generated pattern must not fire on an unmeasurable "
        f"cross-modal reading: {d.reasons}")


def test_a_genuine_muted_clip_is_not_pushed_toward_dnu():
    """The clean-side guarantee the 2026-10-22 widening exists for.

    12% of genuine WhatsApp video (9 of 74 clips) has an all-zero audio
    track. Before the fix, those clips read av=0.5 -- the highest-risk
    cross-modal reading -- and the fusion gate TRUSTED that band, so a
    genuine muted clip could be pushed toward DO_NOT_USE on silence alone.

    Here the clip is genuinely clean on the video side, so it must not reach
    DO_NOT_USE.
    """
    checks = [
        _ck("ext_mismatch_flag", {"declared": "video", "verified": "mp4"}),
        _ck("video_probe", {"duration_s": 8.0, "width": 720, "height": 1280,
                            "fps": 30.0, "has_audio": True}),
        _ck("usable_frames", {"usable_frame_ratio": 1.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.03}),
        _ck("effort_face_forensics", {"prob_deepfake": 0.08}),
        _ck("transform_consistency", {"consistent": True, "consistency_spread": 0.0}),
        # a silent audio track -> the correlation is undefined
        _ck("cross_modal_av", {"av_correlation": float("nan"),
                               "alignment_class": "unknown",
                               "av_risk_addition": 0.0}),
        _ck("havic_crossmodal_model", {"prob_inconsistent": 0.5}),
        CheckResult(name="demamba_general", cost="heavy", status="unavailable",
                    signals={"missing_dependency": "model-weights"}, notes=""),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is not Verdict.DO_NOT_USE, (
        f"a genuine muted clip must NOT be pushed to DNU by an unmeasurable "
        f"cross-modal check (got {d.verdict} raw={d.score})")


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))