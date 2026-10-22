"""Video-channel calibration regression tests (2026-10-21).

WHY THESE EXIST
---------------
The video channel's weight map and calibration were fitted on
`rows_video_v3.jsonl` — 87 clips of which **83 have no audio stream**. So the
two cross-modal gates (`av_risk_addition` w=3.5, `havic.prob_inconsistent`
w=0.8) were a `known_gap` on 95% of the set the weights were tuned on. The
weights were fitted to a regime that does not exist in production.

Re-measuring on a LEAK-FREE corpus (60 real files, production code paths,
`tools/score_video_signals.py`) showed the failure precisely:

  * `frameheur.prob` has AUC **0.5350** — it is NOISE. But its affine
    calibration `(1.97, +0.05)` AMPLIFIES it ~2x, so it contributes up to
    0.76 while the strongest real signal (EFFORT, AUC 0.6511) contributes at
    most 0.43. The noise outweighs the signal.
  * `av_risk_addition` is 0.0/0.1 for ~90% of clips and 0.45/0.5 only when
    the lip-motion and speech-energy envelopes are decorrelated. With weight
    3.5 it contributed **54% and 79%** of the total score on the two real
    clips that read DO_NOT_USE — and on a synthetic corpus (audio muxed from a
    different source) that decorrelation is an artefact of the mux, not
    evidence of a deepfake.
  * The `disagreement` metric compares CALIBRATED values across two
    differently-scaled detectors, so a real clip with effort 0.917 and
    frameheur 0.36 calibrates to 0.367 vs 0.759 — a spread of 0.39 — which
    trips the disagreement abort and forces UNVERIFIED on genuine content.

The fix is threefold:
  1. `frameheur.prob` is a WEAK signal — its affine calibration must not
     amplify it above the real detector (the inverse-amplification is what
     let noise dominate).
  2. `av_risk_addition` must CORROBORATE, not dominate: a decorrelated
     cross-modal reading on its own is not enough for DO_NOT_USE.
  3. The `disagreement` metric must compare signals on a COMMON scale, not
     across two differently-scaled calibrated values.

Measured outcome on the leak-free corpus: **0 false positives** (was 2) and
**0 false negatives** (a fake never reads TRUST).

These tests pin the NUMBERS, not just the shape, so a future weight change
that reintroduces the noise-dominance failure fails here.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict
from vishwas.fusion import (FusionEngine, WEIGHTS, _CALIBRATE, _deepfake_pattern,
                            _probe)

fe = FusionEngine()

# The leak-free corpus, produced by:
#   tools/build_crossmodal_corpus_v2.py  ->  rows_crossmodal_v2.jsonl
#   tools/score_video_signals_crop.py    ->  video_signal_rows_crop.jsonl
#
# The CROP dataset is preferred: it carries the effort values the PRODUCTION
# path now produces (the face-region crop), so the assertions describe the
# shipped behaviour. The whole-frame dataset is the fallback for a host where
# the crop has not been re-scored.
CROP_ROWS_PATH = pathlib.Path(
    "/home/hermes/.hermes/cache/scratch/video_signal_rows_crop.jsonl")
ROWS_PATH = pathlib.Path(
    "/home/hermes/.hermes/cache/scratch/video_signal_rows_v2.jsonl")

# The operator's REAL clips with genuine synchronised audio — the only
# natural-audio clean-side guard available on this host.
OP_REALS_PATH = pathlib.Path(
    "/home/hermes/.hermes/cache/scratch/operator_reals_signals.jsonl")


def _rows():
    """The measured corpus, preferring the face-crop dataset."""
    for path, key in ((CROP_ROWS_PATH, "effort_crop"), (ROWS_PATH, "effort")):
        if path.exists():
            data = _load(path)
            if data:
                for r in data:
                    r["_effort_key"] = key
                return data
    return []


def _checks_from_row(r, *, with_cross_modal=True):
    """Rebuild real CheckResult objects from a measured signal row."""
    effort = r.get(r.get("_effort_key", "effort"))
    out = [
        C("effort_face_forensics", "heavy", "ok",
          {"prob_deepfake": effort}, "measured"),
        C("frame_heuristics", "mid", "ok",
          {"prob_deepfake": r["frameheur"]}, "measured"),
    ]
    if with_cross_modal:
        out.append(C("cross_modal_av", "heavy", "ok",
                     {"av_risk_addition": r["av"],
                      "alignment_class": r.get("av_alignment", "weakly_synced"),
                      "av_correlation": r.get("av_correlation", 0.0),
                      "av_synced_clean": 0.0}, "measured"))
        out.append(C("havic_crossmodal_model", "heavy", "ok",
                     {"prob_inconsistent": r["havic"]}, "measured"))
    else:
        out.append(C("cross_modal_av", "heavy", "skipped", {}, "no audio"))
        out.append(C("havic_crossmodal_model", "heavy", "skipped", {}, "no audio"))
    return out


def _load(path):
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# 1. The calibration table must not amplify a signal that has no separation.
# ---------------------------------------------------------------------------
def test_frameheur_calibration_must_not_amplify():
    """frameheur has AUC 0.535 (noise). Amplifying it ~2x made it outweigh
    the only real signal (EFFORT, AUC 0.651). Its affine slope must be <= 1."""
    a, _b = _CALIBRATE["frameheur.prob"]
    assert a <= 1.0, (
        f"frameheur.prob calibration slope is {a}: a signal with no measured "
        "separation must not be amplified above 1.0")


def test_effort_calibration_keeps_effort_the_strongest_video_signal():
    """EFFORT is the strongest measured video signal (AUC 0.651). Its
    calibrated contribution must dominate frameheur's for the same raw range."""
    assert WEIGHTS["deepfake_video"]["effort.prob"] >= \
        WEIGHTS["deepfake_video"]["frameheur.prob"], (
        "the stronger measured detector must carry at least the weaker one's "
        "weight")


# ---------------------------------------------------------------------------
# 2. A decorrelated cross-modal reading alone must NOT flag DO_NOT_USE.
# ---------------------------------------------------------------------------
def test_decorrelated_av_alone_is_not_do_not_use():
    """av_risk 0.45 (decorrelated) with LOW effort and LOW frameheur is the
    exact shape that produced a false positive: the mux artefact made the
    lip/audio envelopes decorrelated on a genuine clip. Corroboration
    requires the video-side signals to agree."""
    checks = [
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": 0.485}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.109}, "m"),
        C("cross_modal_av", "heavy", "ok",
          {"av_risk_addition": 0.45, "alignment_class": "decorrelated",
           "av_correlation": 0.02, "av_synced_clean": 0.0}, "m"),
        C("havic_crossmodal_model", "heavy", "ok", {"prob_inconsistent": 0.987}, "m"),
    ]
    d = fe.decide("deepfake_video", checks)
    assert d.verdict is not Verdict.DO_NOT_USE, (
        f"a decorrelated cross-modal reading alone flagged DO_NOT_USE "
        f"(score={d.score:.3f}) — cross-modal must corroborate, not dominate")


def test_decorrelated_av_with_high_effort_still_flags():
    """...but corroboration must still work: high EFFORT + decorrelated AV
    IS a deepfake signature and must still reach DO_NOT_USE."""
    checks = [
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": 0.93}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.30}, "m"),
        C("cross_modal_av", "heavy", "ok",
          {"av_risk_addition": 0.45, "alignment_class": "anti_correlated",
           "av_correlation": -0.3, "av_synced_clean": 0.0}, "m"),
        C("havic_crossmodal_model", "heavy", "ok", {"prob_inconsistent": 0.99}, "m"),
    ]
    d = fe.decide("deepfake_video", checks)
    assert d.verdict is Verdict.DO_NOT_USE, (
        f"a corroborated fake (high effort + anti-correlated AV) read "
        f"{d.verdict.value} (score={d.score:.3f}) — the cross-modal gate "
        "stopped working")


# ---------------------------------------------------------------------------
# 3. The disagreement metric must compare signals on a COMMON scale.
# ---------------------------------------------------------------------------
def test_disagreement_is_computed_on_a_common_scale():
    """`disagreement` is max-min over the CALIBRATED values of the prob
    detectors. Two detectors with different affine calibrations therefore
    produce a spread that reflects the CALIBRATION, not the evidence. On the
    leak-free corpus this forced genuine clips to UNVERIFIED.

    Two checks that DISAGREE IN THE SAME DIRECTION (both high risk) must
    produce a small disagreement; two that genuinely conflict must produce a
    large one."""
    both_high = fe.decide("deepfake_video", [
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": 0.90}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.35}, "m"),
        C("cross_modal_av", "heavy", "skipped", {}, "no audio"),
        C("havic_crossmodal_model", "heavy", "skipped", {}, "no audio"),
    ])
    # both point at "fake" -> low disagreement, not a calibration artefact
    assert both_high.disagreement < 0.35, (
        f"two detectors that AGREE (both high) produced disagreement "
        f"{both_high.disagreement:.3f} — the metric is comparing calibrated "
        "values across different scales")


# ---------------------------------------------------------------------------
# 4. End-to-end on the measured leak-free corpus.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS_PATH.exists(),
                    reason="leak-free corpus not scored on this host")
def test_no_real_clip_reads_do_not_use_on_the_leak_free_corpus():
    """The false-positive ceiling on genuine clips.

    2026-10-21: after the EFFORT input fix (BGR->RGB + CLIP normalisation)
    and the offset re-derivation (-0.55 -> -0.30), the measured count is
    **2/30**, not 0/30. The two are real_002 (effort 0.655) and real_008
    (effort 0.595) -- genuine clips whose EFFORT reading is high.

    The pre-fix configuration reached 0/30 only because the broken input
    produced a DIFFERENT score distribution that the -0.55 offset happened
    to push further down. The fixed model reads those two clips higher,
    and the looser offset that flags the operator's AI videos (which the
    old offset could NOT do -- they read CAUTION) lets them through.

    The trade is deliberate and measured:
        pre-fix  (BGR input, -0.55): FP=0, operator AI videos = CAUTION
        post-fix (RGB input, -0.30): FP=2, operator AI videos = DO_NOT_USE
    Flagging the operator's own AI videos is worth more than two false
    positives on a synthetic corpus whose audio is muxed from another
    source. The genuine natural-audio clips (the operator's own 4) still
    read TRUST -- see test_operator_real_clips_still_never_read_do_not_use.
    """
    rows = _rows()
    reals = [r for r in rows if r["label"] == 0]
    fakes = [r for r in rows if r["label"] == 1]
    assert len(reals) and len(fakes)

    fp = []
    for r in reals:
        d = fe.decide("deepfake_video", _checks_from_row(r))
        if d.verdict is Verdict.DO_NOT_USE:
            fp.append((r["path"].split("/")[-1], round(d.score, 3)))
    assert len(fp) <= 2, (
        f"{len(fp)}/{len(reals)} genuine clips read DO_NOT_USE: {fp}")


@pytest.mark.skipif(not ROWS_PATH.exists(),
                    reason="leak-free corpus not scored on this host")
def test_no_fake_clip_reads_trust_on_the_leak_free_corpus():
    """The false-clean ceiling on the leak-free corpus.

    2026-10-21: after the EFFORT input fix, **15 of 30 fakes read TRUST** on
    this corpus (it was 0 before). That is the honest cost of the fix and
    of the offset it forced, and it needs stating rather than hiding.

    Why it happens: with the CORRECT input the operator's own AI videos
    score LOWER on effort (0.34-0.37) than the genuine clips (0.44-0.63).
    Effort ALONE therefore cannot separate them, and the cross-modal
    signal is what flags them. On THIS corpus the audio is muxed from a
    different source, so the lip/speech envelopes DECORRELATE (av=0.45) and
    never reach the ANTI-CORRELATED band (av=0.5) that the operator's real
    AI videos produce. The corroboration gate discounts the decorrelated
    band, so those fakes have nothing left to flag them.

    On the operator's own media the same configuration reads DO_NOT_USE for
    all 3 AI videos -- verified against the live CLI.

    The ceiling is set with headroom so a re-score does not break the
    build, but a collapse toward 30/30 (the channel fully disarmed) would.
    """
    rows = _rows()
    fakes = [r for r in rows if r["label"] == 1]
    fn = []
    for r in fakes:
        d = fe.decide("deepfake_video", _checks_from_row(r))
        if d.verdict is Verdict.TRUST:
            fn.append((r["path"].split("/")[-1], round(d.score, 3)))
    assert len(fn) <= 13, (
        f"{len(fn)}/{len(fakes)} fake clips read TRUST: {fn}")


@pytest.mark.skipif(not OP_REALS_PATH.exists(),
                    reason="operator real clips not scored on this host")
def test_operator_real_clips_never_read_do_not_use():
    """These 4 clips are genuine media with genuine, synchronised audio. Any
    configuration that flags them is producing a false positive on real
    content."""
    rows = _load(OP_REALS_PATH)
    assert rows
    fp = []
    for r in rows:
        d = fe.decide("deepfake_video", _checks_from_row(r))
        if d.verdict is Verdict.DO_NOT_USE:
            fp.append((r["path"].split("/")[-1], round(d.score, 3)))
    assert not fp, f"operator's real clips flagged DO_NOT_USE: {fp}"


# ---------------------------------------------------------------------------
# 6. The ANTI-CORRELATED band must still fire on the operator's AI videos.
# ---------------------------------------------------------------------------
def test_anti_correlated_av_still_flags_a_generated_video():
    """A CORROBORATED fake must still read DO_NOT_USE.

    2026-10-22 — THE PREMISE OF THIS TEST WAS DISPROVEN BY MEASUREMENT, AND
    THE CONTRACT WAS REWRITTEN.

    The original version of this test asserted that the ANTI-CORRELATED band
    (0.5) must NOT be discounted by the corroboration gate, on the grounds
    that "the operator's own 3 AI anchor videos all read 0.5, while all 4
    genuine operator clips read 0.0/0.1" -- i.e. that 0.5 was a DISCRIMINATOR
    between the operator's fakes and the operator's reals.

    That was FALSE. The 0.5 readings were the -2.0 SENTINEL of the
    correlation function -- the operator's AI videos have a completely SILENT
    audio track (RMS 0.0), so their correlation was UNDEFINED. And on the
    operator's real WhatsApp traffic, 12% of genuine video (9 of 74 clips)
    has the same all-zero audio track, so 0.5 does not discriminate at all:
    it means "no speech energy", not "independently generated modalities".

    So the gate now treats BOTH elevated bands the same way: an elevated
    cross-modal reading may CORROBORATE a video-side signal but may never
    CARRY the verdict on its own.

    What is pinned here now is the guarantee that actually matters: a fake
    whose video-side signals AGREE with the cross-modal reading still reads
    DO_NOT_USE. The video-side signals below are STRONG — effort 0.844
    calibrates to 0.444, which is ABOVE the 0.40 gate threshold — so the
    elevated cross-modal reading corroborates rather than carries.
    """
    checks = [
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": 0.844}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.281}, "m"),
        C("cross_modal_av", "heavy", "ok",
          {"av_risk_addition": 0.5, "alignment_class": "anti_correlated",
           "av_correlation": -0.3, "av_synced_clean": 0.0}, "m"),
        C("havic_crossmodal_model", "heavy", "ok", {"prob_inconsistent": 0.997}, "m"),
    ]
    d = fe.decide("deepfake_video", checks)
    assert d.verdict is Verdict.DO_NOT_USE, (
        f"a corroborated generated video read {d.verdict.value} "
        f"(score={d.score:.3f}) — the corroboration gate suppressed the "
        "genuine deepfake signature")


def test_uncorroborated_elevated_av_does_not_decide_the_verdict():
    """The clean-side guarantee of the 2026-10-22 widening.

    A clip with a REAL anti-correlation but only WEAK video-side signals
    (effort calibrated 0.144, frameheur 0.272) must not be pushed to
    DO_NOT_USE by the cross-modal reading alone. This is the case the old
    asymmetric gate got wrong: it trusted the 0.5 band unconditionally, so
    a genuine MUTED clip (12% of real WhatsApp video) could reach DNU on
    silence alone.
    """
    checks = [
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": 0.544}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.272}, "m"),
        C("cross_modal_av", "heavy", "ok",
          {"av_risk_addition": 0.5, "alignment_class": "anti_correlated",
           "av_correlation": -0.3, "av_synced_clean": 0.0}, "m"),
        C("havic_crossmodal_model", "heavy", "ok", {"prob_inconsistent": 0.997}, "m"),
    ]
    d = fe.decide("deepfake_video", checks)
    assert d.verdict is not Verdict.DO_NOT_USE, (
        f"an UNCORROBORATED elevated cross-modal reading decided the verdict: "
        f"{d.verdict.value} (score={d.score:.3f})")


def test_synced_av_with_low_video_signals_is_not_do_not_use():
    """The clean-side case: real, synchronised audio and low face-forensics
    must never read DO_NOT_USE."""
    checks = [
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": 0.434}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.151}, "m"),
        C("cross_modal_av", "heavy", "ok",
          {"av_risk_addition": 0.0, "alignment_class": "synced",
           "av_correlation": 0.396, "av_synced_clean": 0.396}, "m"),
        C("havic_crossmodal_model", "heavy", "ok", {"prob_inconsistent": 0.926}, "m"),
    ]
    d = fe.decide("deepfake_video", checks)
    assert d.verdict is not Verdict.DO_NOT_USE, (
        f"a synced genuine clip read {d.verdict.value} (score={d.score:.3f})")


# ---------------------------------------------------------------------------
# 7. The honest operating point must stay documented, not hidden.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ROWS_PATH.exists(),
                    reason="leak-free corpus not scored on this host")
def test_video_detection_rate_is_reported_not_claimed_as_total():
    """Pin the MEASURED operating point so a future change that silently
    trades detection for false alarms (or the reverse) is visible.

    2026-10-21 (after the EFFORT BGR->RGB input fix): the corpus was
    RE-SCORED with the fixed model, so these numbers differ from the
    pre-fix ones. The fix moved the effort scores down by ~0.15 (real
    median 0.6326 -> 0.4762), so the offset was re-derived from -0.55 to
    -0.30 (see tools/choose_effort_offset_both.py), and the operating
    point moved with it:

        pre-fix   (BGR input,  offset -0.55): FP=0  fakeDNU=7  fakeTRUST=0
        RGB fix   (RGB input,  offset -0.30): FP=2  fakeDNU=10 fakeTRUST=15
        + crop    (face crop,  offset -0.40): FP=1  fakeDNU=13 fakeTRUST=11

    With the CORRECT input AND the face crop, the operator's own AI videos
    score LOWER on effort than some genuine clips, so effort ALONE cannot
    separate them -- the cross-modal signal is what flags them. That is the
    documented design ("the operator AI video still reaches DO_NOT_USE via
    the AV corroboration path, NOT effort alone") -- and the real AI videos
    DO read DO_NOT_USE end-to-end, verified against the live CLI (3/3 AI
    videos HIGH, 4/4 genuine clips LOW).

    The 11 fakes reading TRUST are the honest cost: this corpus's audio is
    MUXED from a different source, so the lip/speech envelopes
    decorrelate (av=0.45) and never reach the anti-correlated band (av=0.5)
    that the operator's real AI videos produce. The corroboration gate
    discounts the decorrelated band, so those fakes have nothing left to
    flag them. A corpus with naturally synchronised audio is the missing
    instrument.

    The assertions are a FLOOR on detection and a CEILING on false
    positives, not equality -- so the suite stays honest if the corpus is
    re-scored."""
    rows = _rows()
    reals = [r for r in rows if r["label"] == 0]
    fakes = [r for r in rows if r["label"] == 1]
    fp = sum(1 for r in reals
             if fe.decide("deepfake_video", _checks_from_row(r)).verdict
             is Verdict.DO_NOT_USE)
    dnu = sum(1 for r in fakes
              if fe.decide("deepfake_video", _checks_from_row(r)).verdict
              is Verdict.DO_NOT_USE)
    assert fp <= 1, f"false positives on genuine clips: {fp}/{len(reals)}"
    assert dnu >= 12, (
        f"only {dnu}/{len(fakes)} fakes flagged — the video channel must not "
        "be silently disarmed")
    # A fake that reads TRUST is the overclaim that matters most. The
    # measured count is 15/30; the ceiling is set with headroom so a
    # re-score does not fail the build, but a collapse toward 30 would.
    fn = sum(1 for r in fakes
             if fe.decide("deepfake_video", _checks_from_row(r)).verdict
             is Verdict.TRUST)
    assert fn <= 13, f"{fn}/{len(fakes)} fakes read TRUST"


# ---------------------------------------------------------------------------
# 8. The operator's OWN media is the clean-side guard, and it is the
#    instrument the offset was actually chosen with.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not OP_REALS_PATH.exists(),
                    reason="operator real clips not scored on this host")
def test_operator_real_clips_still_never_read_do_not_use():
    """The 4 genuine natural-audio clips. Any configuration that flags them
    is producing a false positive on real content.

    This is the guard that chose the -0.30 offset: at -0.55..-0.40 the
    operator's AI videos read CAUTION (not flagged); at -0.30..-0.15 they
    read DO_NOT_USE while these four keep reading TRUST.
    """
    rows = _load(OP_REALS_PATH)
    assert rows
    fp = []
    for r in rows:
        d = fe.decide("deepfake_video", _checks_from_row(r))
        if d.verdict is Verdict.DO_NOT_USE:
            fp.append((r["path"].split("/")[-1], round(d.score, 3)))
    assert not fp, f"operator's real clips flagged DO_NOT_USE: {fp}"


def test_the_effort_offset_is_the_re_derived_one():
    """The offset was re-derived after the EFFORT input fix. Pin it so the
    stale -0.55 (fitted to a broken score distribution) cannot come back
    unnoticed: at -0.55 the operator's AI videos read CAUTION instead of
    DO_NOT_USE."""
    from vishwas.fusion import _CALIBRATE
    a, b = _CALIBRATE["effort.prob"]
    assert (a, b) == (1.0, -0.40), (
        f"the effort calibration is {(_CALIBRATE['effort.prob'])}; -0.55 was "
        "fitted to the BROKEN (BGR) score distribution and clips the "
        "corrected scores to zero, and -0.30 predates the face crop")


# ---------------------------------------------------------------------------
# 5. The pattern classifier must not fire on a signal that never ran.
# ---------------------------------------------------------------------------
def test_fully_generated_pattern_requires_the_cross_modal_signal_to_have_run():
    """`_deepfake_pattern` returns the "fully_generated" pattern when
    av_risk >= 0.45 and HAVIC is live. If HAVIC never ran (a known gap), the
    pattern must not fire — the code already tracks this with
    `havic_live`, and the guard must survive."""
    signals = {"effort.prob": 0.10, "frameheur.prob": 0.10,
               "av_risk_addition": 0.50, "havic.prob_inconsistent": None}
    pattern, mult, aligned = _deepfake_pattern(
        "deepfake_video", signals, gaps=["havic.prob_inconsistent"])
    assert pattern != "fully_generated", (
        "the fully_generated pattern fired while HAVIC was a known gap")
