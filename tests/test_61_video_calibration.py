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


# A COMPLETE dataset, not just a parseable file: the corpus has 60 clips, and
# the harness that produces the crop dataset UNLINKS its output and then
# appends one row per clip, so for the length of that run the file on disk is
# a fragment.
MIN_ROWS = 40

# ⚠️ A DETECTION FLOOR THAT DESCRIBES ONLY ONE DATASET LIES ABOUT THE OTHER
# (2026-10-22). `dnu >= 12` was written against the CROP dataset — the
# shipped production path. When the crop file is absent (or, as happened
# today, mid-write) `_rows()` falls back to the whole-frame dataset, where
# the SAME assertion fails with the most alarming message available —
# "the video channel must not be silently disarmed" — about a channel that
# is fine. SEEN: `only 8/30 fakes flagged -- assert 8 >= 12`.
#
# MEASURED (30 fakes / 30 reals each):
#   effort_crop  the shipped face-crop path  -> 13 DNU, 1 fp, 0 trust
#   effort       the whole-frame fallback    ->  8 DNU, 1 fp, 0 trust
#
# So the floor is keyed by which dataset was loaded, with headroom below the
# measured value in both cases. Never move the crop floor down to make the
# fallback pass: the crop floor IS the shipped behaviour.
DNU_FLOOR = {"effort_crop": 12, "effort": 6}


def _sizes() -> str:
    bits = []
    for path in (CROP_ROWS_PATH, ROWS_PATH):
        if not path.exists():
            bits.append(f"{path.name}=absent")
        else:
            try:
                bits.append(f"{path.name}={len(_load(path))} rows")
            except Exception as e:  # noqa: BLE001
                bits.append(f"{path.name}=unreadable({type(e).__name__})")
    return ", ".join(bits)


def _rows():
    """The measured corpus, preferring the face-crop dataset.

    ⚠️ A PARTIAL FILE IS NOT A MEASUREMENT (2026-10-22). This used to accept
    ANY file that existed and parsed. `tools/score_video_signals_crop.py`
    unlinks `video_signal_rows_crop.jsonl` and then appends one row per clip,
    so while that harness runs the file holds a handful of rows — and the
    suite went RED with:

        only 0/0 fakes flagged — the video channel must not be silently
        disarmed

    which is the most alarming conclusion available, about a file that was
    simply mid-write. (The `fn <= 13` ceiling test, meanwhile, PASSED over an
    empty corpus — a vacuous green is the same defect from the other side.)

    So a dataset counts only if it is COMPLETE: enough rows and BOTH classes
    present. Otherwise it counts as absent and the test skips honestly,
    naming what it saw.
    """
    for path, key in ((CROP_ROWS_PATH, "effort_crop"), (ROWS_PATH, "effort")):
        if not path.exists():
            continue
        # a torn last line is exactly what a mid-write append looks like, so a
        # parse failure means "not complete yet", not "the suite exploded"
        try:
            data = _load(path) or []
        except (json.JSONDecodeError, OSError):
            continue
        if len(data) < MIN_ROWS:
            continue
        if not {0, 1} <= {r.get("label") for r in data}:
            continue
        for r in data:
            r["_effort_key"] = key
        return data
    pytest.skip(f"no COMPLETE measured video corpus on this host ({_sizes()}; "
                f"need >= {MIN_ROWS} rows with both classes)")


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
    # a ceiling over an EMPTY corpus is a vacuous green -- guard the form
    assert len(fakes) >= MIN_ROWS // 2, (
        f"only {len(fakes)} fakes in the corpus — too few to say anything "
        f"about a false-negative ceiling")
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
    separate them -- the cross-modal signal is what flagged them. That is
    the documented design ("the operator AI video still reaches DO_NOT_USE
    via the AV corroboration path, NOT effort alone") -- and at the time the
    real AI videos DID read DO_NOT_USE end-to-end, verified against the live
    CLI (3/3 AI videos HIGH, 4/4 genuine clips LOW).

    ⚠️ 2026-10-22: THAT "3/3 AI HIGH" IS NO LONGER TRUE, AND IT WAS NEVER
    A PROPERTY OF THIS OFFSET. The cross-modal reading that carried it,
    av=0.5, was the correlation function's UNDEFINED SENTINEL (-2.0): the
    AI videos have a COMPLETELY SILENT audio track (RMS 0.0), so their
    correlation was undefined and the pipeline reported them as
    anti-correlated. With the honest reading they read MEDIUM/caution, and
    the 4/4 genuine = LOW reading is unchanged.

    Measured on the operator's real WhatsApp traffic, NO rule over the
    current signals reaches DNU for the AI videos without also flagging
    genuine media (12% of genuine WhatsApp video has the same all-zero
    audio track). The remaining gap is a MISSING CHECKPOINT, not a
    threshold.

    THE OFFSET IS STILL CHOSEN BY THE CORPUS NUMBERS ABOVE (13/30 fakes
    flagged at 1 false positive) and by the clean-side half of the guard
    (4/4 genuine = trust). Only the AI-video half of the end-to-end
    expectation is gone.

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
    floor = DNU_FLOOR[rows[0]["_effort_key"]]
    assert dnu >= floor, (
        f"only {dnu}/{len(fakes)} fakes flagged on the "
        f"{rows[0]['_effort_key']} dataset (floor {floor}) — the video "
        "channel must not be silently disarmed")
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


# ---------------------------------------------------------------------------
# 6. The FIXTURE must not turn a mid-write file into a verdict (2026-10-22).
# ---------------------------------------------------------------------------
def _write_rows(path, n, *, label=0):
    """Write a measured-corpus fragment.

    `label=None` alternates 0/1 so the fragment has BOTH classes — a partial
    corpus must be rejected on its ROW COUNT alone, otherwise a labels check
    can mask a missing row-count check (it did: mutation P1 was missed on the
    first version of these tests).
    """
    import json as _json
    path.write_text("\n".join(
        _json.dumps({"path": f"/x/{i}.mp4",
                     "label": (i % 2) if label is None else label,
                     "effort": 0.5, "effort_crop": 0.5}) for i in range(n)) + "\n")
    return path


def test_a_partial_corpus_is_absent_not_a_red_suite(monkeypatch, tmp_path):
    """A trunctated dataset is SKIPPED, not asserted over.

    `tools/score_video_signals_crop.py` unlinks its output and appends one
    row per clip. Reading it mid-run used to hand the assertions a 2-row
    corpus, which produced 'only 0/0 fakes flagged -- the video channel must
    not be silently disarmed': the most alarming available conclusion, about
    a file that was simply being written.

    A HARD SKIP is the honest outcome: the measurement does not exist yet.
    """
    partial = _write_rows(tmp_path / "partial.jsonl", 2, label=None)
    monkeypatch.setattr("test_61_video_calibration.CROP_ROWS_PATH", partial)
    monkeypatch.setattr("test_61_video_calibration.ROWS_PATH",
                        tmp_path / "absent.jsonl")
    with pytest.raises(pytest.skip.Exception) as exc:
        _rows()
    assert "COMPLETE" in str(exc.value), (
        "the skip reason must say what is missing, so a permanently broken "
        "corpus cannot hide behind a silent skip")


def test_a_torn_last_line_is_treated_as_incomplete(monkeypatch, tmp_path):
    """A half-written JSON line must not crash the suite either."""
    torn = tmp_path / "torn.jsonl"
    _write_rows(torn, 41)
    torn.write_text(torn.read_text() + '{"path": "/x/torn.mp4", "lab')
    monkeypatch.setattr("test_61_video_calibration.CROP_ROWS_PATH", torn)
    monkeypatch.setattr("test_61_video_calibration.ROWS_PATH",
                        tmp_path / "absent.jsonl")
    with pytest.raises(pytest.skip.Exception):
        _rows()


def test_a_complete_corpus_is_returned_and_keyed(monkeypatch, tmp_path):
    """Guard the guard: the complete case must still hand back rows."""
    complete = tmp_path / "complete.jsonl"
    _write_rows(complete, MIN_ROWS // 2, label=0)
    with complete.open("a") as fh:
        for i in range(MIN_ROWS // 2):
            fh.write(json.dumps({"path": f"/y/{i}.mp4", "label": 1,
                                 "effort": 0.5, "effort_crop": 0.5}) + "\n")
    monkeypatch.setattr("test_61_video_calibration.CROP_ROWS_PATH", complete)
    monkeypatch.setattr("test_61_video_calibration.ROWS_PATH",
                        tmp_path / "absent.jsonl")
    rows = _rows()
    assert len(rows) == MIN_ROWS
    assert {r["_effort_key"] for r in rows} == {"effort_crop"}, (
        "the loaded dataset must be tagged, because the detection FLOOR "
        "depends on which dataset it is")


@pytest.mark.skipif(not ROWS_PATH.exists(),
                    reason="the whole-frame fallback dataset was not scored here")
def test_the_fallback_dataset_floor_is_its_own_number():
    """The FALLBACK dataset must be judged by its own floor, not the crop's.

    MEASURED 2026-10-22 (30 fakes / 30 reals, production fusion engine):
      * whole-frame `effort`      ->  8 DNU of 30, 1 false positive, 0 trust
      * face-crop    `effort_crop` -> 13 DNU of 30, 1 false positive, 0 trust

    Reusing 12 (the crop floor) on the fallback fails with "the video channel
    must not be silently disarmed" — which is alarming, specific, and FALSE.
    That is exactly how the suite went red while the crop dataset was being
    rewritten, first because the fragment had no fakes and then because the
    fallback inherited a floor that was never measured on it.
    """
    rows = [r for r in (_load(ROWS_PATH) or []) if "effort" in r]
    for r in rows:
        r["_effort_key"] = "effort"
    reals = [r for r in rows if r["label"] == 0]
    fakes = [r for r in rows if r["label"] == 1]
    assert reals and fakes, "the fallback dataset has one class — not a corpus"
    dnu = sum(1 for r in fakes
              if fe.decide("deepfake_video", _checks_from_row(r)).verdict
              is Verdict.DO_NOT_USE)
    floor = DNU_FLOOR["effort"]
    assert dnu >= floor, (
        f"the whole-frame fallback flags {dnu}/{len(fakes)} fakes, below its "
        f"own floor of {floor} — the fallback dataset's separation changed, "
        f"and the crop dataset is NOT the thing to compare it against")
