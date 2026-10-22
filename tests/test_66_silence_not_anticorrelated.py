"""SILENCE must not read as "anti-correlated deepfake evidence".

THE BUG (found 2026-10-22, live measurement on real WhatsApp traffic)
--------------------------------------------------------------------
`_lagged_crosscorr` returns **-2.0** as its sentinel for "no correlation was
computed" (it initialises `best_r = -2.0` and only replaces it when a lag
produces a higher r). The caller then classifies:

    verdict_class = (... else "anti_correlated" if r < -0.25 ...)
    risk_add = {..., "anti_correlated": 0.5, ...}

So the sentinel -2.0 falls straight into the ANTI-CORRELATED branch and
becomes `av_risk_addition = 0.5` — the HIGHEST risk reading the signal can
produce, and the one the fusion corroboration gate explicitly TRUSTS (the
decorrelated band 0.45 is discounted, the anti-correlated band 0.5 is not).

The correlation is undefined when either envelope is CONSTANT, which happens
when the audio is silent or contains no speech energy. Measured on 67 real
WhatsApp videos from the operator's own traffic: **11 genuine clips landed
in the anti-correlated band**, and **9 of those 11 had `speech std = 0.0`**
(a constant speech envelope) and `corr = -2.0` — the sentinel, not a
correlation.

So a genuine video with a silent or speechless audio track was being given
the strongest possible "these modalities were generated independently"
signal. That is a FAIL-OPEN path in the clean direction: it can push
genuine media toward DO_NOT_USE.

THE FIX
-------
`_lagged_crosscorr` must return a distinct "no correlation" value that the
classifier treats as UNKNOWN, and the caller must map an unknown
correlation to a neutral reading rather than to the highest-risk band. A
clip whose audio carries no speech energy is not evidence of anything: the
honest answer is "the cross-modal check could not measure anything".

These tests pin that contract at both seams.
"""
from __future__ import annotations

import math

import pytest

from vishwas.capabilities import cross_modal as cm


def test_lagged_crosscorr_does_not_return_the_anti_correlated_sentinel():
    """A constant (silent) envelope must not produce the -2.0 that the
    classifier reads as 'anti-correlated'."""
    speech = [0.0] * 30          # silence: a constant envelope
    mouth = [float(i % 7) for i in range(30)]
    r, lag = cm._lagged_crosscorr(mouth, speech)
    assert not (r <= -1.5), (
        f"a constant envelope produced r={r}, which the classifier maps to "
        "'anti_correlated' -> av_risk_addition 0.5 (the highest-risk band)")


def test_lagged_crosscorr_marks_an_undefined_correlation_distinctly():
    """The 'no correlation' case must be DISTINGUISHABLE from a real negative
    correlation, so the classifier can treat it as unknown rather than as
    evidence."""
    speech = [0.0] * 30
    mouth = [float(i % 7) for i in range(30)]
    r_undefined, _ = cm._lagged_crosscorr(mouth, speech)
    assert not math.isfinite(r_undefined), (
        f"the undefined-correlation case must be non-finite, got {r_undefined}")
    # a real negative correlation: two square waves with coprime-ish periods
    # so no lag aligns them
    a = [1.0 if (i // 3) % 2 == 0 else -1.0 for i in range(30)]
    b = [1.0 if (i // 5) % 2 == 0 else -1.0 for i in range(30)]
    r_real, _ = cm._lagged_crosscorr(a, b)
    assert math.isfinite(r_real), (
        "a genuine correlation must be finite")
    assert r_undefined != r_real, (
        "the undefined-correlation case must be distinguishable from a real "
        f"correlation (both gave {r_undefined})")


def test_a_silent_audio_track_does_not_produce_the_highest_risk_band():
    """The end-to-end contract: a clip with no speech energy must not get
    av_risk_addition = 0.5."""
    silence = [0.0] * 30
    mouth = [float(i % 7) for i in range(30)]
    r, lag = cm._lagged_crosscorr(mouth, silence)
    # The classifier's own logic, replicated so the test pins the CONTRACT
    # rather than the implementation.
    if not math.isfinite(r):
        risk = cm.UNKNOWN_AV_RISK
    else:
        verdict_class = ("synced" if r >= 0.35 and lag <= 100
                         else "decorrelated" if abs(r) < 0.15
                         else "anti_correlated" if r < -0.25
                         else "weakly_synced")
        risk = {"synced": 0.0, "weakly_synced": 0.1, "decorrelated": 0.45,
                "anti_correlated": 0.5}.get(verdict_class, 0.1)
    assert risk != 0.5, (
        "a clip with no speech energy was given the highest-risk "
        f"cross-modal reading ({risk})")
    assert risk == cm.UNKNOWN_AV_RISK, (
        f"an unmeasurable cross-modal reading should be UNKNOWN, got {risk}")


def test_a_real_negative_correlation_still_gets_the_high_risk_band():
    """The fix must not disarm the signal it was meant to protect.

    NOTE on the fixture: `_lagged_crosscorr` searches for the MAXIMUM r over
    all lags, so a perfectly ANTI-PHASE pair (sin vs -sin, or a square wave
    against its inverse) is invisible to it — a half-period shift makes the
    pair align and the maximum comes out POSITIVE (measured: r=+0.99). The
    real anti-correlated readings the pipeline produces are modest negatives
    (the operator's AI videos measured -0.299 and -0.443), which come from
    noisy envelopes rather than a clean inversion.

    The fixture below is a MONOTONE descent against a monotone ascent: the
    correlation is negative at EVERY lag, so the maximum-over-lags search
    also returns a negative value (measured r=-1.0). That is the shape the
    classifier's `r < -0.25` branch is for.

    This test pins that a real negative correlation is still classified as
    anti-correlated, i.e. the fix did not blunt the signal."""
    n = 30
    descending = [float(i) for i in range(n)]
    ascending = [float(n - i) for i in range(n)]
    r, lag = cm._lagged_crosscorr(descending, ascending)
    assert math.isfinite(r), "a real correlation must be finite"
    assert r < -0.25, (
        f"a genuinely anti-correlated pair read r={r}; the fix must not "
        "weaken the real signal")
    # Classify exactly as the capability does.
    verdict_class = ("synced" if r >= 0.35 and lag <= 100
                     else "decorrelated" if abs(r) < 0.15
                     else "anti_correlated" if r < -0.25
                     else "weakly_synced")
    assert verdict_class == "anti_correlated", (
        f"a real negative correlation classified as {verdict_class!r} "
        f"(r={r}); the fix must not blunt the genuine signal")
    risk = {"synced": 0.0, "weakly_synced": 0.1, "decorrelated": 0.45,
            "anti_correlated": 0.5}[verdict_class]
    assert risk == 0.5, (
        f"a genuine anti-correlation must still carry the 0.5 risk reading, "
        f"got {risk}")
