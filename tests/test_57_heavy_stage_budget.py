"""The heavy-stage budget must be large enough for the learned detectors.

REGRESSION (2026-10-21)
-----------------------
`orchestrator._default_heavy_stage_budget()` returned 30s. That value was
sized for a host where the learned detectors were NOT installed, so no heavy
stage ever ran and the budget was never reached.

Once the detectors actually load, 30s is too tight. Measured on this host:
    cold aasist3 load      7.0s
    cold xlsr-mamba load   4.7s
    deepfake_video stage  ~27s  (frame extraction + EFFORT over 8 frames
                                + the 5-variant transform battery)

At 30s BOTH the audio and the video channels expired mid-stage and
returned `unable_to_verify` with `pending_heavy` set — i.e. the verdict was
never computed, and the platform reported "I could not verify this" for
every real audio and video it was given. That is the opposite of the
intent: the detectors were working, the budget just expired.

REGRESSION AGAIN (2026-10-22) — THE SAME DEFECT, ONE DAY LATER
-------------------------------------------------------------
The budget was raised to 120s on 2026-10-21, from THAT day's
measurements. Those numbers no longer held, and the defect returned in
exactly the same shape:

    aasist3 checkpoint LOAD (cold)   63.1 s   <- was measured at 7.0 s
    one aasist3 inference             17.6 s
    the audio lane (3 crops)         ~241 s
    the video lane, cold cache        147 s

**⇒ AT 120s THE AUDIO AND VIDEO LANES RETURNED `UNVERIFIED` WITH
`pending_heavy` ON REAL MEDIA** — a correct verdict when the page cache
was warm and NO verdict when it was cold. The models were fine (aasist3
and xlsr both AUC 1.0000); the PIPELINE TIMED OUT.

Two things changed the picture:
  1. the aasist3 lane now loads the checkpoint ONCE for all its crops
     (PR #28), so the audio lane is 1 load + N inferences;
  2. the default budget is now 240s, and the stale "7.0 s load" claim in
     the docstring is replaced with the measured numbers.

**⇒ A BUDGET MEASURED ON A DIFFERENT DAY IS A GUESS.** The floor below is
now derived from the CURRENT measurement, and it is deliberately
conservative: the budget must cover the COLD cost, because the cold case
is exactly the one that produced a wrong verdict.

The per-JOB envelope (VISHWAS_BUDGET_S, 300s default) remains the outer
limit, so a stage can never exceed the job budget; this test pins the INNER
per-stage allowance to a value that covers the measured worst case.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "/home/hermes/vishwas/src")
from vishwas import orchestrator as O


# The measured cold cost, in seconds (2026-10-22).
#   aasist3 checkpoint load (cold)   63.1
#   one aasist3 inference            17.6
#   the video lane, cold cache      147
# The audio lane is now 1 load + N inferences (the crops share the load).
MEASURED_COLD_VIDEO_LANE_S = 147.0
MEASURED_COLD_AUDIO_LANE_S = 63.1 + 3 * 17.6


def test_heavy_stage_budget_covers_the_measured_detector_cost():
    """The default per-stage budget must cover the COLD cost of the
    slowest measured lane — not the warm one.

    The 2026-10-21 version of this test floored the budget at 60s, from a
    "~40s" estimate that had itself gone stale. A floor set from a
    WARM measurement is the same defect in a different guise: the cold
    case is precisely the one that produced a wrong verdict.
    """
    budget = O._default_heavy_stage_budget()
    worst = max(MEASURED_COLD_VIDEO_LANE_S, MEASURED_COLD_AUDIO_LANE_S)
    assert budget >= worst, (
        f"heavy-stage budget is {budget}s, below the measured COLD worst "
        f"case ({worst:.0f}s); heavy stages will expire mid-run and the "
        f"verdict is never computed"
    )


def test_the_budget_is_not_sized_for_a_stale_measurement():
    """The floor that shipped with the 120s default (60s) was derived from
    a "7.0s cold aasist3 load" measurement that is now known to be 63.1s.

    So a budget of 60s would PASS the old test and STILL break the
    product. This test pins the floor against the CURRENT measurement.
    """
    budget = O._default_heavy_stage_budget()
    assert budget >= 200, (
        f"heavy-stage budget is {budget}s — the audio lane's COLD cost is "
        f"{MEASURED_COLD_AUDIO_LANE_S:.0f}s, so a smaller budget returns "
        f"UNVERIFIED on a cold cache"
    )


def test_heavy_stage_budget_is_below_the_job_envelope():
    """A stage must never be able to exceed the per-job budget."""
    import os
    job_budget = float(os.environ.get("VISHWAS_BUDGET_S", "300"))
    assert O._default_heavy_stage_budget() < job_budget, (
        "the per-stage budget must stay under the per-job envelope"
    )


def test_heavy_stage_budget_is_overridable():
    """The env var must still win, so an operator can tune it."""
    import os
    old = os.environ.get("VISHWAS_HEAVY_STAGE_BUDGET_S")
    os.environ["VISHWAS_HEAVY_STAGE_BUDGET_S"] = "45"
    try:
        assert O._default_heavy_stage_budget() == 45
    finally:
        if old is None:
            os.environ.pop("VISHWAS_HEAVY_STAGE_BUDGET_S", None)
        else:
            os.environ["VISHWAS_HEAVY_STAGE_BUDGET_S"] = old


def test_the_budget_docstring_states_the_current_measurement():
    """The docstring is where the NEXT reader gets their number.

    It MAY mention the superseded "7.0s" figure — but only to mark it as
    superseded. What it must NOT do is present a stale measurement as
    current, and it must state the real one.
    """
    import inspect
    doc = inspect.getdoc(O._default_heavy_stage_budget) or ""
    assert "63.1" in doc, (
        "the heavy-stage budget docstring does not state the MEASURED cold "
        "load (63.1s), so a reader cannot tell whether the number is "
        "current")
    # if the old figure appears, it must be inside a sentence that marks
    # it as superseded -- not presented as a live measurement.
    if "7.0s" in doc:
        # The marker that retires a figure may sit on a NEARBY line, so
        # the check is over a WINDOW of context rather than the one line.
        lines = doc.split("\n")
        for i, line in enumerate(lines):
            if "7.0s" not in line:
                continue
            window = "\n".join(lines[max(0, i - 4):i + 5])
            assert ("no longer hold" in window
                    or "superseded" in window
                    or "was measured at" in window
                    or "Those numbers" in window), (
                f"the docstring presents the superseded 7.0s load as "
                f"current, with no marker nearby:\n{window}")
