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

The per-JOB envelope (VISHWAS_BUDGET_S, 300s default) remains the outer
limit, so a stage can never exceed the job budget; this test pins the INNER
per-stage allowance to a value that covers the measured worst case.
"""
from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "/home/hermes/vishwas/src")
from vishwas import orchestrator as O


def test_heavy_stage_budget_covers_the_measured_detector_cost():
    """The default per-stage budget must cover a cold load of every learned
    gate plus the transform battery."""
    budget = O._default_heavy_stage_budget()
    # Measured 2026-10-21: 7.0s (aasist3) + 4.7s (xlsr) + ~27s (video stage
    # with the transform battery) ~= 40s, with headroom for a slower disk.
    assert budget >= 60, (
        f"heavy-stage budget is {budget}s, below the measured worst case "
        f"(~40s); heavy stages will expire mid-run and the verdict is never "
        f"computed"
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
