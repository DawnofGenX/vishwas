"""test_59_job_budget_cap.py — F6: per-stage subprocess timeouts must respect
the job's remaining wall budget (cap at remaining, with a sane floor), so a
stage can never overrun the job's hard_budget_s envelope.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vishwas.events import Artifact, InputType, JobContext, new_job_id


def _mk_ctx(remaining_s: float) -> JobContext:
    art = Artifact(path=Path("/tmp/nonexistent.bin"), original_filename="x.bin",
                   declared_type=InputType.FILE)
    return JobContext(job_id=new_job_id(), artifact=art, quarantine_root=Path("/tmp"),
                      deadline_mono=time.monotonic() + remaining_s,
                      wall_budget_s=max(remaining_s, 1.0))


def test_f6_large_timeout_capped_by_remaining_budget():
    ctx = _mk_ctx(50.0)                # ~50s left in the job envelope
    t = ctx.cap_timeout(420.0)         # mobsf's static 420s request
    assert t <= 50.0 + 0.5, f"stage timeout {t}s exceeds remaining budget"
    assert t >= 5.0, "must keep a sane floor"


def test_f6_requested_below_budget_is_respected():
    ctx = _mk_ctx(50.0)
    t = ctx.cap_timeout(20.0)
    assert abs(t - 20.0) < 0.5, f"a 20s request within budget should stay ~20s, got {t}"


def test_f6_floor_lifts_degenerate_timeout_when_budget_generous():
    ctx = _mk_ctx(300.0)
    t = ctx.cap_timeout(2.0)           # requested below the floor
    assert t >= 5.0
    assert t <= 300.0


def test_f6_floor_never_exceeds_remaining():
    ctx = _mk_ctx(3.0)                 # very little budget left
    t = ctx.cap_timeout(420.0)
    assert t <= 3.0 + 0.5, "floor must not push the timeout above remaining budget"


def test_f6_exhausted_budget_yields_no_time():
    ctx = _mk_ctx(-1.0)
    assert ctx.cap_timeout(420.0) == 0.0


def test_f6_malware_file_caps_its_subprocess_timeouts():
    src = (Path(__file__).resolve().parents[1] / "src" / "vishwas" /
           "capabilities" / "malware_file.py").read_text(encoding="utf-8")
    for req in (420, 480, 240, 90):
        assert f"cap_timeout({req})" in src, (
            f"malware_file.py must cap its {req}s subprocess timeout via ctx.cap_timeout({req})")
    for bad in ("timeout=420", "timeout=480", "timeout=240"):
        assert bad not in src, f"static over-budget timeout still present: {bad}"
