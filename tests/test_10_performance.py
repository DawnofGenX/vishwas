"""P8 performance tests: wall-clock budget floors, conservative short-circuit,
per-stage timing records, and deterministic behavior under a *tiny* budget.

Everything runs on fake capabilities — no models, no network — so these tests
must pass on any CPU within seconds and prove the budget machinery is correct,
not just present.
"""
from __future__ import annotations

import sys
import time
from dataclasses import field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verisafe.events import Artifact, InputType, JobContext, Verdict
from verisafe.fusion import FusionEngine, ReliabilityGate
from verisafe.report import ReportBuilder
from verisafe.orchestrator import Orchestrator, _has_confirmed_danger
from verisafe.capabilities.base import CheckResult


# ------------------------------------------------------------ fakes --------
class CapBase:  # satisfies the Capability protocol structurally
    requires: tuple[str, ...] = ()

    def __init__(self):
        self.calls = []

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        self.calls.append(time.monotonic())
        return self._checks()

    def _checks(self) -> list[CheckResult]:
        raise NotImplementedError


class CleanCap(CapBase):
    def _checks(self):
        return [CheckResult("clean_probe", "cheap", "ok", {}, "nothing suspicious")]


class DangerAVCap(CapBase):
    """Simulates clamscan returning a signature hit."""
    def _checks(self):
        return [CheckResult("clamscan", "cheap", "ok", {"detected": True, "sig": "Eicar-Test-Signature"}, "detection: eicar test string")]


class YaraPlusQuarkCap(CapBase):
    """Two independent static-analysis families agree -> confirmed malware."""
    def _checks(self):
        return [
            CheckResult("yara_x", "mid", "ok", {"hits_norm": 0.8}),
            CheckResult("quark_engine", "heavy", "ok", {"score_norm": 0.75}),
        ]


class WeakHeuristicCap(CapBase):
    """A single medium-confidence heuristic alone must NOT trigger short-circuit."""
    def _checks(self):
        return [CheckResult("yara_x", "mid", "ok", {"hits_norm": 0.6})]


class SlowCap(CapBase):
    """Burns ~3s of wall time; used to observe the 10s floor decision."""
    def _checks(self):
        time.sleep(0.2)  # scaled-down stand-in for a real heavy stage
        return [CheckResult("slow_stage", "heavy", "ok", {})]


def _make_orch(caps: list, hard_budget_s: float):
    o = Orchestrator(capabilities_by_target={"unclassified": caps},
                     fusion=FusionEngine(), reliability=ReliabilityGate(),
                     reporter=ReportBuilder(), hard_budget_s=hard_budget_s,
                     available_deps=set())
    return o


@pytest.fixture()
def clean_artifact(qroot):
    p = qroot / "probe.bin"
    p.write_bytes(b"%PDF-1.7\n%%EOF")
    return make_artifact(qroot, "doc.pdf", InputType.FILE, data=b"%PDF-1.7\n%%EOF")


def make_artifact(job_dir: Path, filename: str, declared: InputType, data: bytes | None = None):
    from verisafe.file_validator import make_artifact as _m
    return _m(job_dir, filename, declared, data=data)


# ------------------------------------------- danger predicate unit checks ----
def test_predicate_clamscan_alone_confirms():
    assert _has_confirmed_danger([CheckResult("clamscan", "cheap", "ok", {"detected": True})])


def test_predicate_yara_needs_corroboration():
    assert not _has_confirmed_danger([CheckResult("yara_x", "mid", "ok", {"hits_norm": 0.9})])
    assert _has_confirmed_danger([
        CheckResult("yara_x", "mid", "ok", {"hits_norm": 0.9}),
        CheckResult("quark_engine", "heavy", "ok", {"score_norm": 0.6}),
    ])


def test_predicate_phish_needs_network_evidence():
    assert not _has_confirmed_danger([CheckResult("phish_heuristics", "mid", "ok", {"score_norm": 0.9})])
    assert _has_confirmed_danger([
        CheckResult("phish_heuristics", "mid", "ok", {"score_norm": 0.9}),
        CheckResult("ssrf_guard", "cheap", "degraded", {"degraded": True}),
    ])


def test_predicate_unusable_results_ignored():
    # a *failed* clamscan check must not count as a detection
    assert not _has_confirmed_danger([CheckResult("clamscan", "cheap", "failed", {})])


# --------------------------------------------------- end-to-end behaviors --
def test_short_circuit_skips_later_heavy_stages(clean_artifact, qroot):
    """Once DangerAV fires, later non-light stages record skip_early_stop."""
    av, slow1, slow2 = DangerAVCap(), SlowCap(), SlowCap()
    orch = _make_orch([av, slow1, slow2], hard_budget_s=300)
    out = orch.handle_incoming({"id": "j-sc", "text": "check this file",
                                "_qroot_override": str(qroot)})
    names = {c.name for c in out.checks}
    assert "clamscan" in names
    assert "skip_early_stop" in "".join(names), f"no early-stop marker in {names}"
    # both later stages' *classes* never ran (records dedupe by name)
    assert slow1.calls == [] and slow2.calls == [], "short-circuited stages must not execute"
    assert out.fusion_trace["short_circuited_at"] == "DangerAVCap"


def test_weak_heuristic_does_not_short_circuit(clean_artifact, qroot):
    """Single uncorroborated heuristic -> everything still runs."""
    weak, after = WeakHeuristicCap(), CleanCap()
    orch = _make_orch([weak, after], hard_budget_s=300)
    out = orch.handle_incoming({"id": "j-wk", "text": "check",
                                "_qroot_override": str(qroot)})
    assert after.calls != [], "stage after weak heuristic was wrongly skipped"
    assert out.fusion_trace.get("short_circuited_at") is None


def test_budget_floor_refuses_to_start_near_deadline(clean_artifact, qroot):
    """With only 5s left at stage start, the stage records a timeout, doesn't burn CPU."""
    fast_burner = type("FastBurner", (CapBase,), {
        "analyze": lambda self, art, ctx: (time.sleep(2.5),
                                           [CheckResult("burned", "heavy", "ok", {})])[1]
    })()
    victim = SlowCap()
    orch = _make_orch([fast_burner, victim], hard_budget_s=8)  # tiny budget
    out = orch.handle_incoming({"id": "j-bf", "text": "budget floor",
                                "_qroot_override": str(qroot)})
    victim_names = [c for c in out.checks if c.name.startswith("SlowCap")]
    assert any(c.status == "skipped" and "10s" in c.notes for c in victim_names), \
        f"victim stage should have been refused by the floor: {[ (c.name,c.status,c.notes) for c in victim_names]}"
    assert victim.calls == [], "floor-refused stage must not run at all"
    assert out.wall_s < 40, f"job blew past its tiny budget: {out.wall_s}s"


def test_per_stage_timings_recorded_for_all_runs(clean_artifact, qroot):
    a, b = CleanCap(), CleanCap()
    orch = _make_orch([a, b], hard_budget_s=300)
    out = orch.handle_incoming({"id": "j-tg", "text": "timing",
                                "_qroot_override": str(qroot)})
    t = out.fusion_trace["stage_timings_s"]
    assert set(t) >= {"CleanCap"}, f"timings missing: {t}"
    assert all(v >= 0 for v in t.values())


def test_outcome_serializes_with_new_fields(clean_artifact, qroot):
    orch = _make_orch([CleanCap()], hard_budget_s=300)
    out = orch.handle_incoming({"id": "j-js", "text": "json",
                                "_qroot_override": str(qroot)})
    d = out.to_dict()
    assert isinstance(d["wall_s"], float)
    assert "stage_timings_s" in out.fusion_trace
