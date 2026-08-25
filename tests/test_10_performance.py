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

from vishwas.events import Artifact, InputType, JobContext, Verdict
from vishwas.fusion import FusionEngine, ReliabilityGate
from vishwas.i18n import t
from vishwas.report import ReportBuilder
from vishwas.orchestrator import Orchestrator, _has_confirmed_danger
from vishwas.capabilities.base import CheckResult


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
    from vishwas.file_validator import make_artifact as _m
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


# --------------------------------------- non-blocking heavy follow-ups (2.1) --
class SlowLearnedCap(CapBase):
    """Stand-in for a T2 learned stage (aasist/effort/havic): slow, heavy."""
    stage_cost = "heavy"

    def __init__(self, delay: float = 1.5, prob: float = 0.81):
        super().__init__()
        self.delay = delay
        self.prob = prob

    def _checks(self):
        time.sleep(self.delay)
        return [CheckResult("learned_probe", "heavy", "ok",
                            {"prob_deepfake": self.prob})]


def _make_orch2(caps: list, heavy_stage_budget_s: float):
    return Orchestrator(capabilities_by_target={"unclassified": caps},
                        fusion=FusionEngine(), reliability=ReliabilityGate(),
                        reporter=ReportBuilder(), hard_budget_s=300,
                        heavy_stage_budget_s=heavy_stage_budget_s,
                        available_deps=set())


def _probe_batch(prob: float) -> list[CheckResult]:
    return [CheckResult("learned_probe", "heavy", "ok", {"prob_deepfake": prob})]


def _expected_followup(orch: Orchestrator, batch: list[CheckResult],
                       cap_name: str, lang: str) -> str:
    """Deterministic expectation: same fusion + template the impl must use."""
    fused = orch.fusion.decide("unclassified", batch)
    return t("heavy_followup", lang, cap=cap_name,
             verdict=t(ReportBuilder.VERDICT_KEY[fused.verdict], lang),
             conf=f"{int(round(fused.confidence * 100))}%")


def test_heavy_over_budget_ships_fast_with_pending_evidence(clean_artifact, qroot):
    """Stage over its budget -> verdict ships NOW with pending_heavy evidence
    plus the plain-language notice; the stage still finishes in background."""
    clean, slow = CleanCap(), SlowLearnedCap(delay=1.6)
    orch = _make_orch2([clean, slow], heavy_stage_budget_s=1.0)
    out = orch.handle_incoming({"id": "j-pend", "text": "check this file",
                                "_qroot_override": str(qroot)})
    assert out.fusion_trace.get("pending_heavy") == \
        [{"cap": "SlowLearnedCap", "expected_s": 1}]
    assert t("heavy_pending_notice", "en") in out.user_message
    assert out.wall_s < 1.6, f"verdict waited for the slow stage ({out.wall_s}s)"
    assert orch.wait_for_pending_followups(timeout_s=15)
    assert slow.calls, "background stage must still run to completion"


def test_followup_fires_within_cap_margin_en(tmp_path):
    slow = SlowLearnedCap(delay=1.5)
    orch = _make_orch2([slow], heavy_stage_budget_s=0.4)
    from vishwas.channels import MessageProcessor
    proc = MessageProcessor(orch, openwa=None, persist_outcomes=False,
                            workdir=tmp_path)
    t0 = time.monotonic()
    res = proc.process({"id": "sess-en", "text": "check this", "sender_lang": "en"})
    assert res["outcome"]["fusion_trace"]["pending_heavy"]
    assert orch.wait_for_pending_followups(timeout_s=15)
    assert len(proc.followups) == 1, "exactly one follow-up expected"
    fu = proc.followups[0]
    assert fu["jid"] == "sess-en"
    assert fu["ts_mono"] - t0 <= slow.delay + 2.0, \
        f"follow-up too late: {fu['ts_mono'] - t0:.2f}s"
    assert fu["reply"] == _expected_followup(orch, _probe_batch(slow.prob),
                                             "SlowLearnedCap", "en")


def test_followup_template_exact_hi(tmp_path):
    slow = SlowLearnedCap(delay=1.2)
    orch = _make_orch2([slow], heavy_stage_budget_s=0.3)
    from vishwas.channels import MessageProcessor
    proc = MessageProcessor(orch, openwa=None, persist_outcomes=False,
                            workdir=tmp_path)
    proc.process({"id": "sess-hi", "text": "यह जाँचो", "sender_lang": "hi"})
    assert orch.wait_for_pending_followups(timeout_s=15)
    assert len(proc.followups) == 1
    reply = proc.followups[0]["reply"]
    assert reply == _expected_followup(orch, _probe_batch(slow.prob),
                                       "SlowLearnedCap", "hi")
    assert "गहरा जाँच" in reply, "hi template must be used, not the en fallback"


def test_session_ended_drops_followup_silently(tmp_path, capsys):
    slow = SlowLearnedCap(delay=1.2)
    orch = _make_orch2([slow], heavy_stage_budget_s=0.3)
    from vishwas.channels import MessageProcessor
    proc = MessageProcessor(orch, openwa=None, persist_outcomes=False,
                            workdir=tmp_path)
    proc.process({"id": "sess-dead", "text": "check this", "sender_lang": "en"})
    proc.end_session("sess-dead")          # user session ends mid-check
    assert orch.wait_for_pending_followups(timeout_s=15)
    assert proc.followups == [], "no follow-up may be delivered to a dead session"
    out = capsys.readouterr().out
    assert "sess-dead" in out and "dropped" in out, \
        "drop must be logged to stdout"
