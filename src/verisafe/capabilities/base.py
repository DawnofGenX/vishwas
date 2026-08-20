"""Capability contract + shared evidence types for all specialist modules."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..events import Artifact, JobContext, Verdict


Cost = Literal["cheap", "mid", "heavy"]
Status = Literal["ok", "degraded", "unavailable", "failed", "skipped"]


@dataclass(slots=True)
class CheckResult:
    """Structured evidence from ONE specialist check.

    signals must be machine-readable (str/int/float/bool/list) so the fusion
    layer can consume it without parsing prose. Notes are translation keys or
    short English facts; they never become the verdict by themselves.
    """
    name: str
    cost: Cost
    status: Status
    signals: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    duration_s: float = 0.0

    def usable(self) -> bool:
        return self.status in ("ok", "degraded")

    @property
    def prob(self) -> float | None:
        """Primary probability signal if this check produced one (0..1)."""
        for k in ("prob_deepfake", "prob_malicious", "prob_phishing", "prob_forged"):
            v = self.signals.get(k)
            if isinstance(v, (int, float)):
                return float(min(1.0, max(0.0, v)))
        return None


class Capability(Protocol):
    """A domain expert module. analyze() returns evidence only — no verdicts."""
    requires: tuple[str, ...]      # dependency ids used by the availability gate

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        ...


def make_unavailable(check_id: str, dep: str, reason: str = "") -> CheckResult:
    return CheckResult(name=f"{check_id}", cost="cheap", status="unavailable",
                       signals={"missing_dependency": dep}, notes=reason or f"dependency '{dep}' unavailable; check skipped by design")


def make_check(name: str, cost: Cost, ok: bool, signals: dict[str, Any], notes: str = "",
               t0: float | None = None, degraded_reason: str = "") -> CheckResult:
    dur = round(time.monotonic() - (t0 if t0 is not None else time.monotonic()), 3)
    status: Status = "ok" if ok else ("degraded" if signals.get("degraded_reason") else "failed")
    if degraded_reason:
        status = "degraded"
    return CheckResult(name=name, cost=cost, status=status, signals=signals,
                       notes=notes, duration_s=dur)
