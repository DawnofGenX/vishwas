"""ReportBuilder: turns verdict+confidence+reasons into plain-language messages.

For non-technical elders the output is exactly two sentences max plus an
optional short tip. No jargon. Confidence is shown as simple words
(low/moderate/high) unless debug flag is set, then percent.
"""
from __future__ import annotations

from typing import Any

from .capabilities.base import CheckResult
from .events import Verdict
from .i18n import t, detect_language

_DEBUG_FLAG_PREFIX = "DEBUG:"


def _band(conf: float) -> str:
    if conf < 0.35:
        return "low"
    if conf < 0.70:
        return "moderate"
    return "high"


class ReportBuilder:
    """Assemble user-facing reply strings. All advice is template-first;
    LLM narrative is layered in ONLY when requested and gated."""

    VERDICT_KEY = {
        Verdict.TRUST: "verdict_trust",
        Verdict.CAUTION: "verdict_caution",
        Verdict.DO_NOT_USE: "verdict_do_not_use",
        Verdict.UNABLE_TO_VERIFY: "verdict_unable",
    }

    def build(self, *, target: str, verdict: Verdict, confidence: float,
              reasons: list[str], checks: list[CheckResult], lang: str = "en",
              artifact_name: str = "", llm_advice: str | None = None,
              use_debug: bool = False) -> "UserReport":
        key = self.VERDICT_KEY[verdict]
        # spec: EVERY reply states result + confidence band + practical advice
        conf_disp = f"{int(round(confidence * 100))}%" if use_debug else _band(confidence)
        parts = [t(key, lang), t("confidence_line", lang, conf=conf_disp)]

        # targeted practical tips by domain
        tip = self._tip_for(target, verdict, lang)
        if tip:
            parts.append(tip)
        skipped = [c.name for c in checks if c.status == "unavailable"]
        if skipped and verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE, Verdict.UNABLE_TO_VERIFY):
            parts.append(t("evidence_missing", lang))
        if llm_advice and verdict not in (Verdict.UNABLE_TO_VERIFY,):
            # keep templates first (trust anchor); LLM adds context after
            parts.append(llm_advice.strip())
        return UserReport("\n\n".join(p for p in parts if p), key=key, lang=lang)

    def _tip_for(self, target: str, verdict: Verdict, lang: str) -> str | None:
        if target in ("url_phishing",) :
            if verdict in (Verdict.DO_NOT_USE, Verdict.CAUTION):
                return t("advice_avoid_links", lang)
        if verdict is Verdict.DO_NOT_USE:
            if "malicious" in target:
                return t("advice_avoid_links", lang)
        return None


class UserReport:
    def __init__(self, text: str, key: str = "", lang: str = "en"):
        self.text = text
        self.key = key
        self.lang = lang

    def __str__(self) -> str:
        return self.text
