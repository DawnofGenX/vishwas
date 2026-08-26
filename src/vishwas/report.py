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


# UX 2026-08-26: richer narrative reply. Verdict -> (label, risk-adjective, emoji),
# presentation-only (derived purely from verdict, like _RISK_OF_VERDICT).
_VERDICT_TILE = {
    Verdict.TRUST: ("LIKELY GENUINE", "Low", "🟢"),
    Verdict.CAUTION: ("SUSPICIOUS", "Medium", "🟠"),
    Verdict.DO_NOT_USE: ("LIKELY FAKE", "High", "🔴"),
    Verdict.UNABLE_TO_VERIFY: ("UNVERIFIED", "Unknown", "⚪"),
}

# Check-name -> (concern_key, signal_key, threshold). Order = priority.
# signal missing or below threshold -> not a concern.
_CONCERN_RULES: list[tuple[str, str, float, str]] = [
    ("aasist_detector", "prob_deepfake", 0.5, "concern_audio_ai"),
    ("xlsr_audio_detector", "prob_deepfake", 0.5, "concern_audio_ai"),
    ("effort_face_forensics", "prob_deepfake", 0.5, "concern_video_face"),
    ("frame_heuristics", "prob_deepfake", 0.5, "concern_video_face"),
    ("cross_modal_av", "prob_inconsistent", 0.5, "concern_av_sync"),
    # IMAGE: the offline frequency-band heuristic (and SPAI heavy when wired)
    # carry prob_deepfake for still images; surface the same face/synthetic concern.
    ("frequency_band_analysis", "prob_deepfake", 0.5, "concern_video_face"),
    ("image_face_forensics", "prob_deepfake", 0.5, "concern_video_face"),
    ("vt_url_reputation", "positives_ratio", 0.05, "concern_url_flag"),
    ("phish_heuristics", "host_string_score", 0.5, "concern_url_typo"),
    ("gov_document", "prob_forged", 0.5, "concern_doc_forged"),
]

_MAX_CONCERNS = 3


def concerns_for(checks: list[CheckResult], target: str, verdict: Verdict,
                 lang: str = "en", _keys: bool = True) -> list[str]:
    """Return the localized ⚠️ concern bullets that fired for these checks.

    Concerns come from the individual CheckResult signals (never the machine-token
    `reasons` list). Only CAUTION/DO_NOT_USE carry concerns; TRUST/UNVERIFIED return
    []. DO_NOT_USE always appends 'concern_unverified_source' (claim unverifiable).
    Set _keys=False to get the concern keys (handy for tests/logging).
    """
    if verdict not in (Verdict.CAUTION, Verdict.DO_NOT_USE):
        return []
    by_name: dict[str, list[CheckResult]] = {}
    for c in checks:
        by_name.setdefault(c.name, []).append(c)

    fired: list[str] = []
    for name, sig, thr, key in _CONCERN_RULES:
        if len(fired) >= _MAX_CONCERNS:
            break
        for c in by_name.get(name, []):
            if not c.usable():
                continue
            val = 0.0
            if isinstance(c.signals, dict):
                val = float(c.signals.get(sig, 0.0) or 0.0)
            if val >= thr:
                fired.append(key)
                break
    if verdict is Verdict.DO_NOT_USE and "concern_unverified_source" not in fired:
        fired.append("concern_unverified_source")  # cap applies to fired-detect, keep unverified
    fired = fired[: _MAX_CONCERNS]
    return fired if _keys else [t(k, lang) for k in fired]


def _recommend_line(verdict: Verdict, lang: str = "en") -> str:
    """User action line keyed by verdict (presentation-only)."""
    if verdict is Verdict.DO_NOT_USE:
        return t("recommend_dont_forward", lang)
    if verdict is Verdict.CAUTION:
        return t("recommend_verify_source", lang)
    if verdict is Verdict.TRUST:
        return t("recommend_nothing", lang)
    return t("evidence_missing", lang)


class ReportBuilder:
    """Assemble user-facing reply strings. All advice is template-first;
    LLM narrative is layered in ONLY when requested and gated."""

    VERDICT_KEY = {
        Verdict.TRUST: "verdict_trust",
        Verdict.CAUTION: "verdict_caution",
        Verdict.DO_NOT_USE: "verdict_do_not_use",
        Verdict.UNABLE_TO_VERIFY: "verdict_unable",
    }

    # UX 2026-08-26: deterministic risk level leading every reply. Derived
    # ONLY from the verdict — no new scoring, presentation layer only.
    _RISK_OF_VERDICT = {
        Verdict.TRUST: "LOW",
        Verdict.CAUTION: "MEDIUM",
        Verdict.DO_NOT_USE: "HIGH",
        Verdict.UNABLE_TO_VERIFY: "UNVERIFIED",
    }

    def build(self, *, target: str, verdict: Verdict, confidence: float,
              reasons: list[str], checks: list[CheckResult], lang: str = "en",
              artifact_name: str = "", llm_advice: str | None = None,
              use_debug: bool = False) -> "UserReport":
        key = self.VERDICT_KEY[verdict]
        # spec: EVERY reply states result + confidence band + practical advice
        # UX fix (2026-08-25): UNABLE replies carry NO confidence line — a
        # confidence number on unverified content overclaims certainty exactly
        # when we have the least evidence (the assured-on-empty-evidence class).
        # risk level leads every reply (2026-08-26), then the plain-language
        # verdict sentence — template-first ordering preserved underneath.
        parts = [t("risk_line", lang, level=self._RISK_OF_VERDICT[verdict]),
                 t(key, lang)]
        # UX 2026-08-26: richer narrative — verdict tile (emoji + label + risk)
        # for CAUTION/DO_NOT_USE, then concern bullets, then recommendation.
        if verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE):
            label, risk, emoji = _VERDICT_TILE[verdict]
            tile = t("verdict_tile", lang, label=label,
                     risk=risk.upper(), emoji=emoji)
            parts.append(tile)
            concerns = concerns_for(checks, target, verdict, lang, _keys=False)
            if concerns:
                bullet = "\n".join(f"⚠️ {c}" for c in concerns)
                head = t("concern_count", lang, n=len(concerns),
                         plural="" if len(concerns) == 1 else "s")
                parts.append(head + "\n" + bullet)
            rec = _recommend_line(verdict, lang)
            if rec:
                parts.append(rec)
        n_ran = sum(1 for c in checks if c.status in ("ok", "degraded"))
        # Fusion v2: surface the deepfake pattern explanation if one fired.
        pat = next((r.split(":", 1)[1] for r in reasons
                    if r.startswith("pattern:")), None)
        if pat and not pat.startswith("_") and t(f"pattern_{pat}", lang):
            parts.append(t(f"pattern_{pat}", lang))
        if verdict is not Verdict.UNABLE_TO_VERIFY:
            band = _band(confidence)
            parts.append(t("confidence_line", lang, conf=band))
        elif n_ran:
            # coverage-aware unable: say what DID run so silence ≠ nothing-checked
            parts.append(t("unable_coverage", lang, n=str(n_ran)))

        # targeted practical tips by domain
        tip = self._tip_for(target, verdict, lang)
        if tip:
            parts.append(tip)
        skipped = [c.name for c in checks if c.status == "unavailable"]
        # Freshness-gate drops (e.g. gov-template-cache-stale) surface as
        # machine tokens on the SAME evidence_missing line; the translated
        # sentence stays untouched, the token list is language-neutral.
        gap_tokens = sorted({str(c.signals["evidence_gap"]) for c in checks
                             if isinstance(c.signals, dict)
                             and c.signals.get("evidence_gap")})
        gaps = skipped + gap_tokens
        if gaps and verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE, Verdict.UNABLE_TO_VERIFY):
            line = t("evidence_missing", lang)
            if gap_tokens:
                line += " [" + ", ".join(gap_tokens) + "]"
            parts.append(line)
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
