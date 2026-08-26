"""Vendored PhishingScanner evidence signal (Lintshiwe/PhishingScanner, MIT).

The 2026-08-26 operator decision makes this capability the SOLE URL-phishing
detector in Vishwas: PhishingScanner (vendored at ``_phishscanner/``) decides
URL phishing on its own. The former VirusTotal reputation, offline-DOM
heuristics and vendored-xgboost url-phishml signals were removed from the
``url_phishing`` fusion target.

The scanner does LIVE network work (whois domain-age, SSL cert, HTTP fetch, BS4
content) and returns a ``ScanResult`` with ``risk_score`` (int 0..100) and
``is_phishing`` (risk_score >= 70). We emit ``url_phish_scanner`` evidence with
``risk_score_norm`` (0..1) that feeds the fusion weight
``phish_scanner.risk_norm``.

HONESTY RULE: we never fabricate a risk score. If the scanner raises, times out
against the wall-clock budget, or cannot run, we emit status=``unavailable``
with an honest note — a degraded/unavailable scan is NEVER presented as a
confident risk number.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ..events import Artifact, JobContext
from .base import CheckResult

_VENDOR = Path(__file__).resolve().parent.parent / "_phishscanner"
_DEFAULT_WALL_S = 12.0  # wall-clock cap on one scan_url when no ctx budget exists


class UrlPhishScannerCapability:
    """requires=() keeps core always-runnable; the vendored scanner loads lazily.

    Emits check ``url_phish_scanner`` -> fusion ``phish_scanner.risk_norm``.
    """

    requires: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._detector = None  # lazy PhishingDetector

    def _load_detector(self):
        """Lazy import under the serving PYTHONPATH returns a plain file import."""
        if self._detector is None:
            import sys

            vendor = str(_VENDOR)
            if vendor not in sys.path:
                sys.path.insert(0, vendor)
            from phishing_scanner import PhishingDetector  # type: ignore

            self._detector = PhishingDetector()
        return self._detector

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _budget_s(ctx: JobContext) -> float:
        """Honour the stage budget when present, else hard wall-clock cap."""
        try:
            rem = float(ctx.remaining_s() or 0.0)
        except Exception:
            rem = float("inf")
        return max(0.0, min(_DEFAULT_WALL_S, rem))

    def _scan(self, url: str, budget_s: float):
        """Run scan_url under a wall-clock timeout in a worker thread.

        Returns (detector, ScanResult) or raises TimeoutError via the wrapper.
        """
        detector = self._load_detector()
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, Any] = {"exc": None}

        def _run():
            try:
                result_holder["r"] = detector.scan_url(url)
            except Exception as e:  # noqa: BLE001 — delegate to the caller
                error_holder["exc"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=budget_s)
        if t.is_alive():
            raise TimeoutError(
                f"PhishingScanner did not finish within {budget_s:.1f}s; "
                "network fetch degraded")
        if error_holder["exc"] is not None:
            raise error_holder["exc"]
        assert "r" in result_holder
        return detector, result_holder["r"]

    # ------------------------------------------------------------ analyze --
    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        raw = art.path.read_text(errors="ignore").strip()
        urls_in_text = ctx.extra.get("urls_in_text") or []
        url_str = (urls_in_text or [raw])[0]
        if not url_str:
            return [CheckResult("url_phish_scanner", "mid", "unavailable",
                                {}, "no URL found in message")]
        budget_s = self._budget_s(ctx)
        t0 = time.monotonic()
        try:
            _detector, res = self._scan(url_str, budget_s)
        except TimeoutError as e:
            return [CheckResult(
                "url_phish_scanner", "mid", "unavailable",
                {"timeout_s": round(budget_s, 1)},
                "PhishingScanner scan timed out against wall-clock budget; "
                "no risk score produced (honest)")]
        except Exception as e:  # noqa: BLE001 — scanner load/runtime failure
            return [CheckResult(
                "url_phish_scanner", "mid", "unavailable",
                {"error_class": type(e).__name__},
                "PhishingScanner could not run a scan; no risk score "
                "fabricated")]
        dur = round(time.monotonic() - t0, 3)
        risk = int(getattr(res, "risk_score", 0))
        is_phishing = bool(getattr(res, "is_phishing", False))
        indicators = list((getattr(res, "indicators", None) or [])[:5])
        return [CheckResult(
            "url_phish_scanner", "mid", "ok",
            {"risk_score": risk,
             "risk_score_norm": round(min(100, max(0, risk)) / 100.0, 3),
             "is_phishing": is_phishing,
             "indicators": indicators,
             "response_time_s": round(float(getattr(res, "response_time", 0.0) or 0.0), 3)},
            ("PhishingScanner flags this as phishing"
             if is_phishing else "PhishingScanner does not flag this as phishing"),
            duration_s=dur)]