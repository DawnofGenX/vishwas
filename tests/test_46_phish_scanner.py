"""PhishingScanner url_phishing tests (2026-08-26 operator decision).

PhishingScanner (vendored, MIT, src/vishwas/_phishscanner/) is the SOLE url
phishing detector — VirusTotal / offline-DOM heuristics / url-phishml were
dropped from the ``url_phishing`` fusion target. These tests are hermetic:

  (a) unit — the capability maps PhishingScanner's ScanResult risk_score to
      risk_score_norm correctly against a stubbed detector (no network);
  (b) honesty — a scan error/timeout yields status=unavailable, NEVER a
      fabricated risk score;
  (c) integration — FusionEngine.decide('url_phishing', ...) reads a
      url_phish_scanner norm of 0.9 as DO_NOT_USE and 0.1 as CAUTION.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vishwas.capabilities import UrlPhishScannerCapability
from vishwas.capabilities.base import CheckResult
from vishwas.events import Artifact, InputType, JobContext
from vishwas.fusion import FusionEngine
from vishwas.events import Verdict


# ---------------------------------------------------------------- helpers --
def _art(qroot: Path, url: str) -> Artifact:
    ap = Artifact(path=qroot / "u.txt", original_filename="u.txt",
                  declared_type=InputType.TEXT)
    ap.path.write_text(url)
    return ap


def _ctx(qroot: Path, ap: Artifact, budget_s: float = 10.0) -> JobContext:
    return JobContext(job_id="t46", artifact=ap, quarantine_root=qroot,
                      deadline_mono=time.monotonic() + budget_s,
                      wall_budget_s=budget_s)


class _StubScanResult:
    def __init__(self, risk_score, is_phishing, indicators=None, response_time=0.1):
        self.url = "https://stub"
        self.timestamp = time.time()
        self.risk_score = risk_score
        self.is_phishing = is_phishing
        self.indicators = indicators if indicators is not None else []
        self.details = {}
        self.response_time = response_time


class _StubDetector:
    """Minimal stand-in for vendored PhishingDetector.scan_url — no network."""
    def __init__(self, scan_result=None, exc=None, block_s=0.0):
        self._result = scan_result
        self._exc = exc
        self._block_s = block_s
        self.calls = []

    def scan_url(self, url):
        self.calls.append(url)
        if self._block_s:
            time.sleep(self._block_s)
            return self._result
        if self._exc is not None:
            raise self._exc
        return self._result


def _cap_with(monkeypatch, detector) -> tuple[UrlPhishScannerCapability, _StubDetector]:
    cap = UrlPhishScannerCapability()
    monkeypatch.setattr(cap, "_load_detector", lambda: detector)
    return cap, detector


# ------------------------------------------------------------- (a) unit ----
@pytest.mark.parametrize("risk,is_phish,expected_norm", [
    (0, False, 0.0),
    (15, False, 0.15),
    (70, True, 0.7),
    (100, True, 1.0),
])
def test_risk_score_norm_mapping(monkeypatch, tmp_path, risk, is_phish, expected_norm):
    ap = _art(tmp_path, "https://www.wikipedia.org")
    cap, det = _cap_with(monkeypatch, _StubDetector(_StubScanResult(risk, is_phish)))
    out = cap.analyze(ap, _ctx(tmp_path, ap))
    assert len(out) == 1
    c = out[0]
    assert c.name == "url_phish_scanner"
    assert c.status == "ok"
    assert c.signals["risk_score"] == risk
    assert c.signals["risk_score_norm"] == pytest.approx(expected_norm)
    assert 0.0 <= c.signals["risk_score_norm"] <= 1.0
    assert c.signals["is_phishing"] is is_phish


def test_no_url_emits_unavailable(monkeypatch, tmp_path):
    ap = Artifact(path=tmp_path / "u.txt", original_filename="u.txt",
                  declared_type=InputType.TEXT)
    ap.path.write_text("   ")  # blank -> no URL found
    cap, _ = _cap_with(monkeypatch, _StubDetector(_StubScanResult(0, False)))
    out = cap.analyze(ap, _ctx(tmp_path, ap))
    assert out[0].status == "unavailable"


# ------------------------------------------------------------- (b) honesty --
def test_scan_error_is_unavailable_not_fabricated(monkeypatch, tmp_path):
    ap = _art(tmp_path, "https://www.wikipedia.org")
    cap, _ = _cap_with(monkeypatch,
                       _StubDetector(exc=RuntimeError("whois server unreachable")))
    out = cap.analyze(ap, _ctx(tmp_path, ap))
    assert out[0].status == "unavailable"
    # no risk_score_norm is fabricated on a failed scan
    assert "risk_score_norm" not in out[0].signals
    assert out[0].signals.get("error_class") == "RuntimeError"


def test_scan_timeout_is_unavailable(monkeypatch, tmp_path):
    ap = _art(tmp_path, "https://www.wikipedia.org")
    # detector blocks forever; ctx budget 0.05s -> wall-clock timeout triggers
    cap, _ = _cap_with(monkeypatch, _StubDetector(_StubScanResult(5, False), block_s=5.0))
    out = cap.analyze(ap, _ctx(tmp_path, ap, budget_s=0.05))
    assert out[0].status == "unavailable"
    assert "timed out" in out[0].notes
    assert "risk_score_norm" not in out[0].signals


# -------------------------------------------------------- (c) integration ----
def test_fusion_high_risk_norm_do_not_use():
    ck = CheckResult("url_phish_scanner", "mid", "ok",
                     {"risk_score": 90, "risk_score_norm": 0.9, "is_phishing": True})
    d = FusionEngine().decide("url_phishing", [ck])
    assert d.verdict is Verdict.DO_NOT_USE
    assert d.score >= 0.70


def test_fusion_low_risk_norm_caution():
    # NOTE 2026-08-26: benign URL clean-side override now promotes low-risk
    # (norm<0.30, not flagged) to TRUST/LOW — so this reads trust, not caution.
    ck = CheckResult("url_phish_scanner", "mid", "ok",
                     {"risk_score": 10, "risk_score_norm": 0.1, "is_phishing": False})
    d = FusionEngine().decide("url_phishing", [ck])
    assert d.verdict is Verdict.TRUST, d.reasons


def test_benign_url_reads_low():
    # clean-side override: scanner ran, not flagged, low score -> TRUST/LOW
    ck = CheckResult("url_phish_scanner", "mid", "ok",
                     {"risk_score": 15, "risk_score_norm": 0.15, "is_phishing": False,
                      "indicators": ["benign"]}, "")
    d = FusionEngine().decide(target="url_phishing", checks=[ck])
    assert d.verdict == Verdict.TRUST, d.reasons


def test_flagged_phish_never_reads_low():
    # scanner flagged it (is_phishing True) and high score -> NEVER TRUST/LOW
    ck = CheckResult("url_phish_scanner", "mid", "ok",
                     {"risk_score": 88, "risk_score_norm": 0.88, "is_phishing": True,
                      "indicators": ["DNS resolution failed", "Connection failed"]}, "")
    d = FusionEngine().decide("url_phishing", [ck])
    assert d.verdict is not Verdict.TRUST, d.reasons


def test_fusion_only_maps_url_phish_scanner_signal():
    """The url_phishing target has exactly one weighted signal: phish_scanner."""
    from vishwas import fusion as _fm
    assert set(_fm.WEIGHTS["url_phishing"]) == {"phish_scanner.risk_norm"}
    spec = _fm._SIGNAL_SOURCES["phish_scanner.risk_norm"]
    assert spec[0] == "url_phish_scanner"
    assert spec[1] == "risk_score_norm"
    # an old-signal CheckResult contributes nothing to the verdict now
    old = CheckResult("vt_url_reputation", "mid", "ok", {"positives_ratio": 1.0})
    d = FusionEngine().decide("url_phishing", [old])
    assert d.verdict is Verdict.UNABLE_TO_VERIFY  # unmapped -> no usable weight


def test_concern_bullets_fire_on_high_risk_norm():
    from vishwas.report import concerns_for
    ck = CheckResult("url_phish_scanner", "mid", "ok",
                     {"risk_score": 70, "risk_score_norm": 0.7, "is_phishing": True})
    got = concerns_for([ck], "url_phishing", Verdict.DO_NOT_USE, "en")
    assert "concern_url_flag" in got
    assert "concern_url_typo" in got