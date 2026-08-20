"""Hermetic tests for the VirusTotal v3 reputation client (D6).

Zero network: a fake ``opener`` transport object stands in for urllib.
Covers brief cases a-g:
  a. no API key -> available() False; capabilities keep exact current output
  b. mock 200 clean stats -> 'low' verdict, ok status, counts passed through
  c. mock 200 with 1 malicious -> 'high' verdict
  d. mock 429 twice then 200 -> retried, success, exactly 3 calls
  e. persistent 429 -> after max retries 'unavailable', no exception escapes
  f. URLError/timeout -> 'unavailable', graceful
  g. base64url() correct against known vectors (URL + sha256 forms)
"""
from __future__ import annotations

import io
import json
import types
import urllib.error

import pytest

from verisafe import vt_client
from verisafe.vt_client import (VtClient, VtResult, api_key, available,
                                base64url, check_hash, check_url, map_verdict)


# --------------------------------------------------------------- fakes ----
class FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """Scripted transport: pops responses/exceptions in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.sleeps = []

    def __call__(self, req, timeout=None):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    def sleep(self, s):
        self.sleeps.append(s)


def _client(script):
    op = FakeOpener(script)
    c = VtClient(opener=op, sleep=op.sleep)
    return c, op


def _file_payload(stats: dict, category: str = "executable") -> dict:
    return {"data": {"attributes": {"last_analysis_stats": stats,
                                    "category": category}}}


def _url_payload(overview: dict, category: str = "website") -> dict:
    return {"data": {"attributes": {"last_analysis_results": {"overview": overview},
                                    "category": category}}}


# ------------------------------------------------------------------- a ----
def test_no_key_unavailable_and_capability_output(monkeypatch):
    monkeypatch.delenv("VERISAFE_VT_API_KEY", raising=False)
    assert api_key() is None
    assert available() is False
    res = check_hash("a" * 64)
    assert res.status == "unavailable"
    assert res.counts == {}
    assert "not provisioned" in res.note

    # Capability keeps its EXACT current no-key result verbatim.
    from verisafe.capabilities.malware_file import MaliciousFileCapability
    cap = MaliciousFileCapability()
    art = types.SimpleNamespace(sha256="ab" * 32)
    ctx = types.SimpleNamespace(vt_api_key=None)
    out = cap._vt(art, ctx)
    assert len(out) == 1
    assert out[0].name == "vt_reputation"
    assert out[0].status == "unavailable"
    assert out[0].signals == {"sha256": "ab" * 32}
    assert out[0].notes == "VirusTotal key not provisioned"


# ------------------------------------------------------------------- b ----
def test_clean_stats_low_verdict(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    stats = {"malicious": 0, "suspicious": 0, "undetected": 71, "harmless": 2}
    c, op = _client([_file_payload(stats)])
    res = c.check_hash("c" * 64)
    assert res.status == "ok"
    assert res.verdict == "low"
    assert res.counts == stats
    assert res.positives_ratio == 0.0
    assert res.category == "executable"
    assert op.calls == 1


# ------------------------------------------------------------------- c ----
def test_one_malicious_high_verdict_and_passthrough(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    stats = {"malicious": 1, "suspicious": 2, "undetected": 70}
    c, op = _client([_file_payload(stats, category="trojan")])
    res = c.check_hash("d" * 64)
    assert res.status == "ok"
    assert res.verdict == "high"
    assert res.counts["malicious"] == 1
    assert res.counts["suspicious"] == 2
    assert res.counts["undetected"] == 70
    assert res.category == "trojan"
    assert res.positives_ratio == pytest.approx(3 / 73)


def test_suspicious_counts_mid(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    assert map_verdict({"malicious": 0, "suspicious": 1}) == "mid"
    assert map_verdict({"malicious": 0, "suspicious": 2}) == "mid"
    assert map_verdict({"malicious": 0, "suspicious": 3}) == "high"
    assert map_verdict({"malicious": 1, "suspicious": 0}) == "high"
    assert map_verdict({}) == "low"


def test_url_overview_shape(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    overview = {"malicious": 0, "suspicious": 0, "undetected": 73}
    c, _op = _client([_url_payload(overview)])
    res = c.check_url("https://example.com/phish")
    assert res.status == "ok"
    assert res.counts == overview
    assert res.verdict == "low"


# ------------------------------------------------------------------- d ----
def test_429_twice_then_success(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    e429 = lambda: urllib.error.HTTPError(  # noqa: E731
        "https://x", 429, "rate limited", {"Retry-After": "2"}, io.BytesIO(b""))
    stats = {"malicious": 0, "harmless": 73}
    c, op = _client([e429(), e429(), _file_payload(stats)])
    res = c.check_hash("e" * 64)
    assert res.status == "ok"
    assert res.counts == stats
    assert op.calls == 3
    # Retry-After honoured on both retries (capped at 30s)
    assert op.sleeps[0] == 2.0
    assert op.sleeps[1] == 2.0


# ------------------------------------------------------------------- e ----
def test_persistent_429_exhausts_to_unavailable(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    e429 = lambda: urllib.error.HTTPError(  # noqa: E731
        "https://x", 429, "rate limited", {}, io.BytesIO(b""))
    c, op = _client([e429()])  # single entry repeats
    res = c.check_hash("f" * 64)
    assert res.status == "unavailable"
    assert res.verdict == "low"
    assert op.calls == 4  # initial + 3 retries
    assert len(op.sleeps) == 3
    assert op.sleeps == [1.0, 4.0, 16.0]


# ------------------------------------------------------------------- f ----
def test_urlerror_graceful_unavailable(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    err = urllib.error.URLError("connection timed out")
    c, op = _client([err])
    res = c.check_url("https://unreachable.example/x")
    assert res.status == "unavailable"
    assert "URLError" in res.note
    assert op.calls == 4  # retried with backoff, then gave up


def test_404_is_clean_negative(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    e404 = lambda: urllib.error.HTTPError(  # noqa: E731
        "https://x", 404, "not found", {}, io.BytesIO(b""))
    c, op = _client([e404()])
    res = c.check_hash("0" * 64)
    assert res.status == "ok"
    assert res.raw_status == 404
    assert res.counts == {}
    assert res.verdict == "low"
    assert op.calls == 1  # 404 is not retried


# ------------------------------------------------------------------- g ----
def test_base64url_known_vectors():
    # sha256 form (already url-safe hex, but must round-trip through b64url)
    h = "d41d8cd98f00b204e9800998ecf8427e"
    enc = base64url(h)
    import base64 as _b
    assert _b.urlsafe_b64decode(enc.encode()).decode() == h
    # URL form with characters that need padding-safe encoding
    u = "https://ex.com/a?b=c&d=e"
    enc_u = base64url(u)
    assert _b.urlsafe_b64decode(enc_u.encode()).decode() == u
    assert "+" not in enc_u and "/" not in enc_u
    # exact vector: "https://ex.com/a"
    assert base64url("https://ex.com/a") == "aHR0cHM6Ly9leC5jb20vYQ=="


# ------------------------------------------------- capability integration --
def test_malware_file_vt_ok_result_shape(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    from verisafe.capabilities.malware_file import MaliciousFileCapability
    cap = MaliciousFileCapability()
    art = types.SimpleNamespace(sha256="ab" * 32)
    ctx = types.SimpleNamespace(vt_api_key="test-key")
    stats = {"malicious": 2, "suspicious": 1, "undetected": 70}
    c, _op = _client([_file_payload(stats)])
    monkeypatch.setattr(vt_client, "check_hash", lambda h: c.check_hash(h))
    out = cap._vt(art, ctx)
    assert out[0].name == "vt_reputation"
    assert out[0].status == "ok"
    assert out[0].signals["positives_ratio"] == pytest.approx(3 / 73, abs=1e-3)
    assert out[0].signals["vt_engines_total"] == 73
    assert out[0].signals["vt_verdict"] == "high"
    assert "2 malicious / 1 suspicious of 73 engines" in out[0].notes


def test_malware_file_vt_unavailable_on_exhaustion(monkeypatch):
    monkeypatch.setenv("VERISAFE_VT_API_KEY", "test-key")
    from verisafe.capabilities.malware_file import MaliciousFileCapability
    cap = MaliciousFileCapability()
    art = types.SimpleNamespace(sha256="cd" * 32)
    ctx = types.SimpleNamespace(vt_api_key="test-key")
    bad = VtResult(status="unavailable", note="rate-limited (HTTP 429), retried 3x")
    monkeypatch.setattr(vt_client, "check_hash", lambda h: bad)
    out = cap._vt(art, ctx)
    assert out[0].status == "failed"
    assert out[0].signals["error_class"] == "VtUnavailable"
    assert out[0].notes == "VT lookup failed; local engines still run"
