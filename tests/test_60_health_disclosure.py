"""test_60_health_disclosure.py — F7: unauthenticated /health must not leak the
capability inventory / quarantine occupancy. The local host (loopback — where
the cron health probe runs) and a client presenting a valid VISHWAS_HEALTH_TOKEN
still get the full ops detail.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _StubOrch:
    i18n_default = "en"
    def handle_incoming(self, msg, followup_sender=None):
        class _O:
            user_message = "ok"
            def to_dict(self):
                return {"verdict": "trust", "checks": []}
        return _O()


def _proc(tmp_path):
    from vishwas.channels import MessageProcessor
    return MessageProcessor(_StubOrch(), openwa=None, workdir=tmp_path,
                            persist_outcomes=False)


def _invoke_health_get(proc, deps, client_ip, headers=None):
    """Call WebhookHandler.do_GET('/health') directly (no socket bound)."""
    from vishwas.app import WebhookHandler
    h = WebhookHandler.__new__(WebhookHandler)
    h.path = "/health"
    h.requestline = "GET /health HTTP/1.0"
    h.request_version = "HTTP/1.0"
    h.processor = proc
    h._deps = set(deps)
    h.client_address = (client_ip, 12345)
    h.headers = dict(headers or {})
    buf = io.BytesIO()
    h.wfile = buf
    h.do_GET()
    return json.loads(buf.getvalue().partition(b"\r\n\r\n")[2].decode())


def test_f7_network_client_gets_minimal_payload(tmp_path, monkeypatch):
    """A non-loopback client must NOT see the capability inventory / occupancy."""
    monkeypatch.delenv("VISHWAS_HEALTH_TOKEN", raising=False)
    proc = _proc(tmp_path)
    payload = _invoke_health_get(proc, {"vt", "model-weights"}, "203.0.113.5")
    assert payload["status"] == "ok"
    assert "uptime_s" in payload
    for leaked in ("deps", "deps_available", "quarantines_open", "device",
                   "jobs_total", "jobs_ok", "jobs_failed"):
        assert leaked not in payload, f"network client saw sensitive field: {leaked}"


def test_f7_loopback_client_gets_full_payload(tmp_path, monkeypatch):
    """The local health probe (loopback) still gets the full ops detail."""
    monkeypatch.delenv("VISHWAS_HEALTH_TOKEN", raising=False)
    proc = _proc(tmp_path)
    payload = _invoke_health_get(proc, {"vt", "model-weights"}, "127.0.0.1")
    assert payload["deps"] == {"available": ["model-weights", "vt"], "count": 2}
    assert "quarantines_open" in payload
    assert "device" in payload


def test_f7_ipv6_loopback_gets_full_payload(tmp_path, monkeypatch):
    monkeypatch.delenv("VISHWAS_HEALTH_TOKEN", raising=False)
    proc = _proc(tmp_path)
    assert "deps" in _invoke_health_get(proc, {"vt"}, "::1")


def test_f7_valid_token_grants_detail_to_remote(tmp_path, monkeypatch):
    monkeypatch.setenv("VISHWAS_HEALTH_TOKEN", "s3cr3t-token")
    proc = _proc(tmp_path)
    payload = _invoke_health_get(proc, {"vt"}, "203.0.113.5",
                                 headers={"X-Vishwas-Health-Token": "s3cr3t-token"})
    assert "deps" in payload


def test_f7_wrong_token_gets_minimal(tmp_path, monkeypatch):
    monkeypatch.setenv("VISHWAS_HEALTH_TOKEN", "s3cr3t-token")
    proc = _proc(tmp_path)
    payload = _invoke_health_get(proc, {"vt"}, "203.0.113.5",
                                 headers={"X-Vishwas-Health-Token": "wrong"})
    assert "deps" not in payload


def test_f7_is_loopback_classifier():
    from vishwas.app import _is_loopback
    assert _is_loopback("127.0.0.1") is True
    assert _is_loopback("::1") is True
    assert _is_loopback("203.0.113.5") is False
    assert _is_loopback("10.0.0.5") is False
    assert _is_loopback("not-an-ip") is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
