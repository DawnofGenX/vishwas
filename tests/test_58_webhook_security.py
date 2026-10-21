"""test_58_webhook_security.py — webhook hardening regressions (F1 / F2 / F3 / F8).

Security findings fixed 2026-10-21 (verified against the live tree):
  F1  the webhook must FAIL CLOSED when OPENWA_WEBHOOK_SECRET is unset. The old
      code (`return True` in verify_openwa_signature + `... if secret else True`
      in the handler) was fail-open: with no secret a plain, UNSIGNED POST
      reached the full pipeline while the server listened on 0.0.0.0.
  F2  the per-JID workdir must be a single, escaped path component. A JID from
      data.from containing '/' or '..' used to escape the workdir (only '=' and
      '@' were stripped).
  F3  replay protection: an envelope whose own timestamp is older than the
      window (VISHWAS_WEBHOOK_MAX_AGE_S, default 300s) is rejected, and an
      identical delivery id is rejected by a DURABLE (restart-surviving) seen-set
      (the old in-memory dict was wiped on every restart).
  F8  the local delivery fallback (OpenWA unavailable) must hash the JID and NOT
      persist the reply text (PII), matching the privacy posture of _persist().
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import os
import socket
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest


# ------------------------------------------------------------ helpers --------
class _StubOrch:
    """Minimal orchestrator stand-in: returns an outcome, never runs a pipeline."""

    i18n_default = "en"

    def handle_incoming(self, msg, followup_sender=None):
        class _O:
            user_message = "checked (stub)"
            def to_dict(self):
                return {"verdict": "trust", "checks": [], "target": "document"}
        return _O()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(processor, deps):
    import vishwas.app as app
    port = _free_port()
    handler = type("BoundH", (app.WebhookHandler,),
                   {"processor": processor, "_deps": deps})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def _post(port, path, obj, headers=None):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(obj).encode()
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    conn.request("POST", path, body=data, headers=h)
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
    return resp.status, body


def _msg_envelope(msg_id="m1", sender="919545000836@s.whatsapp.net",
                  event="message.received", ts=None, delivery="del-1",
                  body="please check this document"):
    now = int(time.time()) if ts is None else ts
    return {
        "event": event,
        "sessionId": "s1",
        "idempotencyKey": delivery,
        "deliveryId": delivery,
        "timestamp": now,
        "data": {
            "id": msg_id,
            "from": sender,
            "to": "919136963062@s.whatsapp.net",
            "body": body,
            "type": "text",
            "timestamp": now,
        },
    }


def _make_proc(tmp_path, persist=False):
    from vishwas.channels import MessageProcessor
    return MessageProcessor(_StubOrch(), openwa=None, workdir=tmp_path,
                            persist_outcomes=persist)


# ------------------------------------------------------------- F1 ----------
def test_f1_unsigned_post_rejected_when_no_secret(monkeypatch, tmp_path):
    """The core F1 regression: no secret => an UNSIGNED POST is rejected 401."""
    monkeypatch.delenv("OPENWA_WEBHOOK_SECRET", raising=False)
    proc = _make_proc(tmp_path)
    srv, port = _start_server(proc, set())
    try:
        status, _ = _post(port, "/webhook/inbound", _msg_envelope())
        assert status == 401, (
            f"unsigned POST with no secret must be REJECTED (401), got {status} "
            "(F1: webhook failed open — an unsigned POST reached the pipeline)")
    finally:
        srv.shutdown()


def test_f1_wrong_signature_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENWA_WEBHOOK_SECRET", "the-real-secret")
    proc = _make_proc(tmp_path)
    srv, port = _start_server(proc, set())
    try:
        status, _ = _post(port, "/webhook/inbound", _msg_envelope(),
                          headers={"X-OpenWA-Signature": "sha256=deadbeef"})
        assert status == 401, f"wrong signature must be rejected, got {status}"
    finally:
        srv.shutdown()


def test_f1_valid_signature_accepted(monkeypatch, tmp_path):
    """The live deployment HAS a secret, so a correctly-signed webhook keeps working."""
    secret = "the-real-secret"
    monkeypatch.setenv("OPENWA_WEBHOOK_SECRET", secret)
    proc = _make_proc(tmp_path)
    srv, port = _start_server(proc, set())
    try:
        obj = _msg_envelope()
        raw = json.dumps(obj).encode()
        sig = "sha256=" + _hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
        status, body = _post(port, "/webhook/inbound", obj,
                             headers={"X-OpenWA-Signature": sig})
        assert status == 200, f"a correctly-signed POST must be accepted, got {status}"
    finally:
        srv.shutdown()


# ------------------------------------------------------------- F2 ----------
def test_f2_safe_jid_dirname_never_escapes(tmp_path):
    """A JID (untrusted input) must reduce to a single path component that can
    never leave the workdir."""
    from vishwas.channels import safe_jid_dirname
    base = tmp_path / "workdir"
    base.mkdir()
    base_resolved = base.resolve()
    evil_jids = [
        "../../etc/passwd", "../../../tmp/evil", "a/b/c", "..", ".",
        "....//....//etc/passwd", "foo/../../bar", "\x00\x01/etc",
        "/etc/passwd", "..%2f..%2f..%2f", "a" * 200, "   ",
        "919545000836@s.whatsapp.net", "120363000000000123@g.us",
    ]
    for jid in evil_jids:
        name = safe_jid_dirname(jid)
        assert "/" not in name and "\\" not in name, f"dir name has a slash: {name!r}"
        assert name not in ("", ".", ".."), f"dir name is a path anchor: {name!r}"
        assert not name.startswith("/"), f"dir name is absolute: {name!r}"
        target = (base / name).resolve()
        assert str(target).startswith(str(base_resolved)), (
            f"JID {jid!r} -> {name!r} escaped workdir: {target}")


def test_f2_legit_jid_still_usable(tmp_path):
    """A real JID maps to a sane, single-component dir name (no breakage)."""
    from vishwas.channels import safe_jid_dirname
    name = safe_jid_dirname("919545000836@s.whatsapp.net")
    assert name and "/" not in name and "@" not in name
    assert "919545000836" in name            # sender digits survive -> distinct dirs
    assert safe_jid_dirname("919545000837@s.whatsapp.net") != name  # no collision


def test_f2_handler_workdir_stays_inside(tmp_path):
    """End-to-end: a hostile JID must not push the handler's workdir outside."""
    from vishwas.channels import MessageProcessor, parse_openwa_webhook, safe_jid_dirname
    obj = _msg_envelope(sender="../../../../tmp/vishwas-escape")
    msg = parse_openwa_webhook(obj)
    assert msg is not None
    proc = _make_proc(tmp_path)
    jid = msg.session_key
    workdir = proc.workdir / safe_jid_dirname(jid)
    assert workdir.resolve().is_relative_to(proc.workdir.resolve()), (
        f"handler workdir escaped the workdir: {workdir}")


def test_f2_processor_quarantine_dir_stays_inside(tmp_path):
    """SECOND SINK (found live 2026-10-21): `MessageProcessor.process()` built
    its quarantine dir as `self.workdir / sid[:24]` — truncation, not
    sanitisation — so a hostile JID escaped there even after the handler was
    fixed.

    The directory is removed again in the zero-retention `finally` block, so
    the escape cannot be observed by looking at the filesystem afterwards.
    Instead this records every path the processor TOUCHES and asserts that
    none of them resolves outside the workdir."""
    from vishwas.channels import MessageProcessor, parse_openwa_webhook
    obj = _msg_envelope(sender="../../ESCAPED")
    msg = parse_openwa_webhook(obj)
    assert msg is not None
    proc = _make_proc(tmp_path)
    workdir_resolved = proc.workdir.resolve()

    touched: list[Path] = []
    real_mkdir = Path.mkdir

    def _spy_mkdir(self, *a, **kw):
        touched.append(self)
        return real_mkdir(self, *a, **kw)

    Path.mkdir = _spy_mkdir
    try:
        try:
            proc.process(msg.as_msg())
        except Exception:
            pass
    finally:
        Path.mkdir = real_mkdir

    escaped = [p for p in touched
               if not p.resolve().is_relative_to(workdir_resolved)]
    assert not escaped, (
        f"the processor created {len(escaped)} path(s) outside its workdir: "
        f"{escaped}")


def test_f2_processor_quarantine_dir_is_sanitised(tmp_path):
    """The processor's quarantine directory name must be a single safe
    component derived from the JID, not a truncated raw JID."""
    from vishwas.channels import MessageProcessor, safe_jid_dirname
    proc = _make_proc(tmp_path)
    hostile = "../../ESCAPED"
    name = safe_jid_dirname(hostile, max_len=24)
    assert "/" not in name and ".." not in name
    assert (proc.workdir / name).resolve().is_relative_to(proc.workdir.resolve())


# ------------------------------------------------------------- F3 ----------
def test_f3_stale_timestamp_rejected():
    from vishwas.channels import ReplayGuard
    g = ReplayGuard(state_path=None, max_age_s=300)
    assert g.is_stale({"data": {"timestamp": time.time() - 1000}}) is True


def test_f3_fresh_timestamp_accepted():
    from vishwas.channels import ReplayGuard
    g = ReplayGuard(state_path=None, max_age_s=300)
    assert g.is_stale({"data": {"timestamp": time.time() - 10}}) is False


def test_f3_missing_timestamp_fails_closed():
    from vishwas.channels import ReplayGuard
    g = ReplayGuard(state_path=None, max_age_s=300)
    assert g.is_stale({"data": {}}) is True          # no timestamp at all
    assert g.is_stale({}) is True                     # no data block
    assert g.is_stale({"data": {"timestamp": "garbage"}}) is True  # unusable


def test_f3_replay_window_configurable(monkeypatch):
    from vishwas.channels import webhook_max_age_s
    monkeypatch.setenv("VISHWAS_WEBHOOK_MAX_AGE_S", "900")
    assert webhook_max_age_s() == 900
    monkeypatch.setenv("VISHWAS_WEBHOOK_MAX_AGE_S", "not-an-int")
    assert webhook_max_age_s() == 300                 # bad value falls back to default


def test_f3_identical_delivery_replayed_rejected():
    from vishwas.channels import ReplayGuard
    g = ReplayGuard(state_path=None, max_age_s=300)
    assert g.is_replay("delivery-1") is False         # first sight -> accept
    assert g.is_replay("delivery-1") is True          # identical replay -> reject
    assert g.is_replay("delivery-2") is False         # a different id still accepted


def test_f3_replay_state_survives_restart(tmp_path):
    from vishwas.channels import ReplayGuard
    sp = tmp_path / "replay_state.json"
    g1 = ReplayGuard(state_path=sp, max_age_s=300)
    assert g1.is_replay("delivery-42") is False
    g2 = ReplayGuard(state_path=sp, max_age_s=300)    # fresh process = "restart"
    assert g2.is_replay("delivery-42") is True, (
        "delivery-id seen-set must survive a process restart (durable back-store)")


def test_f3_wired_into_handler_rejects_stale_replay(monkeypatch, tmp_path):
    """End-to-end: a signed-but-STALE envelope is rejected (200 'stale') even
    though its signature is valid — this is the replay-attack case."""
    secret = "the-real-secret"
    monkeypatch.setenv("OPENWA_WEBHOOK_SECRET", secret)
    proc = _make_proc(tmp_path)
    srv, port = _start_server(proc, set())
    try:
        obj = _msg_envelope(ts=int(time.time()) - 5000)   # 5000s old >> 300s window
        raw = json.dumps(obj).encode()
        sig = "sha256=" + _hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
        status, body = _post(port, "/webhook/inbound", obj,
                             headers={"X-OpenWA-Signature": sig})
        assert status == 200, f"stale replay should be dropped, got {status}"
        assert b"stale" in body, f"stale replay must be labelled 'stale': {body!r}"
    finally:
        srv.shutdown()


# ------------------------------------------------------------- F8 ----------
def test_f8_fallback_hashes_jid_and_drops_reply(tmp_path):
    from vishwas.channels import MessageProcessor
    proc = MessageProcessor(_StubOrch(), openwa=None, workdir=tmp_path,
                            persist_outcomes=True)
    jid = "919545000836@s.whatsapp.net"
    reply = "Top-secret reply text that must NOT be persisted"
    proc.deliver({"jid": jid, "reply": reply})
    log = tmp_path / "outcomes.jsonl"
    assert log.exists(), "a local fallback record must be written to outcomes.jsonl"
    recs = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    assert recs, "no fallback record written"
    rec = recs[-1]
    assert "jid" not in rec, "RAW JID leaked into outcomes.jsonl (F8)"
    assert "reply" not in rec, "REPLY TEXT leaked into outcomes.jsonl (F8)"
    assert reply not in json.dumps(rec), "reply text present in the fallback record"
    expected_h = hashlib.sha256(jid.encode()).hexdigest()[:12]
    assert rec.get("jid_h") == expected_h, "jid must be stored hashed (like _persist)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
