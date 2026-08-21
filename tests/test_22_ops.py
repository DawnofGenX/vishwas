"""Task 2.3 ops surface: rich GET /health, thread-safe job counters.

Hermetic by construction: the HTTP handler is exercised by DIRECT invocation
(no socket is ever bound), the orchestrator is a stub, and quarantine roots
are tmp_path-scoped. No network, no weights.
"""
import io
import json
import threading
from pathlib import Path

import pytest


# ------------------------------------------------------------- stubs --------

class _FakeOutcome:
    """Minimal JobOutcome stand-in: only what MessageProcessor.process() reads."""

    def __init__(self, verdict: str = "trust"):
        self.user_message = "checked; looks fine"
        self._verdict = verdict

    def to_dict(self) -> dict:
        return {"verdict": self._verdict}


class _FakeOrch:
    """Orchestrator stand-in: returns an outcome, or raises when fail=True."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def handle_incoming(self, msg_dict, followup_sender=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated pipeline crash")
        return _FakeOutcome()


def _make_processor(tmp_path, orch):
    from verisafe.channels import MessageProcessor
    return MessageProcessor(orch, openwa=None, persist_outcomes=False,
                            workdir=tmp_path / "work")


def _invoke_health_get(proc, deps=()):
    """Call WebhookHandler.do_GET('/health') directly against a BytesIO wfile."""
    from verisafe.app import WebhookHandler
    h = WebhookHandler.__new__(WebhookHandler)   # skip socket/stdio wiring
    h.path = "/health"
    h.requestline = "GET /health HTTP/1.0"
    h.request_version = "HTTP/1.0"
    h.processor = proc
    h._deps = set(deps)
    buf = io.BytesIO()
    h.wfile = buf
    h.do_GET()
    raw = buf.getvalue()
    status_line, _, body = raw.partition(b"\r\n\r\n")
    assert b" 200 " in status_line, status_line
    headers = raw.partition(b"\r\n\r\n")[0]
    assert b"Content-Type: application/json" in headers
    return json.loads(body.decode())


# ------------------------------------------------- rich /health payload -----

def test_health_payload_keys_and_types(tmp_path):
    from verisafe.app import health_snapshot
    proc = _make_processor(tmp_path, _FakeOrch())
    payload = _invoke_health_get(proc, deps={"vt", "llm"})

    # full schema present
    for key in ("status", "uptime_s", "jobs_total", "jobs_ok", "jobs_failed",
                "quarantines_open", "deps", "deps_available"):
        assert key in payload, f"missing /health key: {key}"

    assert payload["status"] == "ok"
    assert isinstance(payload["uptime_s"], int)
    for key in ("jobs_total", "jobs_ok", "jobs_failed", "quarantines_open"):
        assert isinstance(payload[key], int) and not isinstance(payload[key], bool)

    # deps summary object built from detect_available_deps()-style input
    assert payload["deps"] == {"available": ["llm", "vt"], "count": 2}
    # backward compat: pre-2.3 flat sorted list still exposed
    assert payload["deps_available"] == ["llm", "vt"]

    # zero jobs so far -> counters at baseline
    assert (payload["jobs_total"], payload["jobs_ok"], payload["jobs_failed"]) == (0, 0, 0)


def test_health_uptime_is_number_and_nonnegative(tmp_path):
    import time as _time
    from verisafe.app import health_snapshot
    proc = _make_processor(tmp_path, _FakeOrch())
    seen = []
    for _ in range(3):
        payload = health_snapshot(proc, set())
        assert isinstance(payload["uptime_s"], (int, float))
        assert not isinstance(payload["uptime_s"], bool)
        assert payload["uptime_s"] >= 0
        seen.append(payload["uptime_s"])
    _time.sleep(1.1)
    assert health_snapshot(proc, set())["uptime_s"] >= seen[0]  # monotonic, never decreases


def test_health_quarantines_open_counts_job_dirs(tmp_path, monkeypatch):
    import verisafe.quarantine as q
    from verisafe.app import health_snapshot

    # direct: only subdirectories count, files ignored, missing root -> 0
    root = tmp_path / "q"
    root.mkdir()
    assert q.count_open_quarantines(root) == 0
    for name in ("job-a", "job-b"):
        (root / name).mkdir()
    (root / "stray.txt").write_text("not a dir")
    assert q.count_open_quarantines(root) == 2
    assert q.count_open_quarantines(tmp_path / "does-not-exist") == 0

    # wired into /health via monkeypatched root lookup (hermetic — no real root)
    monkeypatch.setattr(q, "count_open_quarantines",
                        lambda root=None: 7)
    payload = health_snapshot(_make_processor(tmp_path, _FakeOrch()), set())
    assert payload["quarantines_open"] == 7


def test_health_404_for_unknown_paths(tmp_path):
    from verisafe.app import WebhookHandler
    h = WebhookHandler.__new__(WebhookHandler)
    h.path = "/nope"
    h.requestline = "GET /nope HTTP/1.0"
    h.request_version = "HTTP/1.0"
    h.processor = _make_processor(tmp_path, _FakeOrch())
    h._deps = set()
    buf = io.BytesIO()
    h.wfile = buf
    h.do_GET()
    assert b" 404 " in buf.getvalue().partition(b"\r\n\r\n")[0]


# ------------------------------------- job counters on the outcome path -----

def test_counters_increment_on_ok_and_failed_outcomes(tmp_path):
    ok_orch = _FakeOrch(fail=False)
    proc = _make_processor(tmp_path, ok_orch)
    res = proc.process({"id": "j1", "session_key": "sess-ok", "text": "check this"})
    assert "reply" in res and ok_orch.calls == 1

    bad_orch = _FakeOrch(fail=True)
    proc2 = _make_processor(tmp_path, bad_orch)
    with pytest.raises(RuntimeError):
        proc2.process({"id": "j2", "session_key": "sess-bad", "text": "check this"})

    snap = proc.counters.snapshot()
    assert snap == {"jobs_total": 1, "jobs_ok": 1, "jobs_failed": 0}
    snap2 = proc2.counters.snapshot()
    assert snap2 == {"jobs_total": 1, "jobs_ok": 0, "jobs_failed": 1}

    # counters are per-processor (per server process), independent instances
    assert proc.counters is not proc2.counters


def test_counters_thread_safe_under_contention():
    from verisafe.channels import JobCounters
    c = JobCounters()
    N_THREADS, N_ITERS = 8, 400

    def worker():
        for _ in range(N_ITERS):
            c.record_started()
            c.record_ok()

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = N_THREADS * N_ITERS
    assert c.snapshot() == {"jobs_total": total, "jobs_ok": total, "jobs_failed": 0}


def test_counter_reset_semantics_documented():
    """Counters are in-memory: a fresh processor starts at zero (restart sim)."""
    from verisafe.channels import JobCounters
    c = JobCounters()
    c.record_started(); c.record_ok(); c.record_failed()
    fresh = JobCounters()   # what a restarted process holds
    assert fresh.snapshot() == {"jobs_total": 0, "jobs_ok": 0, "jobs_failed": 0}
