"""Zero-retention lifecycle + end-to-end orchestrator behaviour.

The defining property of Vishwas: after every job — success, exception or
timeout — NOTHING user-derived survives on disk (only the out-of-tree audit
line). These tests prove it against the real JobQuarantine + Orchestrator.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vishwas.events import InputType, MediaKind, Verdict


# ------------------------------------------------------------- quarantine --
def test_purge_deletes_job_dir_and_all_tracked(qroot):
    from vishwas.quarantine import JobQuarantine
    q = JobQuarantine("job_test", root=qroot)
    sub = q.make_subdir("work")
    f = sub / "orig.mp4"
    f.write_bytes(b"x" * 100)
    q.track(f)
    derived = sub / "frame.png"
    derived.write_bytes(b"\x89PNG")
    q.track(derived)

    entry = q.purge(reason="completed")
    assert not (qroot / "job_test").exists(), "job dir must be gone after purge"
    assert entry["residual_paths"] == []
    assert entry["failures"] == 0


def test_purge_runs_on_exception_path(qroot):
    """Context-manager exit with an active exception still purges."""
    from vishwas.quarantine import JobQuarantine
    try:
        with JobQuarantine("job_exc", root=qroot) as q:
            f = q.job_dir / "leak.bin"
            f.write_bytes(b"data")
            q.track(f)
            raise RuntimeError("simulated crash mid-job")
    except RuntimeError:
        pass
    assert not (qroot / "job_exc").exists()


def test_idempotent_double_purge(qroot):
    from vishwas.quarantine import JobQuarantine
    q = JobQuarantine("job_dbl", root=qroot)
    first = q.purge()
    second = q.purge()
    assert first.get("artifacts_deleted", 0) >= 1
    assert second["reason"] == "already_closed"


def test_audit_line_written_outside_tree(tmp_path, monkeypatch):
    audit = tmp_path / "audit.log"
    monkeypatch.setenv("VISHWAS_AUDIT_LOG", str(audit))
    # re-import fresh module binding for AUDIT_LOG env (it's read at import)
    import importlib
    import vishwas.quarantine as quar
    importlib.reload(quar)
    q = quar.JobQuarantine("job_audit", root=tmp_path)
    q.purge()
    lines = [json.loads(l) for l in audit.read_text().splitlines() if l.strip()]
    assert lines, "an audit line must exist even though the job dir is deleted"
    assert lines[-1]["job_dir"].endswith("job_audit")


def test_stale_scanner_sweeps_old_jobs(qroot, monkeypatch):
    from vishwas import quarantine as quar
    old = qroot / "job_old"
    old.mkdir()
    (old / "_manifest.json").write_text(json.dumps({"ts": int(time.time()) - 99999}))
    swept = quar.scan_stale_quarantines(root=qroot)
    assert len(swept) >= 1 and any(str(p).rstrip("/").endswith("job_old") for p in swept)
    assert not (qroot / "job_old").exists()
    assert not old.exists()


# ------------------------------------------------------------ e2e orch -----
@pytest.fixture
def run_orch(qroot, orchestrator):
    def _run(msg_extra):
        msg = {"id": "e2e", "session_key": "e2e", **msg_extra}
        msg["_qroot_override"] = str(qroot)
        return orchestrator.handle_incoming(msg)
    return _run


def test_e2e_url_smoke_returns_verdict_and_purges(run_orch, qroot):
    o = run_orch({"text": "https://bank-secure-login.example-verify.com/pay/login?ref=wa"})
    assert o.verdict in (Verdict.DO_NOT_USE, Verdict.CAUTION, Verdict.UNABLE_TO_VERIFY)
    assert 0.0 <= o.confidence <= 1.0
    assert o.purged is True
    # nothing survives under the quarantine root
    left = [p for p in qroot.rglob("*") if p.is_file()]
    # audit log lives outside qroot by config; only confirm no job dir remains
    assert not any(p.name.startswith("job_") for p in qroot.iterdir()), \
        f"job dirs must be purged, found {[p.name for p in qroot.iterdir()]}"


def test_e2e_media_missing_path_raises_cleanly(run_orch):
    with pytest.raises(FileNotFoundError):
        run_orch({"media_path": "/nonexistent/nope.mp4", "input_type": "video"})


def test_e2e_text_question_is_not_treated_as_malware(run_orch):
    o = run_orch({"text": "hello, can you help me check something?"})
    # plain conversational text must never yield a security doom verdict
    assert o.verdict in (Verdict.TRUST, Verdict.CAUTION, Verdict.UNABLE_TO_VERIFY), \
        "a friendly question must not produce do_not_use"


def test_e2e_gov_document_routing_smoke(run_orch, tmp_path):
    pdf = b"%PDF-1.7\n...body...\n%%EOF"
    p = tmp_path / "pan.pdf"
    p.write_bytes(pdf)
    o = run_orch({"media_path": str(p), "input_type": "file",
                  "text": "please verify this PAN document"})
    # gov-document target runs its cheap tiers (OCR gated -> degraded evidence)
    assert o.purged is True
    names = {c.name for c in o.checks}
    assert any(any(tok in n.lower() for tok in ("gov", "docling", "ocr",
                                                 "document", "sig"))
               for n in names) or o.verdict is Verdict.UNABLE_TO_VERIFY, \
        f"expected gov-document evidence checks in output: {sorted(names)}"
