"""Regression coverage for the ClamAV guard in MaliciousFileCapability._clamav.

Finding B (ZERO_RETENTION_E2E_2026-08-21.md): the availability guard called
os.access("clamscan", os.X_OK) on a BARE name, which is cwd-relative and
therefore always False unless VERISAFE_CLAMSCAN_BIN pointed at an absolute
path -> ClamAV silently reported 'unavailable' even when installed with a
fresh DB. Fixed in f9eea84 (shutil.which resolution + corrected rc semantics:
'FOUND' in output is the detection signal; rc==2 is an error, NOT a hit).

These tests are fully hermetic: a fake clamscan shell stub emulates ClamAV
output/rc conventions, so no real AV or network is needed.
"""
import os
import stat

import pytest

from verisafe.capabilities.malware_file import MaliciousFileCapability
from verisafe.events import Artifact, InputType, JobContext

EICAR = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"

FAKE_CLAMSCAN = """#!/bin/sh
# Emulates clamscan CLI contract: last arg is the scanned file.
# POSIX-sh + builtins ONLY: the test replaces PATH with the stub dir, so
# external coreutils (grep/cat) would not resolve.
for a in "$@"; do target="$a"; done
# NB: read returns 1 at EOF-without-newline but still fills $line — never
# reset it in an || fallback.
line=""
IFS= read -r line < "$target" 2>/dev/null || true
case "$line" in
    *EICAR-STANDARD-ANTIVIRUS-TEST-FILE*)
        echo "/scanned/path: Eicar-Test-Signature FOUND"
        exit 1 ;;
esac
echo "----------- SCAN SUMMARY -----------"
echo "Infected files: 0"
exit 0
"""

FAKE_CLAMSCAN_BROKEN = """#!/bin/sh
# Emulates a clamscan invocation ERROR (rc=2 per ClamAV semantics).
echo "libclamav error: database file corrupt" >&2
exit 2
"""


def _write_executable(path, content):
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def guarded_env(monkeypatch, tmp_path):
    """Isolate the guard from operator env; return (bin_dir, db_dir, qroot)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "daily.cvd").write_bytes(b"fake-db")
    qroot = tmp_path / "q"
    qroot.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("VERISAFE_CLAMD_DB", str(db_dir))
    monkeypatch.delenv("VERISAFE_CLAMSCAN_BIN", raising=False)
    return bin_dir, db_dir, qroot


def _artifact(tmp_path, text="hello world"):
    p = tmp_path / "sample.txt"
    p.write_text(text)
    return Artifact(path=p, original_filename="sample.txt",
                    declared_type=InputType.FILE, sha256="t", size_bytes=p.stat().st_size)


def _ctx(art, qroot):
    return JobContext(job_id="t", artifact=art, quarantine_root=qroot)


def test_clamscan_runs_when_bin_on_path_and_db_present(guarded_env, tmp_path):
    """Finding B core regression: bare 'clamscan' + DB present => scan RUNS.

    Pre-fix behaviour returned status='unavailable' because os.access() on a
    bare name is cwd-relative; the guard must resolve via PATH instead.
    """
    bin_dir, _db, qroot = guarded_env
    _write_executable(bin_dir / "clamscan", FAKE_CLAMSCAN)
    res = MaliciousFileCapability()._clamav(_artifact(tmp_path), _ctx(None, qroot))[0]
    assert res.status == "ok", f"ClamAV did not run [{res.notes}] signals={res.signals}"
    assert res.signals["detected"] is False


def test_clamav_found_line_is_a_detection(guarded_env, tmp_path):
    """rc=1 + 'NAME FOUND' line => detected=True, sig carries the FOUND line."""
    bin_dir, _db, qroot = guarded_env
    _write_executable(bin_dir / "clamscan", FAKE_CLAMSCAN)
    art = _artifact(tmp_path, EICAR)
    res = MaliciousFileCapability()._clamav(art, _ctx(art, qroot))[0]
    assert res.status == "ok"
    assert res.signals["detected"] is True
    assert "FOUND" in res.signals["sig"]
    assert "Eicar-Test-Signature" in res.notes


def test_clamav_error_rc_is_not_a_detection(guarded_env, tmp_path):
    """rc=2 (error) without a FOUND line must NOT count as a detection."""
    bin_dir, _db, qroot = guarded_env
    _write_executable(bin_dir / "clamscan", FAKE_CLAMSCAN_BROKEN)
    res = MaliciousFileCapability()._clamav(_artifact(tmp_path), _ctx(None, qroot))[0]
    assert res.signals.get("detected") is False


def test_clamav_honest_unavailable_when_binary_missing(guarded_env, tmp_path):
    """No clamscan reachable via PATH => truthful 'unavailable', scan skipped."""
    _bin_dir, _db, qroot = guarded_env  # bin_dir left empty on purpose
    res = MaliciousFileCapability()._clamav(_artifact(tmp_path), _ctx(None, qroot))[0]
    assert res.status == "unavailable"


def test_clamav_absolute_bin_override_still_honoured(guarded_env, tmp_path, monkeypatch):
    """VERISAFE_CLAMSCAN_BIN with an absolute path bypasses PATH lookup."""
    bin_dir, _db, qroot = guarded_env
    alt = _write_executable(tmp_path / "alt-clamscan", FAKE_CLAMSCAN)
    monkeypatch.setenv("VERISAFE_CLAMSCAN_BIN", str(alt))
    assert os.access(str(alt), os.X_OK)
    del bin_dir
    res = MaliciousFileCapability()._clamav(_artifact(tmp_path), _ctx(None, qroot))[0]
    assert res.status == "ok"
