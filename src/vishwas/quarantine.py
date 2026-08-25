"""Zero-retention data lifecycle.

Every job processes inside its own quarantine dir under QUARANTINE_ROOT.
A manifest tracks every path created (original + derived artifacts: frames,
audio, OCR text, sandbox dirs, PCAPs, dumps, browser profiles). The purger
runs on success, exception AND timeout, then appends an audit line OUTSIDE
the quarantine tree. A stale-scanner sweeps quarantines older than the TTL.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

QUARANTINE_ROOT = Path(os.environ.get("VISHWAS_QUARANTINE", str(Path.home() / "vishwas" / "quarantine")))
AUDIT_LOG = Path(os.environ.get("VISHWAS_AUDIT_LOG", str(QUARANTINE_ROOT / ".." / "logs" / "purge_audit.log")))
STALE_TTL_S = int(os.environ.get("VISHWAS_STALE_TTL_S", 7200))


class JobQuarantine:
    """Owns the isolated workspace for one job and guarantees cleanup."""

    def __init__(self, job_id: str, root: Path | None = None):
        self.root = (root or QUARANTINE_ROOT)
        self.job_dir = self.root / job_id
        self.manifest_path = self.job_dir / "_manifest.json"
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self._created: list[Path] = []
        self._closed = False

    # -- tracking ---------------------------------------------------------
    def track(self, *paths: Path | str) -> list[Path]:
        """Register paths as job-owned; parents are created. Returns one entry
        per requested path (existing files or freshly-created parents) so
        callers can use the returned path as a write destination."""
        out: list[Path] = []
        for p in paths:
            pp = Path(p)
            if pp.exists():
                out.append(pp)
            else:
                pp.parent.mkdir(parents=True, exist_ok=True)
                out.append(pp)
            self._created.append(pp)
        self._persist_manifest()
        return out

    def make_subdir(self, name: str) -> Path:
        d = self.job_dir / name
        d.mkdir(parents=True, exist_ok=True)
        self._created.append(d)
        self._persist_manifest()
        return d

    def _persist_manifest(self) -> None:
        try:
            self.manifest_path.write_text(json.dumps({
                "job_dir": str(self.job_dir),
                "created": [str(p) for p in dict.fromkeys(self._created)],
                "ts": int(time.time()),
            }, indent=0))
        except OSError:
            pass

    # -- teardown ----------------------------------------------------------
    def purge(self, reason: str = "completed") -> dict:
        """Delete everything created, incl. the job dir itself. Idempotent."""
        if self._closed:
            return {"job_dir": str(self.job_dir), "deleted": False, "reason": "already_closed"}
        self._closed = True
        deleted, failures = 0, 0
        # explicit manifest first (covers files added after last persist)
        tracked = set(self._created)
        try:
            m = json.loads(self.manifest_path.read_text())
            tracked |= {Path(x) for x in m.get("created", [])}
        except Exception:
            pass
        for p in sorted(tracked, key=str, reverse=True):
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                    if not p.exists():
                        deleted += 1
                    continue
                if p.exists() or p.is_symlink():
                    p.unlink(missing_ok=True)
                    deleted += 1
            except OSError:
                failures += 1
        try:
            if self.job_dir.exists():
                shutil.rmtree(self.job_dir, ignore_errors=False)
                deleted += 1
        except OSError:
            failures += 1
        entry = {
            "job_dir": str(self.job_dir),
            "ts": int(time.time()),
            "reason": reason,
            "artifacts_deleted": deleted,
            "failures": failures,
            "residual_paths": [str(p) for p in self.job_dir.rglob("*")] if self.job_dir.exists() else [],
        }
        append_audit(entry)
        return entry

    def __enter__(self) -> "JobQuarantine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.purge(reason="exception" if exc_type else "completed")

    def close_safely(self, reason: str = "completed") -> None:
        self.purge(reason=reason)


def append_audit(entry: dict) -> None:
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def count_open_quarantines(root: Path | None = None) -> int:
    """Number of job dirs currently open (still present) under the root.

    JobQuarantine.purge() removes the job dir itself, so a surviving
    subdirectory means the job has not been closed yet (in flight, or left
    behind by a crash until stale_sweep collects it). Best-effort: any read
    error counts as 0 rather than breaking /health.
    """
    root = root or QUARANTINE_ROOT
    try:
        return sum(1 for p in root.iterdir() if p.is_dir())
    except OSError:
        return 0


def scan_stale_quarantines(root: Path | None = None, now: float | None = None) -> list[str]:
    """Remove job dirs whose manifest ts is older than STALE_TTL_S (crash safety)."""
    root = root or QUARANTINE_ROOT
    now = now or time.time()
    swept: list[str] = []
    if not root.exists():
        return swept
    for job_dir in root.iterdir():
        if not job_dir.is_dir():
            continue
        mp = job_dir / "_manifest.json"
        try:
            if mp.exists():
                ts = json.loads(mp.read_text()).get("ts", 0)
            else:
                ts = job_dir.stat().st_mtime
            if now - ts > STALE_TTL_S:
                q = JobQuarantine(job_dir.name, root=root)
                q._closed = False  # fresh handle pointing at existing dir
                entry = q.purge(reason="stale_scan")
                swept.append(str(job_dir))
                append_audit({"note": "swept", **entry})
        except Exception:
            continue
    return swept
