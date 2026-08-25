"""Shared pytest config for Vishwas.

Sets hermetic quarantine/audit paths BEFORE any vishwas module is imported
(quarantine.py reads env at import time), adds src/ to sys.path, and provides
fixtures for building a fully-gated-off orchestrator (no heavy deps -> cheap
deterministic behaviour, no network, no weights).
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

# --- hermetic isolation (must happen before importing vishwas.*) ----------
os.environ.setdefault("VISHWAS_QUARANTINE", str(_ROOT / ".test-quarantine"))
os.environ.setdefault("VISHWAS_AUDIT_LOG", str(_ROOT / ".test-quarantine" / "audit.log"))
os.environ.setdefault("VISHWAS_STALE_TTL_S", "7200")
sys.path.insert(0, str(_ROOT / "src"))

# Local dependency library (trimmed asn1crypto for the PAdES/CMS stage).
# APPEND (not insert): keeps existing resolution for every already-importable
# module; only un-resolved names (asn1crypto) fall through here.
_PYLIBS = Path.home() / "pylibs"
if _PYLIBS.is_dir() and str(_PYLIBS) not in sys.path:
    sys.path.append(str(_PYLIBS))


import pytest


@pytest.fixture
def qroot(tmp_path):
    """A throwaway quarantine root for one test (assertable post-run)."""
    d = tmp_path / "q"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def orchestrator(qroot):
    """Fully gated-off orchestrator: cheap tiers only, deterministic, offline."""
    import vishwas.app as app
    orch = app.build_orchestrator(set())   # empty dep set => only cheap checks run
    return orch


@pytest.fixture
def orch_with_deps(qroot):
    """Orchestrator with lightweight deps enabled (vt/browser off by default)."""
    import vishwas.app as app
    orch = app.build_orchestrator(set())
    return orch


def residual_files(root: Path) -> list[Path]:
    """Every file left under a quarantine root (should be [] after purge)."""
    out = []
    if root.exists():
        out = [p for p in root.rglob("*") if p.is_file()]
    return out


@pytest.fixture
def assert_zero_retention():
    """Return a helper that fails the test if anything survives a quarantine dir."""
    def _check(root: Path):
        left = residual_files(root)
        assert not left, f"zero-retention violated, residual: {[str(p) for p in left]}"
    return _check
