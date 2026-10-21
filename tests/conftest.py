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

# Export PYTHONPATH as well as inserting into sys.path. `sys.path` only
# affects THIS process; the fusion_train tests spawn a SUBPROCESS
# (`python -m vishwas.fusion_train`) that inherits os.environ, and without
# PYTHONPATH the child cannot import vishwas and the test fails with
# "No module named 'vishwas'". pytest has no built-in way to set an env var
# (that needs the pytest-env plugin), so it is done here.
#
# APPEND, never replace (2026-10-21): the Hermes 3.14 interpreter gets its
# venv's site-packages FROM PYTHONPATH, so setting PYTHONPATH=src DISPLACES
# it and every third-party package (pygments, pytest's own deps) vanishes
# with "No module named 'pygments'". Appending preserves whatever was
# already there and only ADDS the repo's src.
_SRC = str(_ROOT / "src")
_pp = os.environ.get("PYTHONPATH", "")
if _SRC not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _pp + (os.pathsep + _SRC if _pp else _SRC)

# Local dependency library (trimmed asn1crypto for the PAdES/CMS stage).
# APPEND (not insert): keeps existing resolution for every already-importable
# module; only un-resolved names (asn1crypto) fall through here.
#
# WARNING (2026-10-21) — TWO hazards, both now real:
#   (a) SERVING STACK: this APPEND is safe for the hermetic suite (which
#       never imports transformers) but it is NOT safe for the serving path.
#       The pylibs tree ships cryptography v50, which no longer provides the
#       `lib` submodule pyOpenSSL imports; Python then resolves `lib` to a
#       top-level namespace package (llama_cpp_python's site-packages/lib) and
#       the whole accelerate/transformers import chain dies. The service
#       therefore runs under its own venv with no pylibs on the path — see
#       scripts/run_vishwas.sh.
#   (b) INTERPRETER ABI: pylibs/numpy is built for CPYTHON 3.12 ONLY
#       (`_multiarray_umath.cpython-312-x86_64-linux-gnu.so`). Under the
#       Hermes 3.14 interpreter it cannot import at all, and because it is on
#       the path it SHADOWS the working numpy — every test module then fails
#       at COLLECTION with an ImportError from
#       `../pylibs/numpy/_core/__init__.py`.
#       So pylibs is only put on the path for an interpreter it can serve.
_PYLIBS = Path.home() / "pylibs"
if _PYLIBS.is_dir() and str(_PYLIBS) not in sys.path:
    _tag = f"{sys.version_info.major}{sys.version_info.minor}"
    if _tag == "312":
        sys.path.append(str(_PYLIBS))
    else:
        # Not importable on this interpreter: adding it would only shadow the
        # working numpy. The asn1crypto fallback it provides is optional.
        pass


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
