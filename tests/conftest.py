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
#
# WARNING (2026-10-21) — THREE hazards, all now real:
#   (a) SERVING STACK: putting pylibs on the path is NOT safe for the serving
#       path. The pylibs tree ships cryptography v50, which no longer provides
#       the `lib` submodule pyOpenSSL imports; Python then resolves `lib` to a
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
#   (c) ORDER (fixed 2026-10-21): APPENDING pylibs put the interpreter's own
#       numpy 2.4.6 ahead of pylibs' numpy 2.5.2, and then `import cv2` blew
#       up with "numpy.core.multiarray failed to import" — because the
#       INTERPRETER's cv2 is compiled against numpy 1.x's C API while its own
#       numpy is 2.4.6. pylibs' cv2 5.0.0 and pylibs' numpy 2.5.2 are a
#       MATCHED pair, so pylibs must LEAD the path on a 3.12 interpreter.
#       Measured: pylibs first -> numpy 2.5.2 + cv2 5.0.0 + torch 2.4.1 all
#       import cleanly; pylibs appended -> cv2 raises ImportError.
_PYLIBS = Path.home() / "pylibs"
if _PYLIBS.is_dir() and str(_PYLIBS) not in sys.path:
    _tag = f"{sys.version_info.major}{sys.version_info.minor}"
    if _tag == "312":
        # PREPEND: pylibs' numpy and cv2 are a matched pair and must win over
        # the interpreter's mismatched numpy/cv2 combination.
        sys.path.insert(0, str(_PYLIBS))
    else:
        # Not importable on this interpreter: adding it would only shadow the
        # working numpy. The asn1crypto fallback it provides is optional.
        pass


import pytest


def _vishwas_tree_stamp() -> str:
    """Return a one-line stamp of the tree the tests are running against.

    "vishwas tree: <sha>" / "vishwas tree: <sha> (DIRTY WORKTREE)" /
    "vishwas tree: UNKNOWN (no git info — a run's provenance CANNOT be
    established)".
    """
    import subprocess
    repo = str(Path(__file__).resolve().parents[1])
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if not sha:
            return ("vishwas tree: UNKNOWN (no git info — a run's provenance "
                    "CANNOT be established)")
        dirty = ""
        d = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True, timeout=10)
        if d.stdout.strip():
            dirty = " (DIRTY WORKTREE)"
        return f"vishwas tree: {sha}{dirty}"
    except Exception:
        return "vishwas tree: UNKNOWN (git not available)"


def pytest_report_header(config):
    """Stamp the COMMIT into every test run's header.

    2026-10-22. Four background test runs during one session reported
    plausible but STALE counts (600, 544, 543+62, 0.7789-vs-0.7711), and
    every one had to be dated by hand — usually by re-running
    `pytest --collect-only` against each candidate commit in a scratch
    worktree, which costs minutes per ambiguity.

    The root cause is that a test run's output does not say which tree it
    ran against, so a stale run is indistinguishable from a regression until
    you go and check. This header makes every run self-dating.

    NOTE: `-q` suppresses the header, so the same stamp is ALSO written by
    `pytest_terminal_summary` below, which prints in both modes.
    """
    return _vishwas_tree_stamp()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Repeat the tree stamp where `-q` cannot hide it."""
    terminalreporter.write_line(_vishwas_tree_stamp())


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
