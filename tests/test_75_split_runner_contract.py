"""THE SPLIT RUNNER MUST PIN ITS INTERPRETER AND PROPAGATE ITS EXIT CODE.

2026-10-22. Two defects, both found by RUNNING the runner (not by reading
it) — the runner was the thing that had been reporting "green" all along.

DEFECT 1 — IT INVOKED A BARE `python`
--------------------------------------
    $ bash scripts/run_tests_split.sh a
    --- PART A (exit 1) ---
    Running as unit: run-....scope; invocation ID: ...
    /home/hermes/.hermes/installs/<id>/environments/<id>/venv/bin/python:
        No module named pytest

On this host `python` resolves to the Hermes 3.14 venv, which has no pytest.
So the runner produced **no test results at all** — the "PART A (exit 1)"
line was pytest's own error, not a test outcome. The runner was reporting a
failure, but a caller could not tell it apart from a test failure.

The interpreter must be pinned explicitly, and it must be one that can
actually import pytest.

DEFECT 2 — SINGLE-PART MODE SWALLOWED THE FAILURE
--------------------------------------------------
    $ bash scripts/run_tests_split.sh a; echo $?
    ... No module named pytest
    0

The `a)` / `b)` arms ended with `tree_stamp`, so the arm's exit status was
the stamp's (always 0) — the part's exit code was discarded. A caller that
ran one part and checked `$?` saw SUCCESS for a run that had tested
nothing.

Note the `both` arm was correct: it captured `rc_a` / `rc_b` explicitly.
The defect was in the two single-part arms only.

WHAT THESE TESTS PIN
--------------------
  1. the runner never invokes a bare `python`;
  2. it can REPORT the interpreter it resolved, and that interpreter can
     import pytest;
  3. a part that fails makes the runner exit NON-ZERO — including in
     single-part mode;
  4. the tree stamp is still the LAST line (PR #21's contract) even when
     the part failed.

The failure is injected with a stub interpreter that passes the runner's
"can it import pytest?" probe but exits non-zero when asked to RUN pytest.
That is deterministic and costs no wall time.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_tests_split.sh"

# A stub interpreter: it answers the runner's "can you import pytest?" probe
# with success, and exits 7 when asked to RUN pytest. That separates "the
# runner could not start pytest" from "pytest ran and failed".
STUB = """#!/usr/bin/python3
import sys
if '-m' in sys.argv:
    sys.exit(7)
sys.exit(0)
"""


def _runner_src() -> str:
    if not RUNNER.exists():
        import pytest
        pytest.skip("runner not present")
    return RUNNER.read_text()


def _run(args: list[str], env_extra: dict[str, str] | None = None,
         timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("VISHWAS_TEST_PYTHON", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", str(RUNNER)] + args,
        cwd=str(REPO), capture_output=True, text=True,
        timeout=timeout, env=env)


def _stub(tmp_path: pathlib.Path) -> str:
    p = tmp_path / "stub_python"
    p.write_text(STUB)
    p.chmod(0o755)
    return str(p)


# --- 1. no bare `python` ----------------------------------------------------

def test_the_runner_never_invokes_a_bare_python():
    """A bare `python` is whatever the PATH happens to resolve — on this host
    that is a Hermes venv with no pytest."""
    src = _runner_src()
    bad = re.findall(r"(?m)^\s*(?:timeout\s+[^|;&]*?)?\bpython\s+-m\s+pytest",
                     src)
    assert not bad, (
        "the split runner still invokes a bare `python`; it must use an "
        f"explicitly resolved interpreter (found: {bad!r})")


def test_the_runner_documents_why_the_interpreter_is_pinned():
    src = _runner_src()
    assert "VISHWAS_TEST_PYTHON" in src, (
        "the runner offers no interpreter override, so it cannot be pinned "
        "to an environment that actually has pytest")
    assert re.search(r"pytest", src), (
        "the runner does not mention pytest at all — the very thing whose "
        "absence broke it")


# --- 2. it can report the interpreter it resolved ---------------------------

def test_the_runner_reports_the_interpreter_it_resolved():
    """`run_tests_split.sh python` must print an absolute path to an
    interpreter that can import pytest (or say NONE, loudly)."""
    r = _run(["python"], timeout=180)
    out = [l for l in (r.stdout or "").splitlines() if l.strip()]
    assert out, (
        f"the runner printed no interpreter (rc={r.returncode}, "
        f"stderr={(r.stderr or '')[-300:]!r})")
    first = out[0].strip()
    assert first != "NONE" or len(out) > 1, (
        "the runner reported NONE without saying why")
    if first == "NONE":
        return  # no interpreter on this host — a loud, explicit failure
    assert first.startswith("/"), (
        f"the resolved interpreter is not an absolute path: {first!r} — "
        "a relative/bare name is PATH-dependent, which is the defect")
    probe = subprocess.run([first, "-c", "import pytest"],
                           capture_output=True, text=True, timeout=120)
    assert probe.returncode == 0, (
        f"the runner resolved {first!r}, but that interpreter cannot import "
        f"pytest: {probe.stderr[-200:]!r}")


# --- 3. a failing part must make the runner exit non-zero ------------------

def test_a_failing_part_makes_the_runner_exit_non_zero(tmp_path):
    """Single-part mode must NOT swallow the part's exit code."""
    stub = _stub(tmp_path)
    r = _run(["a"], {"VISHWAS_TEST_PYTHON": stub}, timeout=180)
    assert r.returncode != 0, (
        "run_tests_split.sh a exited 0 even though the part failed "
        f"(stdout={(r.stdout or '')[-400:]!r})")


def test_a_failing_part_in_both_mode_makes_the_runner_exit_non_zero(tmp_path):
    stub = _stub(tmp_path)
    r = _run([], {"VISHWAS_TEST_PYTHON": stub}, timeout=300)
    assert r.returncode != 0, (
        "run_tests_split.sh (both) exited 0 even though the parts failed "
        f"(stdout={(r.stdout or '')[-400:]!r})")


# --- 4. the stamp contract still holds on failure --------------------------

def test_the_tree_stamp_is_still_the_last_line_when_a_part_fails(tmp_path):
    """PR #21's contract: the stamp must survive a caller's `tail`. A
    FAILED run is exactly when a caller pipes the output."""
    stub = _stub(tmp_path)
    r = _run(["a"], {"VISHWAS_TEST_PYTHON": stub}, timeout=180)
    out = [l for l in (r.stdout or "").splitlines() if l.strip()]
    assert out, "the runner printed nothing at all"
    assert "vishwas tree:" in out[-1], (
        "the tree stamp is not the last line of a FAILED run: "
        f"last={out[-1]!r}")
    assert "vishwas tree:" in "\n".join(out[-3:]), (
        "the tree stamp does not survive a `tail -3` of a failed run")


def test_the_runner_explains_a_missing_interpreter_instead_of_a_pytest_error():
    """When it cannot find a usable interpreter the runner must say so ITSELF.
    The defect's symptom was a pytest error message that read like a test
    failure."""
    src = _runner_src()
    assert re.search(r"FATAL", src), (
        "the runner has no loud failure path for 'no usable interpreter'")
