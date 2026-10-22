"""THE TREE STAMP MUST SURVIVE A CALLER'S `tail`.

2026-10-22. The tree stamp (PR #18) prints from pytest's
`pytest_terminal_summary` hook — but pytest prints its own summary line
("N passed in Xs") AFTER that hook, so the stamp lands **second-to-last**:

    ...
    vishwas tree: 241ca63
    593 passed, 2 skipped in 57.31s

**The very commands used to capture background-run notifications pipe
through `tail -4` / `tail -3`** — and a `tail` of the last few lines DROPS
the stamp. A background run reported with a count and no provenance is
exactly the situation the stamp was added to prevent.

WHY THE STAMP CANNOT SIMPLY BE MOVED LATER IN PYTEST
------------------------------------------------------
Measured empirically, pytest's hook order is:

    conftest pytest_sessionfinish  ->  pytest_terminal_summary  ->  "N passed"

and `pytest_unconfigure` also runs BEFORE the "N passed" line. So there is
no conftest hook that lands after it. (Patching
`TerminalReporter.pytest_sessionfinish` does not work either: the hook
callbacks were bound at plugin-registration time, and the reporter's method
is a GENERATOR, so a plain-function wrapper silently swallows the
"N passed" line.)

THE FIX
-------
**The split runner prints the stamp itself, after the part's output.** The
runner is what the background-run capture commands actually invoke, so
this is the layer that matters — and the stamp becomes the LAST line of the
runner's output, surviving any `tail -N`.

The pytest-level stamp stays where it is (it is still correct and useful
for a bare `pytest` invocation); the runner's own stamp is what makes the
provenance survive a capture.

These tests pin:
  1. the split runner's output ENDS with the tree stamp;
  2. the stamp survives `tail -1` and `tail -3` of the runner's output;
  3. the runner reports a DIRTY worktree as such.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_tests_split.sh"


def _runner_src() -> str:
    if not RUNNER.exists():
        import pytest
        pytest.skip("runner not present")
    return RUNNER.read_text()


def test_the_runner_prints_the_tree_stamp_itself():
    src = _runner_src()
    assert "tree_stamp" in src, (
        "the split runner does not print the tree stamp itself, so its own "
        "output carries no provenance")
    assert "vishwas tree:" in src, (
        "the runner's stamp helper does not emit the 'vishwas tree:' prefix")


def _case_arm(src: str, label: str) -> str:
    """Return the body of a `case` arm, given its label (e.g. "a").

    The arm is read as a BLOCK, from its label to its terminating `;;`.
    A one-line assumption is wrong: the single-part arms grew to a block
    when they had to propagate the part's exit code (2026-10-22), and the
    old line-based lookup silently stopped finding them.
    """
    import re
    m = re.search(rf"(?m)^\s*{re.escape(label)}\)\s*$", src)
    assert m, f"the runner no longer has the {label!r} arm"
    end = src.find(";;", m.end())
    body = src[m.end():end] if end != -1 else src[m.end():]
    # the block ends at the next top-level case label if there is no `;;`
    nxt = re.search(r"(?m)^\s{2}\S", body)
    if nxt:
        body = body[:nxt.start()]
    return body


def test_the_runner_prints_the_stamp_after_the_parts_output():
    """The stamp must come AFTER the part's output, or a `tail` of the
    runner's output would drop it.

    2026-10-22: the single-part arms grew from one line to a block, because
    they now have to PROPAGATE the part's exit code (the one-line form
    ended with `tree_stamp`, whose status is always 0 — so a failed part
    reported success). The assertion below therefore reads the ARM as a
    block, not as a single line, and checks the same two things it always
    did: the arm runs the part, and the stamp comes after it.
    """
    src = _runner_src()
    for label, part in (("a", "part_a"), ("b", "part_b")):
        body = _case_arm(src, label)
        assert part in body, (
            f"the runner's {label!r} arm does not run {part}")
        assert "tree_stamp" in body, (
            f"the runner's {label!r} arm does not print the stamp, so the "
            "provenance is not the last thing in its output")
        assert body.index("tree_stamp") > body.index(part), (
            f"the runner's {label!r} arm prints the stamp BEFORE running the "
            "part, so a `tail` of its output would show the stamp without "
            "the results it describes")


def test_the_single_part_arms_propagate_the_parts_exit_code():
    """2026-10-22. The one-line arms ended with `tree_stamp`, so the arm's
    exit status was the stamp's — always 0. A caller that ran one part and
    checked `$?` saw SUCCESS for a run that had tested nothing.

    The arm must therefore capture the part's status and exit with it.
    """
    src = _runner_src()
    for label in ("a", "b"):
        body = _case_arm(src, label)
        assert "rc_" in body, (
            f"the runner's {label!r} arm does not capture the part's exit code")
        assert "exit $rc" in body, (
            f"the runner's {label!r} arm does not exit with the part's exit "
            "code — a failed part would be reported as a successful run")


def test_the_runner_reports_a_dirty_worktree():
    src = _runner_src()
    assert "DIRTY WORKTREE" in src, (
        "the runner's stamp does not report a dirty worktree -- but a dirty "
        "tree is the case where the commit alone does not describe what ran")


def test_the_stamp_survives_a_tail_of_the_runners_output():
    """End to end: run the runner's own stamp helper and tail it."""
    if not RUNNER.exists():
        import pytest
        pytest.skip("runner not present")
    # Invoke the runner's stamp function in isolation (it is a shell
    # function, so source the script and call it).
    r = subprocess.run(
        ["bash", "-c",
         f'source "{RUNNER}" 2>/dev/null; tree_stamp'],
        cwd=str(REPO), capture_output=True, text=True, timeout=120)
    out = [l for l in (r.stdout or "").splitlines() if l.strip()]
    assert out, (
        f"the runner's stamp helper printed nothing (rc={r.returncode}, "
        f"stderr={(r.stderr or '')[-200:]!r})")
    last = out[-1]
    assert "vishwas tree:" in last, (
        f"the stamp is not the last line of the helper's output: {last!r}")
    # and it must survive a tail of the last line
    assert "vishwas tree:" in "\n".join(out[-1:])
    assert "vishwas tree:" in "\n".join(out[-3:])


def test_the_pytest_level_stamp_still_exists():
    """The runner's stamp is the capture-path fix, but the pytest-level one
    is still what a bare `pytest` invocation shows."""
    conftest = (REPO / "tests" / "conftest.py").read_text()
    assert "vishwas tree:" in conftest, (
        "the pytest-level tree stamp is gone")
    assert "pytest_terminal_summary" in conftest, (
        "the pytest-level stamp is no longer emitted in quiet mode")
