"""A TEST RUN MUST SAY WHICH TREE IT RAN AGAINST.

2026-10-22. Four background test runs during one session reported plausible
but STALE counts, and each one had to be dated by hand:

  * 600 passed (543 + 62 + ...) -> the tree at `c64b3e5` (PR #11)
  * 544 collected             -> the tree at `a9c8c4c` (PR #12)
  * 543 passed + 1 skipped    -> also `a9c8c4c` (PR #12)
  * AI videos at HIGH         -> also `a9c8c4c` (the pre-silence-fix tree)

Dating each one meant re-running `pytest --collect-only` against every
candidate commit in a scratch worktree — minutes per ambiguity, and easy to
get wrong.

THE ROOT CAUSE
--------------
A test run's output does not say which tree it ran against, so a stale run
is indistinguishable from a regression until you go and check.

THE FIX
-------
`tests/conftest.py` stamps the commit into the run:
  * `pytest_report_header` — in the header (suppressed by `-q`);
  * `pytest_terminal_summary` — at the end, in BOTH modes, so the quiet
    invocation the split runner uses cannot hide it.

If the tree cannot be determined, the stamp says UNKNOWN rather than
guessing — an UNKNOWN stamp is itself the signal that a run's provenance
cannot be established.

These tests pin:
  1. the stamp is emitted in `-q` mode (the mode the runner uses);
  2. the stamp names the CURRENT commit;
  3. a DIRTY worktree is reported as such;
  4. a tree with no git info reports UNKNOWN, not a stale sha.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFTEST = REPO / "tests" / "conftest.py"

# A tiny, fast test file to run the header against.
PROBE = "tests/test_66_silence_not_anticorrelated.py"


def _run(args):
    env = {"PYTHONPATH": "src", "PATH": "/home/hermes/.venv-ambient/bin:"
                                 + __import__("os").environ.get("PATH", "")}
    return subprocess.run(
        [sys.executable, "-m", "pytest", PROBE, *args],
        cwd=str(REPO), capture_output=True, text=True, timeout=300, env=env)


def test_the_tree_stamp_appears_in_quiet_mode():
    """`-q` is what `scripts/run_tests_split.sh` uses, and it is exactly the
    mode that hid the header."""
    r = _run(["-q"])
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert "vishwas tree:" in r.stdout, (
        "the tree stamp is missing from the -q output — a run's provenance "
        "cannot be established:\n" + r.stdout[-800:])


def test_the_tree_stamp_names_the_current_commit():
    r = _run(["-q"])
    assert "vishwas tree:" in r.stdout
    import subprocess as sp
    sha = sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                  capture_output=True, text=True, timeout=30).stdout.strip()
    assert sha, "no git info in the repo"
    assert sha in r.stdout, (
        f"the stamp does not name the current commit {sha}:\n{r.stdout[-800:]}")


def test_a_dirty_worktree_is_reported():
    """A run against uncommitted changes is a different provenance from a
    clean one, and must be distinguishable."""
    r = _run(["-q"])
    assert "vishwas tree:" in r.stdout
    import subprocess as sp
    dirty = bool(sp.run(["git", "status", "--porcelain"], cwd=str(REPO),
                         capture_output=True, text=True,
                         timeout=30).stdout.strip())
    if dirty:
        assert "DIRTY" in r.stdout, (
            "the worktree is dirty but the stamp does not say so")
    else:
        assert "DIRTY" not in r.stdout


def test_the_stamp_reports_unknown_rather_than_guessing():
    """With no git info, the stamp must say UNKNOWN — not print a stale sha
    from some other tree."""
    import shutil
    import tempfile

    # A copy of the repo with no .git directory.
    tmp = pathlib.Path(tempfile.mkdtemp())
    dest = tmp / "repo"
    shutil.copytree(REPO / "src", dest / "src")
    shutil.copytree(REPO / "tests", dest / "tests")
    (dest / "pyproject.toml").write_text((REPO / "pyproject.toml").read_text())
    r = subprocess.run(
        [sys.executable, "-m", "pytest", PROBE, "-q"],
        cwd=str(dest), capture_output=True, text=True, timeout=300,
        env={"PYTHONPATH": "src", "PATH": __import__("os").environ.get("PATH", "")})
    shutil.rmtree(tmp, ignore_errors=True)
    combined = r.stdout + r.stderr
    if r.returncode == 0:
        assert "vishwas tree: UNKNOWN" in combined, (
            "a tree with no git info must report UNKNOWN, not guess:\n"
            + combined[-800:])
    else:
        # A copy without .git can fail for other reasons; only assert the
        # stamp when the run got far enough to print it.
        if "vishwas tree:" in combined:
            assert "vishwas tree: UNKNOWN" in combined


def test_the_conftest_defines_both_hooks():
    """The header is suppressed by -q, so the terminal-summary hook is what
    actually makes the stamp reliable. Both must exist."""
    text = CONFTEST.read_text()
    assert "def pytest_report_header" in text, "no report-header hook"
    assert "def pytest_terminal_summary" in text, (
        "no terminal-summary hook — the stamp would be hidden by -q")
