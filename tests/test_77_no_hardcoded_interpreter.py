"""NO TEST MAY HARDCODE THE INTERPRETER IT SPAWNS A SUBPROCESS WITH.

2026-10-22. `tests/test_76_gpg_no_leak.py` spawned its nested pytest as
`/usr/bin/python3 -m pytest`, and CI failed with:

    /usr/bin/python3: No module named pytest

because in CI the pytest lives in the `setup-python` interpreter, NOT in
`/usr/bin/python3`. The test passed on the developer machine and failed
in CI — the exact failure mode of a hardcoded path.

**This is the SAME defect class PR #23 fixed in the split runner**, and
it appeared AGAIN, in a test written in the same session that learned
the lesson. The lesson did not transfer.

THE RULE
---------
A test that spawns a subprocess must use `sys.executable` — the
interpreter that is running the test. By construction it is the one
that has the test's dependencies installed, on every machine.

`/usr/bin/python3` is a HOST-SPECIFIC path that happens to be right
here. It is wrong in CI, wrong in a venv, and wrong in a container.

WHAT THIS PINS
--------------
  * no test spawns `python`/`python3` by bare name or by an absolute
    path when it means "the interpreter running this test";
  * the exceptions are enumerated and justified — the ones that are
    genuinely about a DIFFERENT interpreter (the `.venv-ambient` lane,
    which is the point of those tests).

The one permitted absolute path is a STUB the test creates itself (a
fake interpreter), and that is not an interpreter at all.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
TESTS_DIR = REPO / "tests"

# Interpreters that a test may legitimately name because the test is
# ABOUT that interpreter (a different, out-of-tree environment).
ALLOWED = (
    "/home/hermes/.venv-ambient/bin/python",   # the torch/CUDA lane
)


def _test_files() -> list[pathlib.Path]:
    return sorted(p for p in TESTS_DIR.glob("test_*.py"))


def test_there_are_tests_to_check():
    files = _test_files()
    assert len(files) > 20, (
        f"only {len(files)} test files found — the glob is probably wrong, "
        "which would make every other assertion in this file vacuous")


def test_no_test_spawns_a_bare_python_for_a_subprocess():
    """A bare `python`/`python3` is whatever the PATH resolves."""
    offenders = []
    for p in _test_files():
        text = p.read_text()
        # A subprocess that runs a python interpreter. The interpreter may
        # be given directly in the argument list, OR via a helper in the
        # same file — the second form is what the GPG test uses, and a
        # scan that only looked at literal argument lists would miss it.
        for m in re.finditer(
                r"subprocess\.(?:run|Popen|check_output|call)\s*\(\s*"
                r"\[([^\]]{0,200})\]", text, re.S):
            arglist = m.group(1)
            if "python" not in arglist:
                continue
            for tok in re.findall(r"['\"]([^'\"]*python[^'\"]*)['\"]", arglist):
                if tok in ALLOWED:
                    continue
                if tok == "python" or tok == "python3":
                    offenders.append((p.name, tok, "bare name"))
                elif tok.startswith("/") and "python" in tok:
                    offenders.append((p.name, tok, "hardcoded absolute path"))

        # Now the INDIRECT form: a helper that BUILDS a command to spawn.
        # Three filters, each of which a real false positive needed:
        #   * the function must not be a TEST (a test that mentions a
        #     python path is usually monkeypatching a deliberate dead
        #     interpreter, e.g. test_59 proving the code degrades
        #     honestly);
        #   * the body must actually SPAWN (contain a subprocess call),
        #     not merely mention one;
        #   * the path must be in the ARGUMENT of that spawn.
        for m in re.finditer(
                r"def\s+(\w+)\s*\([^)]*\)[^:]*:(.{0,600}?)(?=\ndef\s|\nclass\s|\Z)",
                text, re.S):
            fname, body = m.group(1), m.group(2)
            if fname.startswith("test") or fname.startswith("_test"):
                continue
            if "subprocess." not in body:
                continue
            for tok in re.findall(r"['\"]([^'\"]*python[^'\"]*)['\"]", body):
                if tok in ALLOWED or tok == "sys.executable":
                    continue
                if tok == "python" or tok == "python3":
                    offenders.append((p.name, tok, "bare name in a helper"))
                elif tok.startswith("/") and "python" in tok:
                    offenders.append((p.name, tok,
                                      "hardcoded absolute path in a helper"))
    assert not offenders, (
        "these tests spawn a subprocess with a hardcoded interpreter; use "
        "sys.executable instead (the interpreter running the test is the "
        "one that has its dependencies installed):\n"
        + "\n".join(f"    {n}: {t} ({why})" for n, t, why in offenders))


def test_the_shipped_tests_use_sys_executable_where_they_spawn_python():
    """A positive check: the tests that DO spawn a python subprocess
    should be using sys.executable.

    This catches the case where someone "fixes" a failing CI job by
    hardcoding the CI's interpreter path instead of using sys.executable
    — which passes CI and breaks the next environment.
    """
    spawners = []
    for p in _test_files():
        text = p.read_text()
        if "subprocess" not in text:
            continue
        if "sys.executable" in text:
            continue
        # spawns a python interpreter but never mentions sys.executable
        if re.search(r"subprocess\.(?:run|Popen|check_output|call)\s*\(\s*"
                     r"\[[^\]]*python", text, re.S):
            spawners.append(p.name)
    assert not spawners, (
        "these tests spawn a python subprocess but never use sys.executable: "
        f"{spawners}")


def test_the_gpg_leak_test_uses_sys_executable():
    """The specific regression, pinned by name."""
    p = TESTS_DIR / "test_76_gpg_no_leak.py"
    if not p.exists():
        import pytest
        pytest.skip("the GPG leak test is not present")
    text = p.read_text()
    assert "sys.executable" in text, (
        "the GPG leak test does not use sys.executable for its nested "
        "pytest — it will fail in CI, where pytest is not in "
        "/usr/bin/python3")
    assert not re.search(r"/usr/bin/python3['\"]", text), (
        "the GPG leak test still hardcodes /usr/bin/python3 for its nested "
        "pytest — that is the exact path that failed in CI")
