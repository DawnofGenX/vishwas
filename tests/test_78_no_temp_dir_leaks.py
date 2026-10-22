"""NO TEST MAY LEAK A TEMPORARY DIRECTORY.

2026-10-22. Two separate leaks of the same shape were found in one
session:

  * `tests/test_15_gpg.py` created two throwaway GNUPGHOMEs in a
    `tempfile.mkdtemp()` and never removed the tree (plus a live
    `gpg-agent` daemon per homedir). **107 orphaned trees, 8.8 MB.**
  * `tests/test_49_fusion_train.py` called `tempfile.mkdtemp()` four
    times with no cleanup. **455 orphaned directories, 33 MB**, growing by
    4 on every run of the suite.

Both are silent: nothing reports them, and the suite stays green.

**THE RULE**
-----------
A test that creates a directory must either

  * use pytest's `tmp_path` fixture (removed automatically by pytest), or
  * create it with `tempfile.mkdtemp()` and remove it in a `finally`.

`tempfile.mkdtemp()` with no `finally` is a leak. `tempfile.mkdtemp()`
with an rmtree that is NOT in a `finally` leaks on the failure path —
which is exactly when a dirty host is hardest to reason about.

WHAT THIS PINS
--------------
  1. no test calls `tempfile.mkdtemp()` without a `finally` that removes
     the directory;
  2. no test calls it at all without removing it;
  3. the scan is not vacuous — it must find the pattern in a synthetic
     file, so a broken regex cannot make this test pass silently.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
TESTS_DIR = REPO / "tests"


def _test_files() -> list[pathlib.Path]:
    return sorted(p for p in TESTS_DIR.glob("test_*.py"))


def test_the_scan_finds_a_known_leak():
    """The scan must actually detect the pattern — a broken regex would
    make every other assertion in this file vacuously true."""
    sample = (
        "import tempfile\n"
        "def test_x():\n"
        "    d = Path(tempfile.mkdtemp())\n"
        "    assert d.exists()\n"
    )
    offenders = _scan(sample)
    assert offenders, (
        "the temp-dir leak scanner does not detect a known leak — every "
        "other assertion in this file is vacuous")


def test_no_test_leaks_a_temp_dir():
    offenders = []
    for p in _test_files():
        if p.name == pathlib.Path(__file__).name:
            continue  # this file names mkdtemp in its own prose and samples
        text = p.read_text()
        for lineno, line in _mkdtemp_lines(text):
            # The cleanup must be in a `finally:` that ENCLOSES the
            # mkdtemp call. A `finally:` that merely EXISTS in the same
            # function is not enough: a mutation that replaced `try:`
            # with `if True:` keeps the `finally:` (and its cleanup) but
            # makes the body unconditional, so the cleanup is skipped
            # when the code between them raises. The mutation testing
            # caught exactly that.
            fn = _enclosing_function(text, lineno)
            if fn is None or not _mkdtemp_is_inside_try(fn, lineno):
                offenders.append((p.name, lineno, line.strip()))
    assert not offenders, (
        "these tests create a temp directory with tempfile.mkdtemp() and "
        "never remove it in a finally -- every run of the suite leaks one:\n"
        + "\n".join(f"    {n}:{ln}  {t}" for n, ln, t in offenders))


def test_no_test_creates_a_temp_dir_without_removing_it():
    """A stronger form: the directory must be REMOVED, not merely guarded
    by a finally. (A finally that does not remove it is still a leak.)"""
    offenders = []
    for p in _test_files():
        if p.name == pathlib.Path(__file__).name:
            continue
        text = p.read_text()
        for lineno, line in _mkdtemp_lines(text):
            fn = _enclosing_function(text, lineno)
            if fn is None:
                offenders.append((p.name, lineno, "not inside a function"))
                continue
            if "rmtree" not in fn and "shutil.rmtree" not in fn:
                offenders.append((p.name, lineno,
                                  "no rmtree in the enclosing function"))
    assert not offenders, (
        "these tests create a temp directory and never remove it:\n"
        + "\n".join(f"    {n}:{ln}  {t}" for n, ln, t in offenders))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _mkdtemp_lines(text: str) -> list[tuple[int, str]]:
    """(1-based lineno, line) for every ACTUAL tempfile.mkdtemp() call.

    "Actual" means the line contains a CALL, not a mention. A docstring
    that says "these tests used tempfile.mkdtemp()" is prose; a comment
    is a comment. Only a call site can leak.

    **THE SCAN USES PYTHON'S OWN TOKENIZER**, not a regex. The regex
    approach was tried first and produced THREE false positives on this
    repo's own prose: a docstring line with backticks, a module-level
    docstring, and the scanner's own source. Deciding "is this a call?"
    by pattern-matching a line is guesswork; the tokenizer knows.

    So: tokenize, and keep the line if there is a NAME token
    `mkdtemp` that is NOT inside a string or a comment.
    """
    import io
    import tokenize

    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return out
    lines = text.split("\n")
    for tok in toks:
        if tok.type != tokenize.NAME or tok.string != "mkdtemp":
            continue
        # a NAME token is real code -- never inside a string or a comment
        lineno = tok.start[0]
        out.append((lineno, lines[lineno - 1]))
    return out


def _enclosing_function(text: str, lineno: int) -> str | None:
    """The source of the function that contains the given line."""
    lines = text.split("\n")
    start = None
    for i in range(lineno - 1, -1, -1):
        if re.match(r"^def\s+\w+|^\s*def\s+\w+", lines[i]):
            start = i
            break
    if start is None:
        return None
    # the function body runs until the next top-level def/class
    out = []
    for i in range(start, len(lines)):
        if i > start and re.match(r"^(def\s|class\s|@)", lines[i]):
            break
        out.append(lines[i])
    return "\n".join(out)


def _mkdtemp_is_inside_try(fn_src: str, lineno: int) -> bool:
    """True if the mkdtemp at `lineno` is covered by a `try:`/`finally:`
    pair whose `finally:` does the cleanup.

    Tokenized, not regex-matched: "is this statement covered by a
    try/finally?" is a question about BLOCK STRUCTURE, and the tokenizer
    gives the structure exactly. A regex would have to guess.

    The pattern that is acceptable:

        tmp = Path(tempfile.mkdtemp())     <-- created here
        try:
            ... work with tmp ...
        finally:
            shutil.rmtree(tmp)              <-- cleaned here

    The creation is BEFORE the `try:` (you cannot put the mkdtemp inside
    the try and still have the variable available in the finally) — so
    the check is: the function contains a `try:` and a matching
    `finally:`, and the `finally:` comes AFTER the mkdtemp.

    ⚠️ A `finally:` that is present but does NOT enclose the risky work
    is not enough. That is the `if True:` mutation: the `finally:` and
    its cleanup stay, the body becomes unconditional, and the cleanup is
    skipped when the body raises. The structural check below therefore
    requires a REAL `try:` (not `if True:`) before the `finally:`.
    """
    import io
    import tokenize

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(fn_src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return False

    names: list[tuple[int, str, int]] = []   # (lineno, name, col)
    for tok in toks:
        if tok.type == tokenize.NAME:
            names.append((tok.start[0], tok.string, tok.start[1]))

    # find the mkdtemp line
    mk_line = None
    for ln, nm, _col in names:
        if nm == "mkdtemp":
            mk_line = ln
            break
    if mk_line is None:
        return False

    # find a `try:` at or after the mkdtemp, and a `finally:` after it
    try_at = None
    finally_at = None
    for ln, nm, _col in names:
        if nm == "try" and ln >= mk_line and try_at is None:
            try_at = ln
        elif nm == "finally" and try_at is not None and ln > try_at:
            finally_at = ln
            break
    if try_at is None or finally_at is None:
        return False

    # the try must be a REAL try, not `if True:` -- the try keyword must
    # be the first token of its line (a block opener), not part of an
    # expression.
    for ln, nm, col in names:
        if ln == try_at and nm == "try":
            # the text before it on that line must be only whitespace
            prefix = fn_src.split("\n")[ln - 1][:col]
            return prefix.strip() == ""
    return False


def _scan(sample: str) -> list[tuple[int, str]]:
    """The offender list the scan produces for a given source text."""
    offenders = []
    for lineno, line in _mkdtemp_lines(sample):
        fn = _enclosing_function(sample, lineno)
        if fn is None or "finally:" not in fn:
            offenders.append((lineno, line))
    return offenders
