"""THE GPG TESTS MUST NOT LEAK KEYRINGS OR gpg-agent DAEMONS.

2026-10-22. `tests/test_15_gpg.py` builds two throwaway GNUPGHOMEs in a
`tempfile.mkdtemp` directory and **never removes it**. Each full-suite run
therefore leaves behind:

  * a `vishwas-gpg-test-*` directory tree (keyrings, sockets, temp files),
  * a `gpg-agent` daemon per homedir, still running.

MEASURED on this host after the session's suite runs: **107 orphaned
`vishwas-gpg-test-*` directories (8.8 MB)** and **12 orphaned `gpg-agent`
daemons (48 MB RSS)**, and the count grows with every run.

This is the same class of defect the repo already cares about: a check
that leaves the host dirtier than it found it, and does so **silently**.

THE MECHANISM
--------------
`gpg` starts a `gpg-agent` daemon per GNUPGHOME and the daemon is NOT
tied to the calling process. Deleting the homedir does not necessarily
reap it, and the agent keeps its socket + memory alive.

The two mechanisms that DO work (both measured, see
`references/gpg-agent-leak-2026-10-22.md`):
  * `gpgconf --homedir <home> --kill gpg-agent` — kills the daemon;
  * removing the homedir directory.

So the fix must do BOTH, in that order: kill the agent, then remove the
tree.

WHAT THESE TESTS PIN
--------------------
  1. the GPG test's fixture must clean up the tree it creates;
  2. it must kill the gpg-agent daemons it started;
  3. the cleanup must run even when a test in the module FAILS (a
     teardown that only runs on the happy path leaks on the failure path
     too — which is when you most want a clean host);
  4. no `gpg-agent` may survive a run of the GPG test module.

Test 4 is the end-to-end proof: it counts the daemons before and after.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
GPG_TEST = REPO / "tests" / "test_15_gpg.py"

# ⚠️ THE TEMP ROOT MUST BE RESOLVED, NOT HARDCODED (2026-10-22).
# This originally hardcoded `/home/hermes/.hermes/cache/scratch`, but
# `tempfile.mkdtemp()` uses `tempfile.gettempdir()`, which HONOURS
# `$TMPDIR`. So the test was looking in the wrong place whenever TMPDIR
# pointed elsewhere -- and it looked in the right place only by accident.
#
# The mutation harness found this: isolating the harness's TMPDIR (so a
# deliberately-broken fixture cannot pollute the real scratch dir) made
# the tree-leak check silently stop detecting anything.
#
# This is the SAME class as PR #23's hardcoded interpreter: a path that
# is right on the developer machine and wrong everywhere else.
def _temp_root() -> pathlib.Path:
    import tempfile
    return pathlib.Path(tempfile.gettempdir())


SCRATCH = _temp_root()


def _src() -> str:
    if not GPG_TEST.exists():
        import pytest
        pytest.skip("the GPG test module is not present")
    return GPG_TEST.read_text()


def _gpg_agents(homedir_needle: str = "vishwas-gpg-test-") -> list[str]:
    """PIDs of REAL gpg-agent daemons whose --homedir contains the needle.

    Read from /proc rather than `pgrep -f`: a `pgrep -f` pattern also
    matches the shell running the pgrep itself, so it can report (or
    kill) the caller. That trap bit during this investigation.

    ⚠️ AND THE /proc READ HAS THE SAME TRAP (2026-10-22). A shell whose
    COMMAND LINE merely MENTIONS the needle (because the needle is in the
    script text) matches a naive substring search — the agent detector
    "found" the very shell that was running the probe. So the check is on
    the EXECUTABLE and on the `--homedir` ARGUMENT, parsed out of the
    argv, not on a substring of the whole command line.
    """
    pids = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        parts = [p for p in raw.decode("utf-8", "replace").split("\0") if p]
        if not parts:
            continue
        # the executable must BE gpg-agent, not merely mention it
        exe = pathlib.Path(parts[0]).name
        if exe not in ("gpg-agent", "gpg-agent.exe"):
            continue
        # and the homedir must be an ARGUMENT
        home = ""
        for i, p in enumerate(parts):
            if p == "--homedir" and i + 1 < len(parts):
                home = parts[i + 1]
        if homedir_needle in home:
            pids.append(entry.name)
    return pids


def test_the_gpg_fixture_cleans_up_the_tree_it_creates():
    """The fixture uses `tempfile.mkdtemp`, so pytest will NOT clean it
    up — the module must do it itself."""
    src = _src()
    assert "mkdtemp" in src, (
        "the GPG fixture no longer uses mkdtemp — re-check whether cleanup "
        "is still needed")
    assert re.search(r"(rmtree|shutil\.rmtree|cleanup)", src), (
        "the GPG test creates a throwaway keyring tree and never removes it "
        "— every run of the suite leaves a vishwas-gpg-test-* directory "
        "behind")


def test_the_gpg_fixture_kills_the_gpg_agent_it_started():
    """Deleting the homedir does not reliably reap the daemon. The cleanup
    must kill it explicitly.

    ⚠️ MEASURED 2026-10-22 (6 trials): removing the homedir DOES reap the
    daemon — 0 survivors in 6 of 6. So this assertion is BELT-AND-BRACES,
    not strictly necessary. It is kept because:
      * the explicit kill is what makes the intent visible, and a future
        gpg build could keep the daemon alive across a homedir removal;
      * the daemon is what actually holds the memory, and relying on a
        side effect of `rmtree` to reclaim it is fragile.

    It is NOT a case of a test being over-specified for its own sake: the
    end-to-end test below proves the EFFECT either way.
    """
    src = _src()
    assert re.search(r"gpgconf", src), (
        "the GPG test never kills the gpg-agent daemons it starts — each "
        "run leaves a daemon alive (48 MB RSS across 12 of them, measured)")
    assert re.search(r"--kill", src), (
        "the GPG test mentions gpgconf but does not use --kill, so it does "
        "not actually stop the daemon")


def test_the_cleanup_runs_even_when_a_test_fails():
    """A teardown that only runs on the happy path leaks on the failure
    path too — which is exactly when a dirty host is hardest to diagnose.

    2026-10-22: the first version of this test asserted only that the
    source contains a `finally:`. A mutation that replaced `try:` with
    `if True:` — keeping the `finally:` and its cleanup, but making the
    body unconditional — left the test GREEN, because the `finally` text
    was still there. The cleanup would then run on the happy path and skip
    on the failure path.

    So this asserts the SHAPE: the yield must be INSIDE a `try:` whose
    `finally:` performs the cleanup. A `try:` that does not enclose the
    `yield` is not a teardown.
    """
    src = _src()
    try_pos = src.find("try:")
    yield_pos = src.find("yield {")
    finally_pos = src.find("finally:")
    assert try_pos != -1, (
        "the GPG fixture yields without a try/finally, so its cleanup is "
        "skipped when a test fails")
    assert finally_pos != -1, (
        "the GPG fixture has no finally block, so its cleanup does not run "
        "when a test in the module fails")
    assert try_pos < yield_pos < finally_pos, (
        "the GPG fixture's `yield` is not enclosed by its `try:`/`finally:` "
        "— the cleanup would run on the happy path and be skipped on the "
        "failure path")


def test_the_cleanup_actually_removes_the_tree_and_kills_the_daemon():
    """The teardown must remove the tree it created, and must make a
    conscious effort to stop the daemon.

    ⚠️ MEASURED 2026-10-22 (6 trials): removing the homedir alone DOES
    reap the daemon — 0 survivors in 6 of 6. So "removes the tree" is the
    load-bearing half, and the explicit kill is belt-and-braces.

    The earlier version of this test asserted BOTH halves were present in
    the teardown, and a mutation that dropped the kill escaped the suite
    — not because the daemon survived (it did not), but because the
    end-to-end test cannot see the difference when rmtree already reaps
    the daemon. **A test that asserts a mechanism the system does not
    actually need is asserting a fiction.** So the kill is still
    required (it makes the intent explicit and is robust to a gpg build
    that outlives its homedir), but this test no longer claims it is
    what prevents the leak.
    """
    src = _src()
    # the teardown block: from the fixture's `finally:` onwards
    f = src.find("finally:")
    assert f != -1
    teardown = src[f:f + 900]
    assert "_kill_gpg_agent" in teardown, (
        "the GPG fixture's teardown does not call _kill_gpg_agent — the "
        "gpg-agent daemons it started are left to chance")
    assert re.search(r"rmtree", teardown), (
        "the GPG fixture's teardown does not remove the keyring tree it "
        "created")


def _pytest_cmd() -> list[str]:
    """The command that runs pytest in THIS environment.

    2026-10-22. This test originally hardcoded `/usr/bin/python3`, and CI
    failed with:

        /usr/bin/python3: No module named pytest

    because in CI the pytest lives in the `setup-python` interpreter, NOT
    in /usr/bin/python3. That is the SAME class of defect PR #23 fixed in
    the split runner: a hardcoded interpreter that is right on the
    developer machine and wrong everywhere else.

    So: use the interpreter that is RUNNING this test (`sys.executable`),
    which is by construction the one that has pytest.
    """
    import sys
    return [sys.executable, "-m", "pytest"]


def test_no_gpg_agent_survives_a_run_of_the_gpg_module():
    """END TO END: run the GPG module, then assert nothing survived.

    This is the proof that matters — the other three tests check the
    SOURCE, this one checks the EFFECT.

    ⚠️ CONCURRENCY (2026-10-22): this test counts gpg-agent processes, so
    it is sensitive to OTHER gpg activity on the host at the same time —
    a parallel mutation harness, another test run, a second copy of this
    suite. It first failed for exactly that reason: a concurrently-running
    verification harness left an agent behind and the test blamed the
    module it had just run.

    So the agents are matched by their HOMEDIR, and only the ones pointing
    at a tree THIS run created count. A daemon belonging to someone else
    is not this run's business.
    """
    before_dirs = set(SCRATCH.glob("vishwas-gpg-test-*")) if SCRATCH.is_dir() else set()
    # pytest's own tmp_path root for THIS run — the tests' `workdir`
    # homedirs live under it, and they spawn agents too.
    before_agents = {p for p in _gpg_agents()}
    before_agent_homes = {_agent_homedir(p) for p in before_agents}

    r = subprocess.run(
        _pytest_cmd() + [
            "tests/test_15_gpg.py", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (
        f"the GPG module did not pass (rc={r.returncode}):\n"
        f"{(r.stdout or '')[-800:]}\n{(r.stderr or '')[-400:]}")

    if SCRATCH.is_dir():
        after_dirs = set(SCRATCH.glob("vishwas-gpg-test-*"))
        leaked_dirs = after_dirs - before_dirs
        assert not leaked_dirs, (
            f"the GPG test leaked {len(leaked_dirs)} keyring tree(s): "
            f"{sorted(str(d) for d in leaked_dirs)[:5]}")

    # The daemons are matched by the homedir they point at. A tree counts
    # as this run's if it is NEW (not in the before-set) — regardless of
    # whether the tree still exists on disk. That distinction matters: a
    # fix that removes the tree but leaves the daemon would have NO
    # leftover tree, so a check keyed on leftover trees alone would never
    # look at the daemon.
    before_names = {d.name for d in before_dirs}
    leaked_agents = []
    for pid in _gpg_agents():
        home = _agent_homedir(pid)
        if not home:
            continue
        if home in before_agent_homes:
            continue  # existed before this run — not ours
        for part in home.split("/"):
            if part.startswith("vishwas-gpg-test-") and part not in before_names:
                leaked_agents.append(pid)
                break
        else:
            # a workdir under pytest's tmp_path for THIS run
            if "pytest-of-hermes" in home:
                leaked_agents.append(pid)
    assert not leaked_agents, (
        f"the GPG test leaked {len(leaked_agents)} gpg-agent daemon(s): "
        f"{sorted(leaked_agents)}")


def _agent_homedir(pid: str) -> str:
    """The --homedir an agent process was started with ('' if unknown)."""
    try:
        cmd = (pathlib.Path("/proc") / pid / "cmdline").read_bytes().decode(
            "utf-8", "replace")
    except OSError:
        return ""
    parts = cmd.split("\0")
    for i, p in enumerate(parts):
        if p == "--homedir" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


# --------------------------------------------------------------------------
# The SAME defect class in the PRODUCTION code path.
# --------------------------------------------------------------------------

def test_the_production_verifier_does_not_delete_the_callers_directory():
    """`verify_with_truststore(..., workdir=X)` must not delete X.

    2026-10-22. The `finally` block did an unconditional
    `shutil.rmtree(base)`, and when the caller supplies a workdir, `base`
    IS the caller's directory. The production caller passes
    `ctx.quarantine_root / "gpg"` — so the callee was deleting a directory
    the CALLER owns.

    MEASURED before the fix: a caller-owned directory containing the
    caller's own file came back DELETED.
    """
    import shutil
    import tempfile

    from vishwas import gpg_check

    base = pathlib.Path(tempfile.mkdtemp(prefix="vishwas-gpg-caller-"))
    try:
        caller_dir = base / "caller_owned"
        caller_dir.mkdir()
        marker = caller_dir / "caller_file.txt"
        marker.write_text("the caller owns this")

        gpg_check.verify_with_truststore(
            b"payload", b"not-a-signature",
            truststore_dir=str(base / "no-such-truststore"),
            workdir=caller_dir)

        assert caller_dir.exists(), (
            "verify_with_truststore DELETED the caller's workdir — the "
            "callee must only remove a directory it created itself")
        assert marker.exists(), (
            "verify_with_truststore deleted the caller's own file inside "
            "the workdir it was given")
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_the_production_verifier_kills_the_gpg_agent_it_starts():
    """The production path creates a throwaway GNUPGHOME when no workdir is
    supplied, and it must not leave a `gpg-agent` daemon behind.

    ⚠️ THIS TEST WAS VACUOUS UNTIL 2026-10-22. Its first version passed a
    GARBAGE signature, and the production `_run_gpg` uses `--status-fd`, so
    the public-key operations never autostart an agent — the test was
    asserting about a daemon that was never created, and a mutation that
    removed the kill went undetected.

    So the test now forces the situation it means to test: it creates a
    REAL gpg-agent for a homedir, then checks that the production
    verifier's cleanup reaps it.

    (A `gpg-agent` for a homedir is started by a keygen or a signing
    operation, which the production path does not perform — but a
    CALLER-SUPPLIED workdir may already have one, and that is the case
    the production cleanup must handle.)
    """
    import shutil
    import subprocess
    import tempfile
    import time

    from vishwas import gpg_check

    base = pathlib.Path(tempfile.mkdtemp(prefix="vishwas-gpg-prod-"))
    try:
        # 1) create a REAL gpg-agent for a homedir, the way a caller that
        #    already used that homedir would have.
        home = base / "caller_home"
        home.mkdir(mode=0o700)
        params = home / "_params"
        params.write_text(
            "Key-Type: RSA\nKey-Length: 1024\nSubkey-Type: RSA\n"
            "Subkey-Length: 1024\nName-Real: Agent Probe\n"
            "Name-Email: probe@example.org\nExpire-Date: 0\n"
            "Passphrase: probe\n%commit\n")
        subprocess.run(
            ["gpg", "--homedir", str(home), "--batch", "--no-tty",
             "--pinentry-mode", "loopback", "--gen-key", str(params)],
            capture_output=True, timeout=300)
        time.sleep(1)

        started = _gpg_agents(str(home))
        assert started, (
            "could not start a gpg-agent for the probe — the test would be "
            "vacuous, so it must fail loudly rather than pass vacuously")

        # 2) the production verifier must reap it
        gpg_check.verify_with_truststore(
            b"payload", b"not-a-signature",
            truststore_dir=str(base / "no-such-truststore"),
            workdir=home)
        time.sleep(2)

        survived = _gpg_agents(str(home))
        assert not survived, (
            f"the production verifier left {len(survived)} gpg-agent "
            f"daemon(s) alive for the homedir it used: {sorted(survived)}")
    finally:
        shutil.rmtree(base, ignore_errors=True)
