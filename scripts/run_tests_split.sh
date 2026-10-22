#!/usr/bin/env bash
# Run the Vishwas test suite in two processes.
#
# WHY TWO PROCESSES
# -----------------
# The full suite in ONE process is OOM-KILLED by the cgroup on this host
# (each torch model costs ~4 GB RSS once the aasist3 / xlsr-mamba / SPAI /
# EFFORT / HAVIC checkpoints are loaded). The kill shows up as exit -9 with
# an empty pytest output, and `dmesg` reports "Memory cgroup out of memory:
# Killed process ... anon-rss: 4170064kB". It is NOT a test failure.
#
# The two parts together cover every test file (57 + 7 = 64) with no gap.
#
# ⚠️ WHY EACH PART RUNS IN ITS OWN MEMORY SCOPE (2026-10-22)
# -------------------------------------------------------------
# THIS HOST'S HERMES WORKER SCOPES ARE CAPPED AT 4 GB
# (`hermes-worker-proc_*.scope: memory.max = 4294967296`). A single torch
# checkpoint needs ~4.17 GB RSS, so the HEAVY part CANNOT RUN AT ALL inside
# a worker scope — it is OOM-killed no matter how it is invoked.
#
# The parent `app.slice` has NO limit (`memory.max = max`). So each part is
# run via `systemd-run --user --scope`, which places it in a fresh scope
# OUTSIDE the worker cgroup.
#
# BEFORE (2026-10-22): the two parts ran sequentially in ONE shell, in the
# worker scope. Observed failure: `run_tests_split.sh` with no argument
# (the DOCUMENTED usage) died at ~220 s with NO pytest output at all, and
# `dmesg` showed "Memory cgroup out of memory: Killed process ... (python)
# anon-rss: 4172144kB" — the OOM killer took the shell along with the python
# process, and part A's already-printed output was LOST because the pipe was
# never flushed.
#
# `setsid` alone does NOT fix it: the cgroup limit applies to the cgroup,
# not the process group, so a new process group still inherits the 4 GB
# cap. Only moving to a different cgroup does.
#
# INTERPRETER (2026-10-22) — PINNED, NEVER A BARE `python`
# ----------------------------------------------------------
# The runner used to invoke a bare `python`. On this host `python` resolves
# to the Hermes 3.14 venv, which has NO pytest:
#
#     --- PART A (exit 1) ---
#     /home/hermes/.hermes/installs/<id>/.../venv/bin/python:
#         No module named pytest
#
# So the runner produced NO test results at all, and the "PART A (exit 1)"
# line was pytest's own error rather than a test outcome — a caller could
# not tell "the suite failed" from "the suite never ran".
#
# The interpreter is now resolved explicitly, in this order:
#   1. $VISHWAS_TEST_PYTHON          (explicit override)
#   2. /home/hermes/.venv-ambient/bin/python   (the service venv: torch,
#      torchvision, transformers — the tree that can load every gate)
#   3. /usr/bin/python3              (3.12; torch 2.4.1, no torchvision)
#   4. whatever `command -v python3` finds, ONLY if it can import pytest
# and then it is VERIFIED to import pytest before use. If none can, the
# runner says so ITSELF (FATAL) instead of letting pytest print a
# confusing error.
#
# NOTE the difference between the two fallbacks, measured 2026-10-22 on the
# seven heavy modules:
#     /usr/bin/python3 (3.12, torch 2.4.1)  -> 58 passed, 3 skipped
#     .venv-ambient   (3.11, torch 2.14.1)  -> 60 passed, 2 skipped
# The two extra passes are the torchvision-gated SPAI test. So the venv is
# PREFERRED, not merely allowed.
#
# Usage:
#   scripts/run_tests_split.sh            # both parts (the default)
#   scripts/run_tests_split.sh a          # part A only
#   scripts/run_tests_split.sh b          # part B only
#   scripts/run_tests_split.sh python     # print the resolved interpreter
set -uo pipefail
cd "$(dirname "$0")/.."

# The seven modules that load a heavy torch checkpoint. Splitting them out
# keeps each process's resident set small enough to survive.
HEAVY_MODULES=(
  tests/test_19_aasist_arch.py
  tests/test_20_effort_arch.py
  tests/test_21_havic_arch.py
  tests/test_28_xlsrmamba_spec.py
  tests/test_34_audio_input_sensitivity.py
  tests/test_39_spai_arch.py
  tests/test_51_effnet_image_arch.py
)

IGNORE_ARGS=()
for m in "${HEAVY_MODULES[@]}"; do IGNORE_ARGS+=("--ignore=$m"); done

# Per-part wall budget. A part that exceeds it is a HANG, not a slow run:
# the suite takes ~58 s (A) and ~31 s (B) on this host.
PART_TIMEOUT_S="${VISHWAS_TEST_PART_TIMEOUT_S:-420}"

# The scope limit for a part. It must exceed the ~4.17 GB a single torch
# checkpoint needs, with headroom.
PART_MEM_MAX="${VISHWAS_TEST_MEM_MAX:-6G}"

# Resolve an interpreter that can actually import pytest.
#
# A candidate is accepted only if it exists, is executable, and can import
# pytest. The probe is cheap (one -c import) and it is the difference
# between "the suite failed" and "the suite never ran".
resolve_python () {
  local c
  for c in "${VISHWAS_TEST_PYTHON:-}" \
           "/home/hermes/.venv-ambient/bin/python" \
           "/usr/bin/python3"; do
    [ -n "$c" ] || continue
    [ -x "$c" ] || continue
    if "$c" -c "import pytest" >/dev/null 2>&1; then
      echo "$c"
      return 0
    fi
  done
  # Last resort: anything named python3 on the PATH, but ONLY if it can
  # import pytest. A bare name is PATH-dependent, so it is the weakest
  # option and is only reached when nothing explicit exists.
  c="$(command -v python3 2>/dev/null || true)"
  if [ -n "$c" ] && "$c" -c "import pytest" >/dev/null 2>&1; then
    echo "$c"
    return 0
  fi
  return 1
}

PY="$(resolve_python || true)"
if [ -z "${PY}" ]; then
  echo "FATAL: no interpreter on this host can import pytest." >&2
  echo "  Tried: \$VISHWAS_TEST_PYTHON, /home/hermes/.venv-ambient/bin/python," >&2
  echo "         /usr/bin/python3, and python3 on the PATH." >&2
  echo "  Fix with one of:" >&2
  echo "    pip install pytest            # into the interpreter you want" >&2
  echo "    VISHWAS_TEST_PYTHON=/path/to/python bash scripts/run_tests_split.sh" >&2
  exit 1
fi

# Print the tree this part ran against, as the LAST line of the runner's
# output. 2026-10-22: the stamp pytest prints lands second-to-last, so a
# caller piping through `tail -3` / `tail -4` would drop it -- and the
# background-run capture commands do exactly that. The runner printing it
# itself, after the part's output, makes it survive any tail.
tree_stamp () {
  local sha dirty=""
  sha="$(git -C "$(dirname "$0")/.." rev-parse --short HEAD 2>/dev/null)"
  if [ -z "$sha" ]; then
    echo "vishwas tree: UNKNOWN (no git info)"
    return
  fi
  if [ -n "$(git -C "$(dirname "$0")/.." status --porcelain 2>/dev/null)" ]; then
    dirty=" (DIRTY WORKTREE)"
  fi
  echo "vishwas tree: ${sha}${dirty}"
}

# Run one pytest invocation OUTSIDE the Hermes worker cgroup, in its own
# systemd scope, writing to a file so the output survives a kill.
run_part () {
  local label="$1"; shift
  local out
  out="$(mktemp)"
  systemd-run --user --scope -p "MemoryMax=${PART_MEM_MAX}" \
    timeout --kill-after=30 "${PART_TIMEOUT_S}" \
    "$PY" -m pytest "$@" -q > "$out" 2>&1
  local rc=$?
  echo "--- ${label} (exit ${rc}) ---"
  # The summary line is at the end; print the tail so the counts are visible
  # even if the run was killed part-way.
  tail -6 "$out"
  rm -f "$out"
  return $rc
}

part_a () {
  echo "=== PART A: everything except the torch-heavy modules ==="
  run_part "PART A" tests/ "${IGNORE_ARGS[@]}"
}

part_b () {
  echo "=== PART B: the torch-heavy modules ==="
  run_part "PART B" "${HEAVY_MODULES[@]}"
}

case "${1:-both}" in
  a)
    part_a; rc_a=$?
    tree_stamp
    exit $rc_a
    ;;
  b)
    part_b; rc_b=$?
    tree_stamp
    exit $rc_b
    ;;
  python)
    # Print the interpreter this run would use. A caller that wants to
    # reproduce the run needs to know WHICH python it was.
    echo "$PY"
    ;;
  both)
    part_a; rc_a=$?
    part_b; rc_b=$?
    echo
    if [ "$rc_a" -eq 0 ] && [ "$rc_b" -eq 0 ]; then
      echo "Both parts passed."
      tree_stamp
      exit 0
    fi
    echo "FAILED: part A exit ${rc_a}, part B exit ${rc_b}"
    echo "NOTE: exit 137/-9 with no pytest output is the cgroup OOM killer,"
    echo "      NOT a test failure. The Hermes worker scope is capped at 4 GB"
    echo "      and a torch checkpoint needs ~4.17 GB, so each part must run"
    echo "      via systemd-run --user --scope (which this script does)."
    tree_stamp
    exit 1
    ;;
  *) echo "usage: $0 [a|b|both]" >&2; exit 2 ;;
esac
