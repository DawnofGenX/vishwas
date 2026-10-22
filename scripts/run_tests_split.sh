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
# Usage:
#   scripts/run_tests_split.sh            # both parts (the default)
#   scripts/run_tests_split.sh a          # part A only
#   scripts/run_tests_split.sh b          # part B only
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

# Run one pytest invocation OUTSIDE the Hermes worker cgroup, in its own
# systemd scope, writing to a file so the output survives a kill.
run_part () {
  local label="$1"; shift
  local out
  out="$(mktemp)"
  systemd-run --user --scope -p "MemoryMax=${PART_MEM_MAX}" \
    timeout --kill-after=30 "${PART_TIMEOUT_S}" \
    python -m pytest "$@" -q > "$out" 2>&1
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
  a) part_a ;;
  b) part_b ;;
  both)
    part_a; rc_a=$?
    part_b; rc_b=$?
    echo
    if [ "$rc_a" -eq 0 ] && [ "$rc_b" -eq 0 ]; then
      echo "Both parts passed."
      exit 0
    fi
    echo "FAILED: part A exit ${rc_a}, part B exit ${rc_b}"
    echo "NOTE: exit 137/-9 with no pytest output is the cgroup OOM killer,"
    echo "      NOT a test failure. The Hermes worker scope is capped at 4 GB"
    echo "      and a torch checkpoint needs ~4.17 GB, so each part must run"
    echo "      via systemd-run --user --scope (which this script does)."
    exit 1
    ;;
  *) echo "usage: $0 [a|b|both]" >&2; exit 2 ;;
esac
