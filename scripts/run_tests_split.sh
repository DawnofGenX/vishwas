#!/usr/bin/env bash
# Run the Vishwas test suite in two processes.
#
# WHY TWO PROCESSES
# -----------------
# The full suite in ONE process is OOM-KILLED by the cgroup on this host
# (25 GB total, ~13 GB available, and each torch model costs ~4 GB RSS once
# the aasist3 / xlsr-mamba / SPAI / EFFORT / HAVIC checkpoints are loaded).
# The kill shows up as exit code -9 with an empty pytest output, and
# `dmesg` reports "Memory cgroup out of memory: Killed process ... anon-rss:
# 4170064kB". It is NOT a test failure.
#
# The two parts together cover every test file (54 + 7 = 61) with no gap.
# Each part is a normal pytest invocation, so the output is the usual.
#
# Usage:
#   scripts/run_tests_split.sh            # both parts
#   scripts/run_tests_split.sh a          # part A only
#   scripts/run_tests_split.sh b          # part B only
set -uo pipefail
cd "$(dirname "$0")/.."

# The seven modules that load a heavy torch checkpoint. Splitting them out
# keeps each process's resident set small enough to survive the cgroup.
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

part_a () {
  echo "=== PART A: everything except the torch-heavy modules ==="
  python -m pytest tests/ -q "${IGNORE_ARGS[@]}"
}

part_b () {
  echo "=== PART B: the torch-heavy modules ==="
  python -m pytest "${HEAVY_MODULES[@]}" -q
}

case "${1:-both}" in
  a) part_a ;;
  b) part_b ;;
  both)
    part_a || { echo "PART A FAILED"; exit 1; }
    part_b || { echo "PART B FAILED"; exit 1; }
    echo
    echo "Both parts passed."
    ;;
  *) echo "usage: $0 [a|b|both]" >&2; exit 2 ;;
esac
