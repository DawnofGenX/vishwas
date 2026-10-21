#!/usr/bin/env bash
# vishwas_health_probe.sh — external watchdog for the webhook (silent-watchdog contract).
#
# Checks /health and reports on:
#   1. the webhook being reachable at all (the failure that actually happened);
#   2. the service reporting a non-ok status;
#   3. the model-weights and vt capability gates being present (a degraded
#      detector is the fail-OPEN path: a dead image detector reads TRUST);
#   4. the dependency inventory REGRESSING below a recorded baseline.
#
# DESIGN NOTES (2026-10-21 rewrite)
# ----------------------------------
# TWO BUGS in the previous version, both of which made it useless:
#
# (a) It hardcoded `deps count >= 11` as the threshold. 11 is the number of
#     deps a FULLY-PROVISIONED host emits; this host legitimately emits 7.
#     The old probe therefore reported "DEGRADED" on a perfectly healthy
#     service — 3,706 identical lines in logs/health_alerts.log proved it was
#     crying wolf, which trains you to ignore it. The correct invariant is
#     REGRESSION, not an absolute count: a baseline of the known-good
#     dependency set is recorded once (--write-baseline) and the probe fails
#     only when a dep that WAS present has DISAPPEARED. A dependency that was
#     never provisioned is not a regression.
#
# (b) It ALWAYS exited 0. The "DEGRADED" line was written to the log but the
#     exit status was 0, so cron considered every run a success and nothing
#     ever surfaced the problem. It now exits 1 on any problem.
#
# It also wrote a "DEGRADED" line but printed NOTHING to stderr, so a cron
# job that only mails on stderr output saw silence. It now prints one line to
# stderr AND appends to the log.
#
# Contract: empty stdout = all good (quiet tick). Any problem prints ONE line
# to stderr so cron surfaces it, and appends to the alert log.
#
# Usage:
#   vishwas_health_probe.sh                    # check against the baseline
#   vishwas_health_probe.sh --write-baseline   # record the current dep set
#   vishwas_health_probe.sh --no-baseline      # check without baseline comparison
set -u

URL="${VISHWAS_HEALTH_URL:-http://127.0.0.1:2790/health}"
LOG="${VISHWAS_HEALTH_LOG:-/home/hermes/vishwas/logs/health_alerts.log}"
BASELINE="${VISHWAS_HEALTH_BASELINE:-/home/hermes/vishwas/logs/health_baseline.txt}"

WRITE_BASELINE=0
CHECK_BASELINE=1
for arg in "$@"; do
  case "$arg" in
    --write-baseline) WRITE_BASELINE=1 ;;
    --no-baseline)    CHECK_BASELINE=0 ;;
    *) echo "usage: $0 [--write-baseline|--no-baseline]" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

body=$(curl -s --max-time 8 "$URL") || {
  line="$(date -Iseconds) vishwas health: webhook unreachable at $URL"
  echo "$line" >>"$LOG"
  echo "$line" >&2
  exit 1
}

# --- extract the dep set + status --------------------------------------------
parsed=$(python3 - "$body" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    print("STATUS\tunparseable")
    print("DEPS\t")
    raise SystemExit(0)
deps = d.get("deps", {}).get("available", [])
print("STATUS\t" + str(d.get("status", "?")))
print("DEPS\t" + ",".join(sorted(deps)))
PY
)

status=$(printf '%s\n' "$parsed" | awk -F'\t' '$1=="STATUS"{print $2}')
deps_csv=$(printf '%s\n' "$parsed" | awk -F'\t' '$1=="DEPS"{print $2}')

# --- --write-baseline: record the known-good dep set -------------------------
if [ "$WRITE_BASELINE" = "1" ]; then
  printf '%s\n' "$deps_csv" > "$BASELINE"
  echo "vishwas health: baseline written to $BASELINE (${deps_csv:-none})" >&2
  exit 0
fi

# --- evaluate ----------------------------------------------------------------
problems=""
if [ "$status" != "ok" ]; then
  problems="${problems} status=${status};"
fi
case ",$deps_csv," in
  *,model-weights,*) ;;
  *) problems="${problems} model-weights-MISSING;" ;;
esac
case ",$deps_csv," in
  *,vt,*) ;;
  *) problems="${problems} vt-MISSING;" ;;
esac

# regression check: a dep in the baseline that is gone now
if [ "$CHECK_BASELINE" = "1" ] && [ -s "$BASELINE" ]; then
  missing=""
  IFS=',' read -r -a base_deps <<<"$(cat "$BASELINE")"
  for d in "${base_deps[@]}"; do
    [ -z "$d" ] && continue
    case ",$deps_csv," in
      *",$d,"*) ;;
      *) missing="${missing} ${d}" ;;
    esac
  done
  if [ -n "$missing" ]; then
    problems="${problems} REGRESSED-DEPS:${missing};"
  fi
fi

if [ -n "$problems" ]; then
  line="$(date -Iseconds) vishwas health DEGRADED: ${problems} deps=[${deps_csv}]"
  echo "$line" >>"$LOG"
  echo "$line" >&2
  exit 1
fi
exit 0
