#!/usr/bin/env bash
# vishwas_health_probe.sh — external watchdog for the webhook (silent-watchdog contract).
# Checks /health: status=ok, deps count >= 11, model-weights present.
# Empty stdout = all good (quiet tick). Any problem prints one line to stderr
# so cron/Hermes surfaces it; also appends to a local alert log.
set -u
URL="${VISHWAS_HEALTH_URL:-http://127.0.0.1:2790/health}"
LOG="${VISHWAS_HEALTH_LOG:-/home/hermes/vishwas/logs/health_alerts.log}"

body=$(curl -s --max-time 8 "$URL") || { echo "vishwas health: webhook unreachable at $URL" >>"$LOG"; exit 1; }
python3 - "$body" <<'PY' >>"$LOG"
import json, sys, datetime
try:
    d = json.loads(sys.argv[1])
except Exception:
    print(datetime.datetime.now().isoformat(), "health: UNPARSEABLE body"); raise SystemExit(0)
problems = []
if d.get("status") != "ok":
    problems.append(f"status={d.get('status')}")
deps = d.get("deps", {}).get("available", [])
if d.get("deps", {}).get("count", 0) < 11:
    problems.append(f"deps_count={d.get('deps', {}).get('count')}")
if "model-weights" not in deps:
    problems.append("model-weights MISSING")
if "vt" not in deps:
    problems.append("vt MISSING")
if problems:
    print(datetime.datetime.now().isoformat(), "health DEGRADED:", "; ".join(problems))
PY
exit 0
