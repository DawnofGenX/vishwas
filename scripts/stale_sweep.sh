#!/usr/bin/env bash
# stale_sweep.sh — silent-watchdog wrapper around vishwas.quarantine.scan_stale_quarantines().
#
# Cron-friendly contract:
#   - quiet tick (nothing stale): NO stdout, exit 0  -> zero mail/noise
#   - work done:                 prints swept count  -> visible when it matters
#   - failure:                   message on stderr, non-zero exit -> cron MTA/pager picks it up
#
# Standalone on purpose: does NOT go through run_vishwas.sh (no model deps needed).
#
# Optional env honored by src/vishwas/quarantine.py at import time:
#   VISHWAS_QUARANTINE    quarantine root to scan      (default ~/vishwas/quarantine)
#   VISHWAS_AUDIT_LOG     purge audit log path         (default <root>/../logs/purge_audit.log)
#   VISHWAS_STALE_TTL_S   staleness threshold seconds  (default 7200)

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)"
cd -- "$REPO_ROOT" || { echo "stale_sweep: cannot cd to repo root ($REPO_ROOT)" >&2; exit 1; }

errfile="$(mktemp)"
out="$(PYTHONPATH=src python3 -c 'from vishwas.quarantine import scan_stale_quarantines as s; r=s(); print(len(r)) if r else None' 2>"$errfile")"
rc=$?
err="$(cat "$errfile" 2>/dev/null)"
rm -f "$errfile"

if [ "$rc" -ne 0 ]; then
    echo "stale_sweep: stale-quarantine scan FAILED (exit $rc): ${err:-<no stderr>}" >&2
    exit "$rc"
fi

if [ -n "$out" ]; then
    echo "stale_sweep: swept $out stale quarantine dir(s)"
fi
exit 0
