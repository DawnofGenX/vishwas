# VeriSafe Operations

## Stale-quarantine sweep (`scripts/stale_sweep.sh`)

Every job works inside a quarantine dir under `~/verisafe/quarantine`; if the
process crashes before its purger runs, that dir lingers. The sweep calls
`scan_stale_quarantines()` to delete any quarantine whose manifest `ts` is
older than `VERISAFE_STALE_TTL_S` (default 7200s) and appends an audit line
per removal. It is a silent watchdog: empty output + exit 0 on a quiet tick,
a swept-count line only when something was removed, errors on stderr with a
non-zero exit.

- **Cadence:** run every 15 minutes via cron (TTL is 2h, so worst-case residue
  lives ~2h15m). Suggested line:
  `*/15 * * * * /home/hermes/verisafe/scripts/stale_sweep.sh`
- **Manual run:** `/home/hermes/verisafe/scripts/stale_sweep.sh` (add
  `VERISAFE_QUARANTINE=...` / `VERISAFE_AUDIT_LOG=...` to point elsewhere).
- **Audit log:** `logs/purge_audit.log` (one JSON line per purge/sweep;
  override with `VERISAFE_AUDIT_LOG`).

## Rich `/health` (service observability)

`GET /health` on the webhook service returns: `status`, `uptime_s`,
`jobs_total` / `jobs_ok` / `jobs_failed` (thread-safe counters incremented in
the `MessageProcessor.process()` outcome path), `quarantines_open`
(subdirs under the quarantine root), and a `deps` summary
(`{"available": [...], "count": N}`) plus the backward-compat flat
`deps_available` list.

- **Counters reset on restart** by design (no DB) — treat them as
  since-last-restart figures, not lifetime totals.
- **Daily check:** `curl -s localhost:2785/health | python3 -m json.tool`.
- **Systemd:** see `deploy/verisafe.service.example` for a ready unit
  (`Restart=on-failure`, `RestartSec=5`); install with the comment-block
  steps at the top of that file.
