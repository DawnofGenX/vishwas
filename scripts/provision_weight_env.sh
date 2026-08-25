#!/usr/bin/env bash
# provision_weight_env.sh — idempotent pointer-setter for Vishwas model weights.
#
# Scans /opt/vishwas/models (override with VISHWAS_MODELS_DIR) and prints
# `export VISHWAS_<GATE>_WEIGHTS=<path>` lines for every gate whose primary
# artefact exists, plus a summary table. Absent gates are skipped (printed as
# 'absent') so sourcing this in a fresh environment is always safe.
#
# Usage:
#   source scripts/provision_weight_env.sh          # exports + table into current shell
#   eval "$(bash scripts/provision_weight_env.sh --quiet)"   # exports only (no table)
#   bash scripts/provision_weight_env.sh            # table + exports (human view)
set -u

MODELS_DIR="${VISHWAS_MODELS_DIR:-/opt/vishwas/models}"
QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

# gate env var | primary relative path | alternates (relative, space-sep)
GATES=(
  "VISHWAS_AASIST_WEIGHTS|aasist/best_model.pth|"
  "VISHWAS_EFFORT_WEIGHTS|effort/chameleon/effort_chameleon.pth|effort/ffpp/effort_ffpp.pth effort/genimage/effort_genimage.pth"
  "VISHWAS_HAVIC_WEIGHTS|havic/best_ft/best_ft_model.pth|havic/pt200/pt_model.200.pth"
  "VISHWAS_XLSRMAMBA_WEIGHTS|xlsr-mamba/model.safetensors|"
  "VISHWAS_DEMAMBA_WEIGHTS||"
  "VISHWAS_FAKEMAMBA_WEIGHTS||"
  "VISHWAS_SSL_AUDIO_WEIGHTS||"
  "VISHWAS_IMAGE_FACE_WEIGHTS||"
)

table() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }   # table rows only; exports always print

[ "$QUIET" = 1 ] || printf '%-30s %-45s %12s  %s\n' "GATE" "PATH" "SIZE" "STATUS"
for row in "${GATES[@]}"; do
  IFS='|' read -r envvar primary alts <<<"$row"
  if [ -n "$primary" ] && [ -f "$MODELS_DIR/$primary" ]; then
    size=$(stat -c%s "$MODELS_DIR/$primary" 2>/dev/null || echo '?')
    mb=$(awk -v b="$size" 'BEGIN{printf "%.1f", b/1048576}')
    # PLAIN KEY=VAL: systemd EnvironmentFile silently DROPS `export KEY=VAL`
    # lines (2026-08-24 incident: AASIST/HAVIC gates dark in the webhook while
    # /health stayed green). Shell users: eval with set -a, or source after
    # `set -a`. Never paste these lines into an EnvironmentFile with export.
    printf '%s=%s\n' "$envvar" "$MODELS_DIR/$primary"
    table "$(printf '%-30s %-45s %11s MB  ok' "$envvar" "$primary" "$mb")"
  else
    # try alternates in order
    found=""
    for alt in $alts; do
      if [ -f "$MODELS_DIR/$alt" ]; then found="$alt"; break; fi
    done
    if [ -n "$found" ]; then
      size=$(stat -c%s "$MODELS_DIR/$found" 2>/dev/null || echo '?')
      mb=$(awk -v b="$size" 'BEGIN{printf "%.1f", b/1048576}')
      printf '%s=%s\n' "$envvar" "$MODELS_DIR/$found"
      table "$(printf '%-30s %-45s %11s MB  ok(alt)' "$envvar" "$found" "$mb")"
    else
      table "$(printf '%-30s %-45s %12s  absent' "$envvar" "${primary:-<none-public>}" "-")"
    fi
  fi
done
table ""
table "# provisioned from $MODELS_DIR"
