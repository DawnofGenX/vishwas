#!/usr/bin/env bash
# provision_weight_env.sh — idempotent pointer-setter for VeriSafe model weights.
#
# Scans /opt/verisafe/models (override with VERISAFE_MODELS_DIR) and prints
# `export VERISAFE_<GATE>_WEIGHTS=<path>` lines for every gate whose primary
# artefact exists, plus a summary table. Absent gates are skipped (printed as
# 'absent') so sourcing this in a fresh environment is always safe.
#
# Usage:
#   source scripts/provision_weight_env.sh          # exports + table into current shell
#   eval "$(bash scripts/provision_weight_env.sh --quiet)"   # exports only (no table)
#   bash scripts/provision_weight_env.sh            # table + exports (human view)
set -u

MODELS_DIR="${VERISAFE_MODELS_DIR:-/opt/verisafe/models}"
QUIET=0
[ "${1:-}" = "--quiet" ] && QUIET=1

# gate env var | primary relative path | alternates (relative, space-sep)
GATES=(
  "VERISAFE_AASIST_WEIGHTS|aasist/best_model.pth|"
  "VERISAFE_EFFORT_WEIGHTS|effort/chameleon/effort_chameleon.pth|effort/ffpp/effort_ffpp.pth effort/genimage/effort_genimage.pth"
  "VERISAFE_HAVIC_WEIGHTS|havic/best_ft/best_ft_model.pth|havic/pt200/pt_model.200.pth"
  "VERISAFE_DEMAMBA_WEIGHTS||"
  "VERISAFE_FAKEMAMBA_WEIGHTS||"
  "VERISAFE_SSL_AUDIO_WEIGHTS||"
  "VERISAFE_IMAGE_FACE_WEIGHTS||"
)

table() { [ "$QUIET" = 1 ] || printf '%s\n' "$*"; }   # table rows only; exports always print

[ "$QUIET" = 1 ] || printf '%-30s %-45s %12s  %s\n' "GATE" "PATH" "SIZE" "STATUS"
for row in "${GATES[@]}"; do
  IFS='|' read -r envvar primary alts <<<"$row"
  if [ -n "$primary" ] && [ -f "$MODELS_DIR/$primary" ]; then
    size=$(stat -c%s "$MODELS_DIR/$primary" 2>/dev/null || echo '?')
    mb=$(awk -v b="$size" 'BEGIN{printf "%.1f", b/1048576}')
    printf 'export %s=%s\n' "$envvar" "$MODELS_DIR/$primary"
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
      printf 'export %s=%s\n' "$envvar" "$MODELS_DIR/$found"
      table "$(printf '%-30s %-45s %11s MB  ok(alt)' "$envvar" "$found" "$mb")"
    else
      table "$(printf '%-30s %-45s %12s  absent' "$envvar" "${primary:-<none-public>}" "-")"
    fi
  fi
done
table ""
table "# provisioned from $MODELS_DIR"
