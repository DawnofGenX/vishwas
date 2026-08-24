#!/usr/bin/env bash
# fetch_model_weights.sh — VeriSafe model weights fetcher (guarded, DRY_RUN-default)
#
# Per .delegation/brief_weights.md:
#   - set -euo pipefail
#   - idempotent: skips already-complete downloads by size compare (verify_sizes)
#   - downloads ONLY real + public_no_auth artifacts into
#     ${VERISAFE_MODEL_DIR:-/home/hermes/verisafe/.models}/<gate>/
#   - DRY_RUN default: VERISAFE_FETCH_DRY_RUN defaults to 1 (print-only); set 0 to actually download
#   - never pulls multi-GB HEAVY files on this box unless DRY_RUN=0 and explicitly wanted
#
# Usage:
#   scripts/fetch_model_weights.sh                  # DRY_RUN=1 (default): prints curl commands only
#   VERISAFE_FETCH_DRY_RUN=0 scripts/fetch_model_weights.sh   # real downloads
#   VERISAFE_MODEL_DIR=/data/models scripts/fetch_model_weights.sh
set -euo pipefail

MODEL_DIR="${VERISAFE_MODEL_DIR:-/home/hermes/verisafe/.models}"
DRY_RUN="${VERISAFE_FETCH_DRY_RUN:-1}"

log() { printf '[fetch_model_weights] %s\n' "$*"; }

# One record per downloadable artifact (real + public_no_auth only, per manifest).
# Format: gate|url|expected_size_mb|note
# Gates NOT listed here are intentionally skipped with a reason (see SKIP lines).
FETCHABLE=(
  "aasist|https://huggingface.co/DeepFense/HABLA_WavLM_AASIST_NoAug_Seed42/resolve/main/best_model.pth|3791.7|best_model.pth"
  "havic|https://huggingface.co/JielunPeng/HAVIC/resolve/main/best_ft_model.pth|858.8|best_ft_model.pth (companion pt_model.200.pth ~972.8MB not pulled)"
)

# Deterministic filename for each URL under MODEL_DIR/<gate>/
filename_for_url() {
  local url="$1"
  local path="${url#*/resolve/main/}"
  case "$path" in
    */*) printf '%s' "${path##*/}" ;;  # nested: keep basename
    *)   printf '%s' "$path" ;;
  esac
}

# Compare expected vs actual file size; pass when equal or absent-in-dry-run.
# Returns 0 = OK (present & correct size, or nothing yet), 1 = mismatch/corrupt.
verify_sizes() {
  local gate="$1" url="$2" exp_mb="$3" fname dst exp_bytes act_bytes delta
  fname="$(filename_for_url "$url")"
  dst="$MODEL_DIR/$gate/$fname"
  if [[ ! -f "$dst" ]]; then
    log "verify_sizes $gate/$fname: absent (pending download)"
    return 0
  fi
  exp_bytes="$(python3 -c "import sys; print(int(round(float(sys.argv[1]) * 1000 * 1000)) )" "$exp_mb")"
  # Accept within 1% of expected (publishers round MB display values).
  act_bytes="$(wc -c < "$dst")"
  delta=$(( act_bytes > exp_bytes ? act_bytes - exp_bytes : exp_bytes - act_bytes ))
  if (( delta <= exp_bytes / 100 )); then
    log "verify_sizes $gate/$fname: OK ($act_bytes bytes, expected ~$exp_bytes)"
    return 0
  else
    log "verify_sizes $gate/$fname: MISMATCH (got $act_bytes, expected ~$exp_bytes) — re-download needed"
    return 1
  fi
}

main() {
  log "MODEL_DIR=$MODEL_DIR DRY_RUN=$DRY_RUN"

  # --- Explicitly-skipped gates (HEAVY / unavailable / non-existent / token-gated) ---
  echo "SKIP effort: HEAVY (>~1GB transformer x3 checkpoints, CC BY-NC) — not viable under VeriSafe thermal caps"
  echo "SKIP vbsta: confirmed non-existent project (purged P10)"
  echo "SKIP demamba: no public checkpoint yet (source zip only, PARTIAL)"
  echo "SKIP fakemamba: empty repo (README len=0, no releases/assets)"
  echo "SKIP ssl_audio: HF 401 without token (public_no_auth=false)"
  echo "SKIP phishllm: confirmed non-existent project (purged P10)"
  echo "SKIP image_face: DEPS — weights via existing cv2/OpenCV path (see app.py IMAGE_FACE)"

  # --- Verify any already-downloaded files before touching the network ---
  local rec gate url exp_mb fname dst
  for rec in "${FETCHABLE[@]}"; do
    IFS='|' read -r gate url exp_mb _note <<< "$rec"
    fname="$(filename_for_url "$url")"
    dst="$MODEL_DIR/$gate/$fname"
    if verify_sizes "$gate" "$url" "$exp_mb"; then
      [[ -f "$dst" ]] && { log "IDEMPOTENT-SKIP $gate/$fname (already complete)"; continue; }
    else
      log "RE-DOWNLOAD $gate/$fname (size mismatch)"
      rm -f "$dst"
    fi
  done

  # --- Download loop (dry-run guarded) ---
  for rec in "${FETCHABLE[@]}"; do
    IFS='|' read -r gate url exp_mb note <<< "$rec"
    fname="$(filename_for_url "$url")"
    cmd="curl --create-dirs --location --fail --output '$MODEL_DIR/$gate/$fname' '$url'"
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN would run: $cmd  (${note})"
    else
      log "DOWNloading $gate/$fname"
      eval "$cmd"
      verify_sizes "$gate" "$url" "$exp_mb"
    fi
  done

  log "done."
}

main "$@"
