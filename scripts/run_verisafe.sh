#!/usr/bin/env bash
# VeriSafe service launcher — zero-cloud, CPU-only friendly.
#
# Wires up the two non-system Python dependency trees BEFORE starting the
# app, so detect_available_deps() sees them and the relevant gates engage:
#   1. /home/hermes/pylibs          (opencv/cv2, yara_x, pefile, lief, trimmed asn1crypto)
#      -> also installed as a system .pth (verisafe-pylibs.pth); exported here
#         too for belt-and-suspenders and for cron/manual invocations.
#   2. /home/hermes/docling-python  (isolated ~5.5GB docling + RapidOCR ONNX tree)
#      -> docling's find_spec gate engages the structured-layout extraction tier
#         in gov_document WITHOUT touching site-packages.
#
# Usage:
#   scripts/run_verisafe.sh webhook [--port N] [--host H] [--openwa-url U] [--budget-s S]
#   scripts/run_verisafe.sh cli     [--text ... | --file ...] ...
#
# Extra env honored (all optional):
#   VERISAFE_FFMPEG_THREADS (default 2 — thermal-safe cap for i5-8250U)
#   VERISAFE_RAG_CACHE / VERISAFE_RAG_VERSION
#   VERISAFE_EFFORT_WEIGHTS / VERISAFE_AASIST_WEIGHTS / ... (model weights)
set -euo pipefail

cd "$(dirname "$0")/.."   # project root

export PYTHONPATH="/home/hermes/pylibs:/home/hermes/docling-python:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${VERISAFE_FFMPEG_THREADS:=2}" ; export VERISAFE_FFMPEG_THREADS
: "${VERISAFE_LOG_LEVEL:=INFO}"   ; export VERISAFE_LOG_LEVEL

sub="${1:-webhook}"
shift || true
exec python3 -m verisafe.app "$sub" "$@"
