#!/usr/bin/env bash
# Vishwas service launcher — zero-cloud, CPU-only friendly.
#
# Wires up the two non-system Python dependency trees BEFORE starting the
# app, so detect_available_deps() sees them and the relevant gates engage:
#   1. /home/hermes/pylibs          (opencv/cv2, yara_x, pefile, lief, trimmed asn1crypto)
#      -> also installed as a system .pth (vishwas-pylibs.pth); exported here
#         too for belt-and-suspenders and for cron/manual invocations.
#   2. /home/hermes/docling-python  (isolated ~5.5GB docling + RapidOCR ONNX tree)
#      -> docling's find_spec gate engages the structured-layout extraction tier
#         in gov_document WITHOUT touching site-packages.
#
# Usage:
#   scripts/run_vishwas.sh webhook [--port N] [--host H] [--openwa-url U] [--budget-s S]
#   scripts/run_vishwas.sh cli     [--text ... | --file ...] ...
#
# Extra env honored (all optional):
#   VISHWAS_FFMPEG_THREADS (default 2 — thermal-safe cap for i5-8250U)
#   VISHWAS_RAG_CACHE / VISHWAS_RAG_VERSION
#   VISHWAS_EFFORT_WEIGHTS / VISHWAS_AASIST_WEIGHTS / ... (model weights)
set -euo pipefail

cd "$(dirname "$0")/.."   # project root

export PYTHONPATH="/home/hermes/pylibs:/home/hermes/docling-python:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
: "${VISHWAS_FFMPEG_THREADS:=2}" ; export VISHWAS_FFMPEG_THREADS
: "${VISHWAS_LOG_LEVEL:=INFO}"   ; export VISHWAS_LOG_LEVEL

sub="${1:-webhook}"
shift || true
exec python3 -m vishwas.app "$sub" "$@"
