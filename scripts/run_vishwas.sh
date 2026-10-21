#!/usr/bin/env bash
# Vishwas service launcher — zero-cloud, CPU-only friendly.
#
# INTERPRETER (2026-10-21): this launcher now runs the DEDICATED VENV, not
# whatever `python3` resolves to. Two reasons, both learned the hard way:
#
#   1. `python3` is not stable across contexts. In a login shell it is the
#      Hermes 3.14 toolchain; under the systemd user manager it is
#      /usr/bin/python3 (3.12). The same unit file therefore behaved
#      differently in the two contexts, and the suite collected 448 tests in
#      one and 502 in the other (the difference is the torch-gated arch
#      modules, which skip wholesale when torch is absent).
#
#   2. The system python cannot load the learned gates at all. The wav2vec2
#      frontend is a 1.27 GB `pytorch_model.bin` pickle and transformers
#      4.57+ refuses `torch.load` on torch < 2.6 (CVE-2025-32434); the
#      system torch is 2.4.1.
#
# The venv (/home/hermes/.venv-ambient) carries torch 2.14.1+cu130 (CUDA),
# torchvision, transformers 5.20.2, accelerate, opencv, yara-x, pefile,
# lief, soundfile, scipy, einops, yacs, pyarrow, cryptography and the URL
# scanner's requests/bs4/validators/python-whois.
#
# WHY pylibs IS NOT ON THE PATH (2026-10-21):
#   /home/hermes/pylibs/cryptography (v50, Rust-only) no longer ships the
#   `lib` submodule that pyOpenSSL imports as
#   `from cryptography.hazmat.bindings.openssl import lib as _lib`. Python
#   then resolves `lib` to a TOP-LEVEL namespace package — and
#   llama_cpp_python has installed a `lib/` directory of shared objects into
#   site-packages. That directory has no GEN_EMAIL, so
#   `OpenSSL/crypto.py:867` raises
#   `AttributeError: module 'lib' has no attribute 'GEN_EMAIL'`, which
#   cascades into the accelerate/transformers import chain and kills every
#   torch-gated detector. (This is the error the project notes have blamed on
#   "transformers 5.15" since 2026-08-26. It was never transformers' fault.)
#   The venv has its own copy of everything pylibs provided, so pylibs is
#   simply not needed here — and must not be on the path.
#
# Usage:
#   scripts/run_vishwas.sh webhook [--port N] [--host H] [--openwa-url U] [--budget-s S]
#   scripts/run_vishwas.sh cli     [--text ... | --file ...] ...
#
# Extra env honored (all optional):
#   VISHWAS_FFMPEG_THREADS (default 2 — thermal-safe cap for i5-8250U)
#   VISHWAS_RAG_CACHE / VISHWAS_RAG_VERSION
#   VISHWAS_EFFORT_WEIGHTS / VISHWAS_AASIST_WEIGHTS / ... (model weights)
#   VISHWAS_PYTHON (override the interpreter — used by the test suite)
set -euo pipefail

cd "$(dirname "$0")/.."   # project root

# --- interpreter ------------------------------------------------------------
# The venv is the ONLY environment that can load the learned gates. Fall back
# to the system python with a loud warning rather than failing silently: a
# launcher that quietly degrades the whole platform is how the audio and
# image channels died unnoticed.
VENV_PY="/home/hermes/.venv-ambient/bin/python"
if [ -x "$VENV_PY" ]; then
  PY="$VENV_PY"
else
  echo "FATAL: $VENV_PY is missing." >&2
  echo "  The learned detectors (aasist3/xlsr-mamba/effort/spai) CANNOT be" >&2
  echo "  loaded without it, and the channels will fail OPEN (a dead" >&2
  echo "  detector reads as clean). Rebuild it with:" >&2
  echo "    uv venv --python 3.11 /home/hermes/.venv-ambient" >&2
  echo "    VIRTUAL_ENV=/home/hermes/.venv-ambient uv pip install --python \$VIRTUAL_ENV/bin/python \\" >&2
  echo "      'torch>=2.6,<3' torchvision 'transformers>=4.44' 'accelerate>=0.30' \\" >&2
  echo "      safetensors soundfile scipy einops pillow numpy pyarrow yacs \\" >&2
  echo "      opencv-python-headless requests tldextract python-whois validators \\" >&2
  echo "      beautifulsoup4 lxml cryptography yara-x pefile lief" >&2
  exit 1
fi

# --- PYTHONPATH -------------------------------------------------------------
# ONLY the repo's own src. Deliberately NOT:
#   - /home/hermes/pylibs        -> poisons transformers (see the header)
#   - /home/hermes/docling-python -> no longer exists on this host
# The venv supplies every third-party package the app imports.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

: "${VISHWAS_FFMPEG_THREADS:=2}" ; export VISHWAS_FFMPEG_THREADS
: "${VISHWAS_LOG_LEVEL:=INFO}"   ; export VISHWAS_LOG_LEVEL

sub="${1:-webhook}"
shift || true
exec "$PY" -m vishwas.app "$sub" "$@"
