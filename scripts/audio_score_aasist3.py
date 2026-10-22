#!/usr/bin/env python3
"""Standalone Spectra-AASIST3 audio scoring helper.

Runs under /home/hermes/.venv-ambient/bin/python — the ONE tree whose
transformers/torch/accelerate/numpy are mutually compatible with the vendored
aasist3 arch (the webhook's docling-python/transformers-5.15 stack cannot load
Wav2Vec2Model: accelerate circular-import / lib.GEN_EMAIL at module import, so
in-process deepfake_audio._load_weights() returns None -> 'missing_dependency').

deepfake_audio shells out to this helper for the aasist3 (torch) lane.

Usage:
    /home/hermes/.venv-ambient/bin/python scripts/audio_score_aasist3.py \
        <wav_or_any_decodable> [<wav_or_any_decodable> ...] [--device cpu|cuda]

Prints ONE float per input file on stdout = SPOOF posterior p_spoof in
[0,1] (1 = synthetic spoof, 0 = bonafide), in the order the files were
given. Non-zero exit + stderr note on failure.
Reuses the exact provegeed path (Track C, 2026-08-26):
    SpectraAASIST3Spec().build()  ->  spec.score(net, f32_waveform).
The build() hard-fails on ANY missing checkpoint key (1022/1022 expected).

⚠️ MULTIPLE FILES, ONE LOAD (2026-10-22)
-----------------------------------------
This helper used to take exactly ONE file, so the capability shelled out
once PER CROP and the checkpoint was reloaded every time. Measured on
this host:
    aasist3 checkpoint LOAD (cold)   63.1 s
    one inference                     17.6 s
    one crop, end to end              80.5 s
    a 3-crop verdict                ~241 s
    the heavy-stage budget            120 s

**=> The audio lane blew the budget and returned `UNVERIFIED` with
`pending_heavy` on real media.** The load dominates (CUDA barely helps:
66.3 s vs 80.5 s), so the fix is to load ONCE and score every crop.

So the model is built ONCE, before the loop, and reused for every file.
A multi-file call costs 1 load + N inferences; a single-file call costs
exactly what it did before, so existing callers are unaffected.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO + "/src")
# .venv-ambient has its own numpy; ensure vishwas src resolves its torch/transformers
os.environ.setdefault("PYTHONPATH", _REPO + "/src")


def _decode_f32(path: str) -> np.ndarray:
    """Decode any audio file to 16 kHz mono float32 via ffmpeg (no temp file)."""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
         "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"],
        capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg decode failed: {r.stderr[:200]}")
    return np.frombuffer(r.stdout, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+",
                    help="one or more wav/flac/any ffmpeg-decodable files")
    ap.add_argument("--device", default=os.environ.get("VISHWAS_DEVICE", "cpu"))
    ap.add_argument("--timeout-load", type=int, default=120)
    args = ap.parse_args()

    if not os.environ.get("VISHWAS_AASIST_WEIGHTS"):
        os.environ["VISHWAS_AASIST_WEIGHTS"] = "/opt/vishwas/models/aasist3/model.safetensors"
    if not os.environ.get("VISHWAS_SPECTRA_W2V_DIR"):
        os.environ["VISHWAS_SPECTRA_W2V_DIR"] = "/opt/vishwas/models/aasist3/wav2vec2-xls-r-300m"
    os.environ["VISHWAS_DEVICE"] = args.device

    from vishwas.model_archs.aasist3 import SpectraAASIST3Spec

    t0 = time.time()
    # THE MODEL IS BUILT ONCE, BEFORE THE LOOP (2026-10-22). Building it
    # per file cost a 63 s load per crop and pushed the 3-crop audio lane
    # to ~241 s against a 120 s budget.
    spec = SpectraAASIST3Spec()
    net = spec.build()
    load_s = time.time() - t0

    for path in args.audio:
        try:
            wav = _decode_f32(path)
        except Exception as exc:
            # ONE BAD FILE MUST NOT ABORT THE BATCH (2026-10-22). The
            # helper now takes N files so the checkpoint is loaded ONCE, and
            # a single undecodable crop would otherwise throw away the
            # scores of every OTHER crop — including the ones that were
            # fine. The caller treats a short count as a failure (see
            # `_subprocess_aasist3_scores`), so a bad file degrades the
            # lane honestly instead of silently producing a wrong median.
            print(f"ERROR: decode failed for {path}: {exc}", file=sys.stderr)
            print("NAN", flush=True)
            continue
        if wav.size < 1000:
            print(f"ERROR: decoded audio too short (<1000 samples): {path}",
                  file=sys.stderr)
            print("NAN", flush=True)
            continue
        p_spoof = spec.score(net, wav)
        # score() returns spoof posterior; clamp to [0,1] and print as the contract
        p = float(min(1.0, max(0.0, p_spoof)))
        print(f"{p:.6f}", flush=True)
        print(f"# aasist3 p_spoof={p:.4f} ({time.time()-t0:.1f}s, "
              f"load={load_s:.1f}s, device={args.device}) {path}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())