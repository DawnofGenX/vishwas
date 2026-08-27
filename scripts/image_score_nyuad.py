#!/usr/bin/env python3
"""Standalone NYUAD 3-class AI-image scoring helper (runs under .venv-ambient).

The webhook's docling-python/transformers-5.15 stack cannot import any ViT (same
GEN_EMAIL/circular-import failure as audio Wav2Vec2Model), so image_facecheck shells
out here for the NYUAD second signal. Prints a single float = p_fake (dalle+sd
posterior) in [0,1] on stdout; non-zero exit + stderr note on failure.

Usage:
    /home/hermes/.venv-ambient/bin/python scripts/image_score_nyuad.py <image>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO + "/src")
os.environ.setdefault("VISHWAS_DEVICE", "cpu")


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: image_score_nyuad.py <image>", file=sys.stderr)
        return 2
    from vishwas.model_archs.nyuad import NyuadSpec
    spec = NyuadSpec()
    model = spec.build()
    p = spec.score(model, sys.argv[1])
    print(f"{min(1.0, max(0.0, p)):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())