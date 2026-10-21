"""NYUAD 3-class AI-image detector (second image signal) — env-gated tests.

Runs only under .venv-ambient (the tree whose transformers 5.14.1 loads ViT;
the webhook's docling-python/transformers-5.15 cannot). Skips in bare env.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = "/home/hermes/vishwas"
_AMBIENT_PY = "/home/hermes/.venv-ambient/bin/python"

sys.path.insert(0, _REPO + "/src")


def _ambient():
    return Path(_AMBIENT_PY).exists()


pytestmark = pytest.mark.skipif(not _ambient(), reason="NYUAD runs only under .venv-ambient")


def _score(py: str) -> float:
    """Run a tiny scoring snippet under .venv-ambient and return the printed float."""
    import subprocess
    code = (
        "import sys,os; sys.path.insert(0,'/home/hermes/vishwas/src'); "
        "os.environ['VISHWAS_DEVICE']='cpu'; "
        "from vishwas.model_archs.nyuad import NyuadSpec; "
        "s=NyuadSpec(); m=s.build(); "
        f"print('%.6f'%s.score(m,'{py}'))"
    )
    r = subprocess.run([_AMBIENT_PY, "-c", code], capture_output=True, text=True, timeout=300, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert r.returncode == 0, r.stderr[-300:]
    return float(r.stdout.strip().splitlines()[-1])


def test_nyuad_flux_high(environ_request=None):
    p = _score("/home/hermes/fusion_img/ai/flux_04.png")
    assert p > 0.9, f"flux_04 should read p_fake>0.9, got {p}"


def test_nyuad_real_low():
    p = _score("/home/hermes/fusion_img/real/picsum_15.jpg")
    assert p < 0.1, f"picsum_15 should read p_fake<0.1, got {p}"
    q = _score("/home/hermes/fusion_img/real/op_sih_3.jpg")
    assert q < 0.1, f"op_sih_3 should read p_fake<0.1, got {q}"