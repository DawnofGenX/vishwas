"""NYUAD subprocess load-path regression test.

Locks in the `.venv-ambient` subprocess wiring used for the second image
detector (`image_score_nyuad.py`), mirroring the proven aasist3 pattern in
deepfake_audio. Two layers:
  * hermetic: missing interpreter/helper must yield None (fail honestly),
    never a fabricated score.
  * env-gated: under `.venv-ambient` the helper must actually load ViT and
    produce correct polarity (real low, AI high).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "/home/hermes/vishwas/src")
from vishwas.capabilities import image_facecheck as iface


def _ambient():
    return Path(iface._VENV_AMBIENT_PY).exists()


# ---------------------------------------------------------- hermetic wiring --

def test_nyuad_returns_none_when_interpreter_missing(monkeypatch):
    """Dead interpreter must downgrade to None (honest unavailable), not invent."""
    monkeypatch.setattr(iface, "_VENV_AMBIENT_PY", "/nonexistent/python/xyz")
    assert iface._subprocess_nyuad_score(Path("/tmp/x.png")) is None


def test_nyuad_returns_none_when_helper_missing(monkeypatch):
    """Missing helper script must downgrade to None, not raise."""
    monkeypatch.setattr(iface, "_NYUAD_HELPER", "/nonexistent/helper.py")
    assert iface._subprocess_nyuad_score(Path("/tmp/x.png")) is None


# ------------------------------------------------------- real load (gated) --

@pytest.mark.skipif(not _ambient(), reason=".venv-ambient interpreter not present")
def test_nyuad_loads_and_runs_with_correct_polarity():
    """ViT loads under .venv-ambient and reads real low / AI high."""
    real = iface._subprocess_nyuad_score(
        Path("/home/hermes/fusion_img/real/picsum_15.jpg"))
    fake = iface._subprocess_nyuad_score(
        Path("/home/hermes/fusion_img/ai/flux_04.png"))
    assert real is not None and real < 0.5, f"real photo read high: {real}"
    assert fake is not None and fake > 0.5, f"AI image read low: {fake}"
