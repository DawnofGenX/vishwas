"""AASIST3 subprocess fallback: when weights are provisioned but the in-process
arch cannot load (webhook's docling-python transformers-5.15 stack), the aasist
lane shells out to .venv-ambient to still produce real audio evidence.
Weights-gated: skips under bare-env (no VISHWAS_AASIST_WEIGHTS) / no helper.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, "/home/hermes/vishwas/src")
from vishwas.capabilities import deepfake_audio as daudio


def _has_weights_env():
    p = os.environ.get("VISHWAS_AASIST_WEIGHTS")
    return bool(p) and Path(p).exists()


def test_aasist3_subprocess_helper_polarity_real_spoof():
    """Real aasist3 via the .venv-ambient helper: bonafide LOW, spoof HIGH.
    Requires weights + the helper env; skips otherwise (hermetic bare env)."""
    if not _has_weights_env() or not Path(daudio._VENV_AMBIENT_PY).exists():
        pytest.skip("aasist3 weights / .venv-ambient not provisioned")
    # use official-eval-derived bonafide/spoof wavs if present, else generate a sine
    import subprocess, numpy as np
    from vishwas.model_archs.aasist3 import SpectraAASIST3Spec  # noqa: F401 (import guard)
    # direct helper polarity (single-file, deterministic)
    bona = Path("/tmp/score_bona.wav")
    spoof = Path("/tmp/score_spoof.wav")
    # if polarity fixtures absent, synthesize a clean sine (should read bonafide)
    p = daudio._subprocess_aasist3_score(bona if bona.exists() else _mk_wav('/tmp/helper_bona.wav'), "cpu")
    assert p is not None and p < 0.5, f"bonafide helper returned {p}"
    if spoof.exists():
        q = daudio._subprocess_aasist3_score(spoof, "cpu")
        assert q is not None and q > 0.5, f"spoof helper returned {q}"


def _mk_wav(path):
    import numpy as np, subprocess
    t = np.arange(16000) / 16000.0
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", "16000", "-ac", "1",
                    "-i", "pipe:0", str(path)], input=x.tobytes(), check=True)
    return Path(path)


def test_multicrop_subprocess_fallback_with_provisioned_but_broken_arch(monkeypatch, tmp_path):
    """If _load_weights returns None but weights ARE provisioned, and the field is
    aasist, _multi_crop should take the subprocess path (not report missing_dependency).
    Here we stub _load_weights to None + create a fake helper that prints 0.9, so the
    wiring is proven without real weights."""

    def _fake_load(env):
        return None  # simulate the webhook's arch-unavailable

    monkeypatch.setattr(daudio, "_load_weights", _fake_load)
    monkeypatch.setattr(daudio, "_VENV_AMBIENT_PY", sys.executable)
    helper = tmp_path / "fake_helper.py"
    helper.write_text("import sys\nprint('0.92')\n")
    monkeypatch.setattr(daudio, "_AASIST3_HELPER", str(helper))
    monkeypatch.setenv("VISHWAS_AASIST_WEIGHTS", str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors").write_bytes(b"x")
    # need crop windows -> write aud.wav + crops dir
    cropdir = tmp_path / "crops"
    (tmp_path / "aud.wav").write_bytes(b"RIFF")
    cropdir.mkdir(exist_ok=True)
    for i in range(3):
        (cropdir / f"win{i}.wav").write_bytes(b"RIFF")

    cap = daudio.DeepfakeAudioCapability()
    ctx = types.SimpleNamespace(quarantine_root=tmp_path)
    # stub _crop_windows to return the 3 real (readable) crop paths so the
    # subprocess path is exercised (real aud.wav would let ffprobe build them,
    # but a plain RAF fixture can't — stub the crop producer directly).
    crop_paths = [tmp_path / "crops" / f"win{i}.wav" for i in range(3)]
    for cp in crop_paths:
        cp.parent.mkdir(exist_ok=True); cp.write_bytes(b"RIFF")
    monkeypatch.setattr(daudio, "_crop_windows", lambda *a, **k: crop_paths)
    res = cap._multi_crop("aasist", "VISHWAS_AASIST_WEIGHTS", 3, ctx)
    assert res[0].name == "aasist_detector"
    assert res[0].status == "ok", res[0].notes
    assert res[0].signals["prob_deepfake"] == pytest.approx(0.92, abs=0.01)
    assert "missing_dependency" not in res[0].signals
    assert res[0].signals["source"] == "aasist3.subprocess(.venv-ambient)"


def test_multicrop_still_missing_dependency_when_weights_genuinely_absent(monkeypatch, tmp_path):
    """Genuine provisioning gap (env unset) must still report unavailable, NOT
    invoke the subprocess fallback (which would be a false availability)."""
    def _fake_load(env):
        return None
    monkeypatch.setattr(daudio, "_load_weights", _fake_load)
    monkeypatch.delenv("VISHWAS_AASIST_WEIGHTS", raising=False)
    cropdir = tmp_path / "crops"; cropdir.mkdir(exist_ok=True)
    (cropdir.parent / "aud.wav").write_bytes(b"RIFF")
    cap = daudio.DeepfakeAudioCapability()
    ctx = types.SimpleNamespace(quarantine_root=tmp_path)
    res = cap._multi_crop("aasist", "VISHWAS_AASIST_WEIGHTS", 3, ctx)
    assert res[0].status == "unavailable"
    assert "weights not provisioned" in res[0].notes