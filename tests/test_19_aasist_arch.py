"""AASIST (HABLA_WavLM_AASIST) learned-arch tests — Phase 1 Task 1.2.

Building the AASIST skeleton requires REAL torch (``torch.nn``). The hermetic
suite runs under system python whose torch is a STUB without ``torch.nn``; in
that environment this module degrades to a single clean skip so the baseline
stays green. Under the docling tree (real CPU torch) every test below runs.

No real WEIGHTS are loaded anywhere here — the 315M-param WavLM front-end is
replaced by a tiny fake exposing the same ``extract_features`` contract, and
the HtrgGAT backend + head run with random init. The seam load-order itself is
covered by test_14_model_adapters.py.
"""
from __future__ import annotations

import numpy as np
import pytest

try:  # pragma: no cover - environment-dependent
    import torch
    import torch.nn as nn
    _TORCH_ERROR = None
except Exception as e:
    _TORCH_ERROR = f"{type(e).__name__}: {e}"

if _TORCH_ERROR is None:
    try:
        from vishwas.model_archs import get_arch
        from vishwas.model_archs.aasist import (
            AASISTSpec,
            _AASISTNet,
            _INPUT_SAMPLES,
            _aasist_wav_preprocess,
        )
        from vishwas.model_archs.base import ArchSpec
        _ARCH_ERROR = None
    except Exception as e:  # torch present but arch not vendored -> RED
        _ARCH_ERROR = f"{type(e).__name__}: {e}"
else:
    _ARCH_ERROR = None


if _TORCH_ERROR is not None:
    def test_aasist_arch_skipped():
        pytest.skip(f"real torch.nn unavailable in this env ({_TORCH_ERROR})")
elif _ARCH_ERROR is not None:
    def test_aasist_arch_vendored():
        # Pre-vendor RED state: torch works but the aasist spec is not
        # implemented yet (stub raises / names missing).
        pytest.fail(f"aasist architecture not vendored ({_ARCH_ERROR})")
else:
    class _FakeWavLM(nn.Module):
        """Stand-in for WavLM-Large: emits (B, 200, 1024) like the real
        front-end does for a 64000-sample (4 s @ 16 kHz) input. Deterministic
        (zeros) so score() is reproducible; carries one probe param so the
        'frontend.model.*' state-dict prefix exists like the real model."""

        def __init__(self, cfg=None):
            super().__init__()
            self.cfg = cfg
            self._probe = nn.Parameter(torch.zeros(1))

        def extract_features(self, source, padding_mask=None, mask=False,
                             ret_conv=False, output_layer=None, ret_layer_results=False):
            b = source.shape[0]
            return torch.zeros(b, 200, 1024), None

    @pytest.fixture()
    def spec():
        return AASISTSpec()

    @pytest.fixture()
    def tiny_net(monkeypatch):
        """Real _AASISTNet with the WavLM front-end swapped for _FakeWavLM."""
        import vishwas.model_archs.aasist as mod
        monkeypatch.setattr(mod, "WavLM", _FakeWavLM)
        net = _AASISTNet()
        net.eval()
        return net

    # ------------------------------------------------------------- registry --
    def test_registry_returns_aasist_spec():
        spec = get_arch("aasist")
        assert isinstance(spec, ArchSpec)
        assert spec.name == "aasist"
        assert spec.weight_env == "VISHWAS_AASIST_WEIGHTS"

    # ---------------------------------------------------------------- build --
    def test_build_returns_usable_skeleton(tiny_net):
        keys = set(tiny_net.state_dict().keys())
        assert any(k.startswith("frontend.model.") for k in keys)
        assert any(k.startswith("backend.") for k in keys)
        assert "losses.0.fc.weight" in keys
        assert "losses.0.fc.bias" in keys
        assert tuple(tiny_net.losses[0].fc.weight.shape) == (2, 160)

    def test_build_eval_mode(spec, monkeypatch):
        import vishwas.model_archs.aasist as mod
        monkeypatch.setattr(mod, "WavLM", _FakeWavLM)
        net = spec.build()
        assert not net.training  # inference-only usage (assumption A7)

    # ----------------------------------------------------------- apply_state --
    def test_apply_state_rejects_mismatched_sd(tiny_net, spec):
        bad = dict(tiny_net.state_dict())
        bad["losses.0.fc.weight"] = torch.randn(2, 999)  # shape mismatch
        assert spec.apply_state(tiny_net, bad) is False
        assert spec.last_apply["ok"] is False

    def test_apply_state_rejects_empty_payload(spec, tiny_net):
        assert spec.apply_state(tiny_net, {}) is False
        assert spec.apply_state(tiny_net, None) is False

    def test_apply_state_accepts_own_skeleton(tiny_net, spec):
        ok = spec.apply_state(tiny_net, dict(tiny_net.state_dict()))
        assert ok is True
        assert spec.last_apply["frac_missing"] == 0.0
        assert spec.last_apply["frac_unexpected"] == 0.0

    # ---------------------------------------------------------------- score --
    def test_score_returns_calibrated_probability(tiny_net, spec):
        wav = np.random.randn(_INPUT_SAMPLES).astype(np.float32) * 0.1
        p = spec.score(tiny_net, wav)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_score_deterministic_in_eval(tiny_net, spec):
        wav = np.random.randn(_INPUT_SAMPLES).astype(np.float32) * 0.1
        assert spec.score(tiny_net, wav) == spec.score(tiny_net, wav)

    def test_score_pads_short_input_to_trained_length(tiny_net, spec, monkeypatch):
        seen = {}

        def spy_extract(self, source, *a, **k):
            seen["len"] = source.shape[-1]
            return torch.zeros(source.shape[0], 200, 1024), None

        monkeypatch.setattr(type(tiny_net.frontend["model"]), "extract_features", spy_extract)
        p = spec.score(tiny_net, np.random.randn(16000).astype(np.float32))
        assert seen["len"] == _INPUT_SAMPLES
        assert 0.0 <= p <= 1.0

    def test_score_truncates_long_input(tiny_net, spec, monkeypatch):
        seen = {}

        def spy_extract(self, source, *a, **k):
            seen["len"] = source.shape[-1]
            return torch.zeros(source.shape[0], 200, 1024), None

        monkeypatch.setattr(type(tiny_net.frontend["model"]), "extract_features", spy_extract)
        p = spec.score(tiny_net, np.random.randn(_INPUT_SAMPLES * 2).astype(np.float32))
        assert seen["len"] == _INPUT_SAMPLES
        assert 0.0 <= p <= 1.0

    def test_score_rejects_non_waveform(tiny_net, spec):
        with pytest.raises(TypeError):
            spec.score(tiny_net, "not-a-waveform")

    # ---------------------------------------------------------- preprocessor --
    def test_wav_preprocess_passthrough_1d():
        arr = np.random.randn(16000).astype(np.float32)
        out = _aasist_wav_preprocess(arr)
        assert isinstance(out, np.ndarray) and out.ndim == 1 and out.dtype == np.float32
        np.testing.assert_array_equal(out, arr)
