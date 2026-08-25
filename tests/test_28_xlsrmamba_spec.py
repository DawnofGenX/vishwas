"""XLSR-Mamba learned-arch tests — xlsrmamba family (arXiv 2411.10027, MIT).

Building the XLSR-Mamba skeleton requires REAL torch (``torch.nn`` + einops).
The hermetic suite runs under system python whose torch is a STUB without
``torch.nn``; in that environment this module degrades to a single clean skip
so the baseline stays green. Under the docling tree (real CPU torch) every
test below runs.

No real WEIGHTS are loaded anywhere here — the 300M-param wav2vec2-XLSR front-
end is replaced by a tiny fake emitting the same (B, T, 1024) contract, and
the bidirectional Mamba backend runs at reduced width/depth with random init.
Registry plumbing and adapter registration are checked without any checkpoint.
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
        from vishwas.model_adapters import ADAPTERS, _ARCH_FAMILIES, resolve
        from vishwas.model_archs import get_arch
        from vishwas.model_archs.base import ArchSpec
        from vishwas.model_archs.xlsrmamba import (
            XLSRMambaSpec,
            _INPUT_SAMPLES,
            _XLSRMambaNet,
            _build_net,
        )
        _ARCH_ERROR = None
    except Exception as e:  # real torch present but vendor tree broken -> RED
        _ARCH_ERROR = f"{type(e).__name__}: {e}"
else:
    _ARCH_ERROR = None


if _TORCH_ERROR is not None:
    def test_xlsrmamba_arch_skipped():
        pytest.skip(f"real torch.nn unavailable in this env ({_TORCH_ERROR})")
elif _ARCH_ERROR is not None:
    def test_xlsrmamba_arch_vendored():
        pytest.fail(f"xlsrmamba architecture not vendored ({_ARCH_ERROR})")
else:
    # ----------------------------------------------------- registry seams --
    def test_registry_returns_xlsrmamba_spec():
        spec = get_arch("xlsrmamba")
        assert isinstance(spec, ArchSpec)
        assert spec.name == "xlsrmamba"
        assert spec.weight_env == "VISHWAS_XLSRMAMBA_WEIGHTS"
        assert spec.implemented is True

    def test_adapter_row_registered():
        assert "VISHWAS_XLSRMAMBA_WEIGHTS" in ADAPTERS
        assert _ARCH_FAMILIES.get("VISHWAS_XLSRMAMBA_WEIGHTS") == "xlsrmamba"
        adapter = resolve("VISHWAS_XLSRMAMBA_WEIGHTS")
        assert adapter is not None and adapter.family == "audio"

    # ---------------------------------------------------------- skeleton --
    class _FakeW2V(nn.Module):
        """Stand-in for wav2vec2-XLSR: emits (B, ~208, 1024) like the real
        frontend does for a 66800-sample input. Deterministic; one probe param
        so the 'ssl_model.model.*' prefix exists like the real model."""

        def __init__(self, cfg=None):
            super().__init__()
            self.cfg = cfg
            self._probe = nn.Parameter(torch.zeros(1))

        def forward(self, source, mask=False, features_only=False, **kw):
            b = source.shape[0]
            t = max(1, source.shape[-1] // 320)
            assert features_only and mask is False  # inference contract (X6)
            return {"x": torch.zeros(b, t, 1024)}

    def _tiny_mixer(**kw):
        """Real vendored MixerModel at reduced depth — exercises the actual
        Mamba path (selective-scan shim, flip column, pooling) cheaply while
        keeping d_model=144 so the net's LL/first_bn widths line up."""
        from vishwas.model_archs._xlsrmamba_vendor.mamba_backend import MixerModel

        kw.pop("ssm_cfg", None)
        kw["d_model"] = 144
        kw["n_layer"] = 1
        kw.setdefault("rms_norm", True)
        kw.setdefault("residual_in_fp32", True)
        kw.setdefault("fused_add_norm", False)
        kw.setdefault("device", "cpu")
        return MixerModel(**kw)

    @pytest.fixture()
    def tiny_net(monkeypatch):
        net = _build_net(device="cpu", mixer_cls=_tiny_mixer, w2v_cls=_FakeW2V)
        net.eval()
        return net

    def test_skeleton_prefixes_mirror_checkpoint(tiny_net):
        keys = set(tiny_net.state_dict().keys())
        assert any(k.startswith("ssl_model.model.") for k in keys)
        assert any(k.startswith("conformer.forward_layers.") for k in keys)
        assert any(k.startswith("conformer.backward_layers.") for k in keys)
        for k in ("LL.weight", "first_bn.weight", "conformer.norm_f.weight",
                  "conformer.classifier.weight", "conformer.f_attention_pool.weight",
                  "conformer.b_attention_pool.weight"):
            assert k in keys

    def test_build_eval_mode():
        net = _build_net(device="cpu", mixer_cls=_tiny_mixer, w2v_cls=_FakeW2V)
        assert not net.training  # inference-only usage (X6)

    # -------------------------------------------------------- apply_state --
    def test_apply_state_rejects_mismatched_sd(tiny_net):
        spec = XLSRMambaSpec()
        bad = dict(tiny_net.state_dict())
        bad["conformer.classifier.weight"] = torch.randn(2, 999)
        assert spec.apply_state(tiny_net, bad) is False
        assert spec.last_apply["ok"] is False

    def test_apply_state_rejects_empty_payload():
        spec = XLSRMambaSpec()
        net = object()  # no load_state_dict attribute at all
        assert spec.apply_state(net, {}) is False
        assert spec.apply_state(net, None) is False

    def test_apply_state_accepts_own_skeleton_strict(tiny_net):
        spec = XLSRMambaSpec()
        ok = spec.apply_state(tiny_net, dict(tiny_net.state_dict()))
        assert ok is True
        assert spec.last_apply.get("strict") is True

    # --------------------------------------------------------------- score --
    def test_score_pads_and_returns_calibrated_probability(tiny_net, monkeypatch):
        import vishwas.model_archs.xlsrmamba as mod

        seen = {}

        orig_forward = type(tiny_net).forward

        def spy_forward(self, wav):
            seen["len"] = wav.shape[-1]
            return orig_forward(self, wav)

        monkeypatch.setattr(type(tiny_net), "forward", spy_forward)
        p = XLSRMambaSpec().score(tiny_net, np.random.randn(16000).astype(np.float32))
        assert seen["len"] == _INPUT_SAMPLES  # repeat-tile to trained window (X4)
        assert isinstance(p, float) and 0.0 <= p <= 1.0

    def test_score_truncates_long_input(tiny_net, monkeypatch):
        seen = {}
        orig_forward = type(tiny_net).forward

        def spy_forward(self, wav):
            seen["len"] = wav.shape[-1]
            return orig_forward(self, wav)

        monkeypatch.setattr(type(tiny_net), "forward", spy_forward)
        XLSRMambaSpec().score(
            tiny_net, np.random.randn(_INPUT_SAMPLES * 3).astype(np.float32))
        assert seen["len"] == _INPUT_SAMPLES

    def test_score_deterministic_in_eval(tiny_net):
        wav = np.random.randn(32000).astype(np.float32) * 0.1
        s = XLSRMambaSpec()
        assert s.score(tiny_net, wav) == s.score(tiny_net, wav)

    def test_score_rejects_non_waveform(tiny_net):
        with pytest.raises(TypeError):
            XLSRMambaSpec().score(tiny_net, "not-a-waveform")

    # ------------------------------------------------- label-order honesty --
    def test_label_order_index0_is_spoof(tiny_net):
        """Force logits [spoof, bonafide] = [10, -10]; score must be ~1.0.
        Upstream maps bonafide=1/spoof=0 (X5) — inverted vs fakemamba."""
        class _Fixed(_FakeW2V):
            pass

        net = tiny_net
        # Patch conformer head to emit known logits regardless of input.
        orig = net.conformer.classifier.weight.data.clone()
        bias = net.conformer.classifier.bias.data.clone()
        try:
            with torch.no_grad():
                net.conformer.classifier.weight.zero_()
                net.conformer.classifier.bias.copy_(
                    torch.tensor([10.0, -10.0]))  # [spoof, bonafide]
            p = XLSRMambaSpec().score(net, np.zeros(16000, dtype=np.float32))
        finally:
            with torch.no_grad():
                net.conformer.classifier.weight.copy_(orig)
                net.conformer.classifier.bias.copy_(bias)
        assert p > 0.99
