"""HAVIC vendored-arch tests — Phase 1 Task 1.4a.

Building the HAVIC skeleton requires REAL torch (``torch.nn``).  The hermetic
suite runs under a python whose torch is a STUB without ``torch.nn``; there
this module degrades to a single clean skip so the baseline stays green.
Under the real-torch tree every test below runs.

No real WEIGHTS are loaded anywhere here — the ~300M-param dual-encoder
skeleton is built with random init and validated key-for-key against the
metadata-only probe fixture ``fixtures/havic_best_ft_key_shapes.txt``
(456 keys + shapes from best_ft_model.pth, probed without torch.load).
The single score() forward uses random weights on a batch of one; it proves
the plumbing, not detection quality (polarity is assumption H5 — see
vishwas/model_archs/havic.py).
"""
from __future__ import annotations

import os

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
        from vishwas.model_archs.havic import CHECKPOINT_CHAIN, HavicArch
        from vishwas.model_archs.base import ArchSpec
        from vishwas.model_archs._havic._timm_shim import Attention, DropPath, Mlp
        _ARCH_ERROR = None
    except Exception as e:  # torch present but arch not vendored -> RED
        _ARCH_ERROR = f"{type(e).__name__}: {e}"
else:
    _ARCH_ERROR = None

def _fixture_path():
    return os.path.join(
        os.path.dirname(__file__), "fixtures", "havic_best_ft_key_shapes.txt"
    )


def _fixture_entries():
    """Parse 'key Dx DxD' lines -> {key: shape tuple} (456 entries)."""
    entries = {}
    with open(_fixture_path(), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, shape_s = line.partition(" ")
            entries[key] = tuple(int(p) for p in shape_s.split("x"))
    return entries


if _TORCH_ERROR is not None:

    def test_havic_arch_skipped():
        pytest.skip(f"real torch.nn unavailable in this env ({_TORCH_ERROR})")

elif _ARCH_ERROR is not None:

    def test_havic_arch_vendored():
        # Pre-vendor RED state: torch works but the havic spec is broken.
        pytest.fail(f"havic architecture not vendored ({_ARCH_ERROR})")

else:

    @pytest.fixture(scope="module")
    def net():
        """Full-size HAVIC_FT skeleton with random init (~300M params, once)."""
        n = HavicArch().build()
        return n

    @pytest.fixture()
    def spec():
        return HavicArch()

    # ------------------------------------------------------------- registry --
    def test_registry_returns_havic_spec():
        spec = get_arch("havic")
        assert isinstance(spec, ArchSpec)
        assert isinstance(spec, HavicArch)
        assert spec.name == "havic"
        assert spec.weight_env == "VISHWAS_HAVIC_WEIGHTS"
        assert getattr(spec, "implemented", False) is True
        assert CHECKPOINT_CHAIN == ("best_ft", "pt200")

    # ---------------------------------------------------------------- build --
    def test_skeleton_matches_checkpoint_key_map(net):
        """Every one of the 456 probed best_ft keys must exist in the skeleton
        with the EXACT probed shape — and vice versa (set equality)."""
        entries = _fixture_entries()
        assert len(entries) == 456
        sd = net.state_dict()
        assert set(sd.keys()) == set(entries.keys())
        mismatched = [
            (k, tuple(sd[k].shape), entries[k])
            for k in entries
            if tuple(sd[k].shape) != entries[k]
        ]
        assert not mismatched, f"shape mismatches: {mismatched[:5]}"

    def test_build_eval_mode(net):
        assert not net.training

    # ----------------------------------------------------------- apply_state --
    def test_apply_state_accepts_own_skeleton(net, spec):
        ok = spec.apply_state(net, dict(net.state_dict()))
        assert ok is True
        assert spec.last_apply["frac_missing"] == 0.0
        assert spec.last_apply["frac_unexpected"] == 0.0

    def test_apply_state_strips_module_prefix(net, spec):
        sd = {f"module.{k}": v for k, v in net.state_dict().items()}
        ok = spec.apply_state(net, sd)
        assert ok is True
        assert spec.last_apply["frac_missing"] == 0.0

    def test_apply_state_unwraps_nested_payload(net, spec):
        wrapped = {"model": dict(net.state_dict())}
        ok = spec.apply_state(net, wrapped)
        assert ok is True
        assert spec.last_apply["frac_missing"] == 0.0

    def test_apply_state_rejects_mismatched_sd(net, spec):
        bad = dict(net.state_dict())
        bad["classifier.output_layer.weight"] = torch.randn(1, 999)
        assert spec.apply_state(net, bad) is False
        assert spec.last_apply["ok"] is False

    def test_apply_state_rejects_empty_payload(spec, net):
        assert spec.apply_state(net, {}) is False
        assert spec.apply_state(net, None) is False

    def test_pt200_style_payload_honestly_rejected(net, spec):
        """pt200 carries only encoders+AVIM (+decoders); no heads/reducers/
        pools -> >5% missing -> apply_state False (documented limitation)."""
        pt200_like = {
            k: v
            for k, v in net.state_dict().items()
            if k.startswith(
                ("audio_encoder.", "visual_encoder.", "AudioVisualInteractionModule.")
            )
        }
        assert len(pt200_like) < len(net.state_dict())  # sanity: subset
        assert spec.apply_state(net, pt200_like) is False
        assert spec.last_apply["ok"] is False
        assert spec.last_apply["frac_missing"] > 0.05

    # ------------------------------------------------------- timm shim keys --
    def test_shim_attention_matches_classic_timm_keys():
        attn = Attention(768, num_heads=12, qkv_bias=True)
        keys = set(attn.state_dict().keys())
        assert keys == {"qkv.weight", "qkv.bias", "proj.weight", "proj.bias"}
        assert tuple(attn.qkv.weight.shape) == (2304, 768)
        assert tuple(attn.qkv.bias.shape) == (2304,)
        no_bias = Attention(64, num_heads=4, qkv_bias=False)
        assert "qkv.bias" not in no_bias.state_dict()

    def test_shim_mlp_matches_timm_keys():
        mlp = Mlp(in_features=768, hidden_features=3072)
        keys = set(mlp.state_dict().keys())
        assert keys == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        assert tuple(mlp.fc1.weight.shape) == (3072, 768)

    def test_shim_droppath_identity_in_eval():
        dp = DropPath(0.9)
        x = torch.randn(2, 3, 4)
        dp.eval()
        assert torch.equal(dp(x), x)

    # ---------------------------------------------------------------- score --
    def test_score_rejects_bad_inputs(net, spec):
        audio = torch.randn(1, 1024, 128)
        video = torch.randn(1, 3, 16, 224, 224)
        with pytest.raises(TypeError):
            spec.score(net, "not-a-tuple")
        with pytest.raises(TypeError):
            spec.score(net, (audio,))
        with pytest.raises(TypeError):
            spec.score(net, (torch.randn(7,), video))
        with pytest.raises(TypeError):
            spec.score(net, (audio, torch.randn(3, 16, 224)))
        with pytest.raises(TypeError):
            spec.score(net, (torch.randn(1, 2, 1024, 128), video))  # 2 channels

    def test_score_returns_probability_in_unit_range(net, spec):
        """One real forward pass, batch of one, random weights (slow-ish CPU).

        Proves the vendored forward path runs end-to-end and the posterior is
        calibrated into [0, 1]; NOT a quality/polarity check (assumption H5).
        """
        audio = torch.randn(1, 1024, 128)
        video = torch.randn(1, 3, 16, 224, 224)
        p = spec.score(net, (audio, video))
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
