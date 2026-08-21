"""EFFORT (Effort-AIGI-Detection) learned-arch tests — Phase 1 Task 1.3.

Building the EFFORT skeleton requires REAL torch (``torch.nn``). The hermetic
suite runs under system python whose torch is a STUB without ``torch.nn``; in
that environment this module degrades to a single clean skip so the baseline
stays green. Under the docling tree (real CPU torch) every test below runs.

No real WEIGHTS are loaded anywhere here — the 303M-param ViT-L/14 + OrthAlign
skeleton is built with random init and exercised directly. The seam load-order
itself is covered by test_14_model_adapters.py; the real-checkpoint 100%-key
coverage proof lives in docs/research/ARCH_VENDOR_EVIDENCE_2026-08-21.md.
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
        from verisafe.model_archs import get_arch
        from verisafe.model_archs.effort import (
            EffortSpec,
            _EffortNet,
            _OrthAlignLinear,
        )
        from verisafe.model_archs.base import ArchSpec
        _ARCH_ERROR = None
    except Exception as e:  # torch present but arch not vendored -> RED
        _ARCH_ERROR = f"{type(e).__name__}: {e}"
else:
    _ARCH_ERROR = None


if _TORCH_ERROR is not None:
    def test_effort_arch_skipped():
        pytest.skip(f"real torch.nn unavailable in this env ({_TORCH_ERROR})")
elif _ARCH_ERROR is not None:
    def test_effort_arch_vendored():
        # Pre-vendor RED state: torch works but the effort spec is not
        # implemented yet (stub raises / names missing).
        pytest.fail(f"effort architecture not vendored ({_ARCH_ERROR})")
else:
    @pytest.fixture(scope="module")
    def net():
        """Real _EffortNet with random init (~303M params, built once)."""
        n = _EffortNet()
        n.eval()
        return n

    @pytest.fixture()
    def spec():
        return EffortSpec()

    # ------------------------------------------------------------- registry --
    def test_registry_returns_effort_spec():
        spec = get_arch("effort")
        assert isinstance(spec, ArchSpec)
        assert spec.name == "effort"
        assert spec.weight_env == "VERISAFE_EFFORT_WEIGHTS"
        assert getattr(spec, "implemented", False) is True

    # ---------------------------------------------------------------- build --
    def test_build_returns_usable_skeleton(net):
        keys = set(net.state_dict().keys())
        assert "backbone.embeddings.class_embedding" in keys
        assert "backbone.embeddings.patch_embedding.weight" in keys
        assert "backbone.embeddings.position_embedding.weight" in keys
        # checkpoint typo 'pre_layrnorm' preserved verbatim (assumption E2)
        assert "backbone.pre_layrnorm.weight" in keys
        assert "backbone.post_layernorm.weight" in keys
        assert "head.weight" in keys and "head.bias" in keys
        # 24 encoder layers, each with the full OrthAlign self-attn set
        assert len(net.backbone.encoder.layers) == 24
        prefix = "backbone.encoder.layers.0."
        l0 = {k[len(prefix):] for k in keys if k.startswith(prefix)}
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            for part in ("weight_main", "bias", "S_residual", "U_residual", "V_residual"):
                assert f"self_attn.{proj}.{part}" in l0, f"missing {proj}.{part}"
        assert tuple(net.head.weight.shape) == (2, 1024)
        assert tuple(net.backbone.embeddings.position_embedding.weight.shape) == (257, 1024)

    def test_build_eval_mode(spec):
        n = spec.build()
        assert not n.training  # inference-only usage (assumption E5)

    # ----------------------------------------------------------- apply_state --
    def test_apply_state_strips_module_prefix(net, spec):
        sd = {f"module.{k}": v for k, v in net.state_dict().items()}
        ok = spec.apply_state(net, sd)
        assert ok is True
        assert spec.last_apply["frac_missing"] == 0.0
        assert spec.last_apply["frac_unexpected"] == 0.0

    def test_apply_state_accepts_own_skeleton(net, spec):
        ok = spec.apply_state(net, dict(net.state_dict()))
        assert ok is True
        assert spec.last_apply["frac_missing"] == 0.0

    def test_apply_state_rejects_mismatched_sd(net, spec):
        bad = dict(net.state_dict())
        bad["head.weight"] = torch.randn(2, 999)  # shape mismatch
        assert spec.apply_state(net, bad) is False
        assert spec.last_apply["ok"] is False

    def test_apply_state_rejects_empty_payload(spec, net):
        assert spec.apply_state(net, {}) is False
        assert spec.apply_state(net, None) is False

    # ------------------------------------------------------- OrthAlign math --
    def test_orthalign_linear_matches_formula():
        """W_eff must equal weight_main + S*(U@V) exactly."""
        layer = _OrthAlignLinear(8)
        x = torch.randn(2, 5, 8)
        out = layer(x)
        w = layer.weight_main + layer.S_residual * (layer.U_residual @ layer.V_residual)
        expected = x @ w.t() + layer.bias
        assert torch.allclose(out, expected, atol=1e-5)

    # ---------------------------------------------------------------- score --
    def test_score_returns_calibrated_probability(net, spec):
        img = np.random.default_rng(0).integers(0, 255, (3, 224, 224)).astype(np.float32) / 255.0
        p = spec.score(net, img)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_score_batch_mean(net, spec):
        img = np.random.default_rng(1).integers(0, 255, (4, 3, 224, 224)).astype(np.float32) / 255.0
        p = spec.score(net, img)
        assert isinstance(p, float) and 0.0 <= p <= 1.0

    def test_score_deterministic_in_eval(net, spec):
        img = np.random.default_rng(2).integers(0, 255, (3, 224, 224)).astype(np.float32) / 255.0
        assert spec.score(net, img) == spec.score(net, img)

    def test_score_rejects_non_image(net, spec):
        with pytest.raises(TypeError):
            spec.score(net, "not-an-image")
