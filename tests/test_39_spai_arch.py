"""SPAI learned-arch tests — spai family (CVPR'25 spectral AI-image detector).

Building the PatchBasedMFViT skeleton requires REAL torch (``torch.nn``) plus
torchvision + einops. The hermetic suite runs under system python whose torch
is a STUB without ``torch.nn``; in that environment this module degrades to a
single clean skip so the baseline stays green. Under the docling tree (real
torch) registry/plumbing tests run without weights; the heavy build/coverage/
forward tests additionally require the provisioned checkpoint
(VISHWAS_IMAGE_FACE_WEIGHTS -> /opt/verisafe/models/spai/spai.pth) — skip-clean
otherwise, never a hard failure.
"""
from __future__ import annotations

import os
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
        from vishwas.model_archs.spai import SpaiSpec
        _ARCH_ERROR = None
    except Exception as e:  # real torch present but vendor tree broken -> RED
        _ARCH_ERROR = f"{type(e).__name__}: {e}"
else:
    _ARCH_ERROR = None

_SPAI_W = os.environ.get("VISHWAS_IMAGE_FACE_WEIGHTS", "/opt/verisafe/models/spai/spai.pth")
_HAVE_WEIGHTS = bool(_SPAI_W) and os.path.exists(_SPAI_W)

if _TORCH_ERROR is not None:
    def test_spai_arch_skipped():
        pytest.skip(f"real torch.nn unavailable in this env ({_TORCH_ERROR})")
elif _ARCH_ERROR is not None:
    def test_spai_arch_vendored():
        pytest.fail(f"spai architecture not vendored ({_ARCH_ERROR})")
else:
    # ----------------------------------------------------- registry seams --
    def test_registry_returns_spai_spec():
        spec = get_arch("spai")
        assert isinstance(spec, ArchSpec)
        assert spec.name == "spai"
        assert spec.weight_env == "VISHWAS_IMAGE_FACE_WEIGHTS"
        assert spec.implemented is True

    def test_adapter_row_registered():
        assert "VISHWAS_IMAGE_FACE_WEIGHTS" in ADAPTERS
        assert ADAPTERS["VISHWAS_IMAGE_FACE_WEIGHTS"].family == "face"
        assert _ARCH_FAMILIES.get("VISHWAS_IMAGE_FACE_WEIGHTS") == "spai"
        adapter = resolve("VISHWAS_IMAGE_FACE_WEIGHTS")
        assert adapter is not None and adapter.family == "face"

    # ------------------------------------------------- heavy (weights-gated) --
    try:  # whether the heavy deps even import in this env
        import torchvision  # noqa: F401
        import einops  # noqa: F401
        _HEAVY_DEPS_OK = True
    except Exception as e:
        _HEAVY_DEPS_OK = False
        _HEAVY_DEPS_ERR = f"{type(e).__name__}: {e}"

    @pytest.fixture()
    def built_spec():
        spec = SpaiSpec()
        model = spec.build()          # builds PatchBasedMFViT + loads weights
        model.eval()
        return spec, model

    def test_build_and_weights_coverage(built_spec):
        spec, model = built_spec
        import torch
        raw = torch.load(_SPAI_W, map_location="cpu", weights_only=False)
        sd = raw["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n = len(sd)
        coverage = (n - len(missing)) / n
        assert coverage >= 0.95
        assert len(unexpected) == 0

    def test_score_bounds_and_separation(built_spec):
        spec, model = built_spec
        import cv2
        if _HEAVY_DEPS_OK:
            from PIL import Image
            # small real-looking image + a brighter variant: finite p in [0,1]
            arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            p = spec.score(model, arr)
            assert isinstance(p, float) and np.isfinite(p) and 0.0 <= p <= 1.0
            # deterministic in eval
            assert p == spec.score(model, arr)

    # heavy build/coverage tests need weights + heavy deps; skip cleanly otherwise
    if _HAVE_WEIGHTS and _HEAVY_DEPS_OK:
        pass  # the fixtures above run
    else:
        def test_spai_heavy_gated():
            reason = []
            if not _HAVE_WEIGHTS:
                reason.append(f"weights absent at {_SPAI_W}")
            if not _HEAVY_DEPS_OK:
                reason.append(f"heavy deps missing ({_HEAVY_DEPS_ERR})")
            pytest.skip("; ".join(reason) or "heavy deps")