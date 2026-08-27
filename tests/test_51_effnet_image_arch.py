"""Fine-tuned EfficientNet image checkpoint wiring."""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
    from PIL import Image
    from vishwas.model_adapters import ArchModelWrapper, resolve
    from vishwas.model_archs import get_arch
    from vishwas.model_archs.effnet_image import EfficientNetImageSpec
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment-dependent
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


if _IMPORT_ERROR is not None:
    def test_effnet_image_arch_skipped():
        pytest.skip(f"real torch/Pillow unavailable ({_IMPORT_ERROR})")
else:
    class _TinyBrightnessNet(nn.Module):
        """Small deterministic surrogate for architecture plumbing tests."""

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(12.0))
            self.bias = nn.Parameter(torch.tensor(-6.0))

        def forward(self, x):
            return x.mean(dim=(1, 2, 3), keepdim=True) * self.scale + self.bias

    def _checkpoint(model):
        return {
            "model_state": model.state_dict(),
            "arch": "efficientnet_b0",
            "img_size": 16,
            "positive_class": "ai_generated",
            "normalize": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        }

    def test_registry_returns_effnet_spec():
        spec = get_arch("efficientnet_b0")
        assert isinstance(spec, EfficientNetImageSpec)
        assert spec.weight_env == "VISHWAS_IMAGE_FACE_WEIGHTS"

    def test_checkpoint_load_score_bounds_and_separation(tmp_path, monkeypatch):
        from vishwas.model_archs import effnet_image

        monkeypatch.setattr(effnet_image, "_build_network", _TinyBrightnessNet)
        path = tmp_path / "image_detector.pt"
        torch.save(_checkpoint(_TinyBrightnessNet()), path)
        monkeypatch.setenv("VISHWAS_IMAGE_FACE_WEIGHTS", str(path))

        adapter = resolve("VISHWAS_IMAGE_FACE_WEIGHTS")
        loaded = adapter.load(str(path))
        assert isinstance(loaded, ArchModelWrapper)
        assert loaded.spec.name == "efficientnet_b0"

        dark = tmp_path / "real.png"
        bright = tmp_path / "fake.png"
        Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(dark)
        Image.fromarray(np.full((20, 30, 3), 255, dtype=np.uint8)).save(bright)
        p_real, p_fake = loaded.score(dark), loaded.score(bright)
        assert 0.0 <= p_real < p_fake <= 1.0

    def test_rejects_wrong_positive_class(tmp_path, monkeypatch):
        from vishwas.model_archs import effnet_image

        monkeypatch.setattr(effnet_image, "_build_network", _TinyBrightnessNet)
        checkpoint = _checkpoint(_TinyBrightnessNet())
        checkpoint["positive_class"] = "real"
        path = tmp_path / "wrong_polarity.pt"
        torch.save(checkpoint, path)
        monkeypatch.setenv("VISHWAS_IMAGE_FACE_WEIGHTS", str(path))
        assert resolve("VISHWAS_IMAGE_FACE_WEIGHTS").load(str(path)) is None

    def test_rejects_missing_preprocessing_metadata(tmp_path, monkeypatch):
        from vishwas.model_archs import effnet_image

        monkeypatch.setattr(effnet_image, "_build_network", _TinyBrightnessNet)
        checkpoint = _checkpoint(_TinyBrightnessNet())
        checkpoint.pop("normalize")
        path = tmp_path / "missing_metadata.pt"
        torch.save(checkpoint, path)
        monkeypatch.setenv("VISHWAS_IMAGE_FACE_WEIGHTS", str(path))
        assert resolve("VISHWAS_IMAGE_FACE_WEIGHTS").load(str(path)) is None

    def test_image_capability_emits_ai_probability(tmp_path, monkeypatch):
        from vishwas.capabilities import image_facecheck
        from vishwas.events import Artifact, InputType, JobContext, MediaKind
        from vishwas.model_archs import effnet_image

        monkeypatch.setattr(effnet_image, "_build_network", _TinyBrightnessNet)
        weights = tmp_path / "image_detector.pt"
        torch.save(_checkpoint(_TinyBrightnessNet()), weights)
        monkeypatch.setenv("VISHWAS_IMAGE_FACE_WEIGHTS", str(weights))
        bright = np.full((24, 32, 3), 255, dtype=np.uint8)
        monkeypatch.setattr(image_facecheck, "_load_image", lambda _p: bright)
        monkeypatch.setattr(image_facecheck, "_load_original_image", lambda _p: bright)

        image_path = tmp_path / "input.png"
        image_path.touch()
        art = Artifact(image_path, "input.png", InputType.IMAGE,
                       verified_kind=MediaKind.PNG, size_bytes=1)
        ctx = JobContext("test", art, tmp_path)
        checks = image_facecheck.ImageFaceCheckCapability().analyze(art, ctx)
        learned = next(c for c in checks if c.name == "image_face_forensics")
        assert learned.status == "ok"
        assert 0.9 < learned.signals["prob_deepfake"] <= 1.0
