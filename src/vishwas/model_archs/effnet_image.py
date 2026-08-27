"""EfficientNet-B0 adapter for Vishwas fine-tuned image checkpoints.

The checkpoint contract is owned by ``ml/image_finetune/train_image_detector.py``:
``model_state`` contains an EfficientNet-B0 with a one-logit classifier,
``positive_class`` is ``ai_generated``, and ``img_size`` / ``normalize`` hold
the exact evaluation preprocessing.  Higher scores therefore always mean a
higher probability that the image is AI-generated.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import numpy as np

from .base import ArchSpec
from ..device import resolve_device

_DEFAULT_SIZE = 224
_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_STD = (0.229, 0.224, 0.225)


def _build_network():
    """Build the exact one-logit network used by the fine-tuning pipeline."""
    import torch.nn as nn
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    return model


class EfficientNetImageSpec(ArchSpec):
    name = "efficientnet_b0"
    weight_env = "VISHWAS_IMAGE_FACE_WEIGHTS"
    implemented = True

    def __init__(self) -> None:
        super().__init__()
        self.img_size = _DEFAULT_SIZE
        self.mean = _DEFAULT_MEAN
        self.std = _DEFAULT_STD

    def configure_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        """Adopt and validate preprocessing metadata before building/scoring."""
        if checkpoint.get("arch") != self.name:
            raise ValueError("checkpoint is not an efficientnet_b0 image detector")
        if checkpoint.get("positive_class") != "ai_generated":
            raise ValueError("checkpoint positive_class must be ai_generated")

        try:
            size = int(checkpoint["img_size"])
            norm = checkpoint["normalize"]
            mean = tuple(float(v) for v in norm["mean"])
            std = tuple(float(v) for v in norm["std"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint is missing preprocessing metadata") from exc
        if size <= 0 or len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError("invalid image preprocessing metadata")
        self.img_size, self.mean, self.std = size, mean, std

    def build(self) -> Any:
        model = _build_network()
        model.to(resolve_device())
        model.eval()
        return model

    def score(self, model: Any, x: Any) -> float:
        """Return P(ai_generated) for a path or pipeline BGR image array."""
        import torch

        tensor = self._to_input_tensor(x)
        try:
            tensor = tensor.to(next(model.parameters()).device)
        except (StopIteration, AttributeError):
            pass
        model.eval()
        with torch.no_grad():
            logit = model(tensor)
            return float(torch.sigmoid(logit.reshape(-1)[0]).item())

    def _to_input_tensor(self, x: Any):
        import torch
        from PIL import Image

        if isinstance(x, (str, bytes, os.PathLike)):
            with Image.open(os.fspath(x)) as opened:
                image = opened.convert("RGB")
                image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                arr = np.asarray(image, dtype=np.float32) / 255.0
        else:
            arr = np.asarray(x)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if (arr.ndim == 3 and arr.shape[0] in (1, 3)
                    and arr.shape[2] not in (1, 3, 4)):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
                raise TypeError(f"efficientnet score(): expected an image, got {arr.shape}")
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            # ImageFaceCheckCapability decodes through cv2, hence BGR -> RGB.
            arr = arr[:, :, ::-1]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if arr.max(initial=0.0) <= 1.0 + 1e-6:
                    arr *= 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, mode="RGB")
            image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.0

        chw = np.transpose(arr, (2, 0, 1)).copy()
        t = torch.from_numpy(chw).float()
        mean = torch.tensor(self.mean, dtype=t.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=t.dtype).view(3, 1, 1)
        return ((t - mean) / std).unsqueeze(0)


def get_arch() -> EfficientNetImageSpec:
    return EfficientNetImageSpec()
