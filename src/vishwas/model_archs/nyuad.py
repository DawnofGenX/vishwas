"""ArchSpec for NYUAD-ComNets/NYUAD_AI-generated_images_detector (ViT-B/16, 3-class).

Second, INDEPENDENT content-based AI-image signal (2026-08-27). Complements SPAI
(spectral), which scores some flux-style AI images ~0.0 -> they read LOW. NYUAD
classifies {0: dalle, 1: real, 2: sd}; p_fake = p(dalle)+p(sd). Verified
discriminating on OUR corpus: catches flux_00/01/04/06/07 that SPAI misses (that
the degenerately-drifted dima806 ViTs could NOT).

Runs under the .venv-ambient tree (transformers 5.14.1 + torch 2.13): the webhook's
docling-python/transformers-5.15 stack cannot import any ViT (same GEN_EMAIL/
circular-import failure as the audio Wav2Vec2Model). image_facecheck shells out,
mirroring the aasist3 subprocess pattern.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from .base import ArchSpec
from ..device import resolve_device

_DEFAULT_WEIGHTS = "/opt/vishwas/models/nyuad_det"


class NyuadSpec(ArchSpec):
    name = "nyuad"
    weight_env = "VISHWAS_NYUAD_WEIGHTS"
    implemented = True

    def build(self) -> Any:
        from transformers import ViTForImageClassification
        ck = os.environ.get(self.weight_env, "") or _DEFAULT_WEIGHTS
        model_dir = ck if os.path.isdir(ck) else os.path.dirname(ck)
        model = ViTForImageClassification.from_pretrained(
            model_dir, local_files_only=True, use_safetensors=True)
        model.to(resolve_device())
        model.eval()
        return model

    def score(self, model: Any, x: Any) -> float:
        """AI-generated posterior p_fake in [0,1] = p(dalle) + p(sd)."""
        t = self._to_input_tensor(x)
        try:
            dev = next(model.parameters()).device
            t = t.to(dev)
        except (StopIteration, AttributeError):
            pass
        with torch.no_grad():
            logits = model(t).logits
            probs = torch.softmax(logits, dim=-1)[0]
        # id2label {0: dalle, 1: real, 2: sd} -> fake = dalle + sd
        return float((probs[0] + probs[2]).item())

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _to_input_tensor(x: Any) -> torch.Tensor:
        """Normalise any input to a (1,3,224,224) tensor: resize 224, rescale 1/255,
        normalize mean 0.5 / std 0.5 (the model's ViTFeatureExtractor spec)."""
        from PIL import Image
        if isinstance(x, Image.Image):
            img = x
        elif isinstance(x, (str, bytes)):
            img = Image.open(x)
        else:
            arr = np.asarray(x)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                f = arr.astype(np.float32)
                if f.max() <= 1.0 + 1e-6:
                    f = f * 255.0
                arr = np.clip(f, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr[..., ::-1].copy()).convert("RGB")
        img = img.convert("RGB").resize((224, 224))
        a = np.asarray(img).astype(np.float32) / 255.0
        a = (a - 0.5) / 0.5
        return torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0)


def get_arch() -> NyuadSpec:
    return NyuadSpec()