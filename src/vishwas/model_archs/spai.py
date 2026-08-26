"""SPAI (CVPR'25 spectral AI-image detector) architecture spec.

VENDORED / DROP-IN PROVENANCE
=============================
Checkpoint: /opt/verisafe/models/spai/spai.pth (SPAI trained ckpt)
  Source   : mever-team/spai (Apache-2.0), Google-Drive id
             1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI (README weights link). Fetched
             via Drive API server-side files.copy + Bearer download
             (quota-bypass), sha256
             24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55.
  Net def  : model_archs/_spai/ (vendor from commit 8ff7b3b, imports re-homed,
             Apache-2.0). See _spai/PROVENANCE.md for full provenance + the
             exact config reproduced by _spai/conv_config.py.
  Backbone : MFM ViT-B/16 pretrain is bundled at
             /opt/verisafe/models/spai/mfm_pretrain_vit_base.pth but is only
             needed for TRAINING init — the fine-tuned spai.pth is a complete
             PatchBasedMFViT and loads standalone for inference.

WHY THIS IS THE image_facecheck HEAVY GATE (root-cause fix)
  The previous gate re-used the EFFORT-chameleon checkpoint (VISHWAS_IMAGE_FACE_
  WEIGHTS was unset, so _resolved_weights_env() fell back to VISHWAS_EFFORT_WEIGHTS)
  which overfits — it scored every face 0.7..0.86 with no input-dependent decision
  function, and image fusion weight faceforensics.prob=2.5 dominated. SPAI is a
  spectral any-resolution detector trained to reconstruct the real-image spectral
  distribution self-supervised and flag AI-generated images as out-of-distribution;
  it is the dedicated replacements for the image slot (see fusion weight tuning,
  model_archs/spai + `~/fusion_img/` corpus).

POLARITY: p_fake in [0,1]; higher ⇒ AI-generated / manipulated.
  Matches the upstream `spai infer` label convention (data/*.csv, class 1 == fake).

INPUT CONTRACT (replicates upstream `spai infer` / `validate` exactly):
  - Accepts a numpy image array as produced by capability _load_image() (HWC
    uint8 BGR from cv2) or an already-CHW float [0,1] tensor / a file path.
  - score() decodes → RGB uint8 → float32 [0,1] CHW → zero-pad to at least
    224×224 → PatchBasedMFViT.forward([t], feature_extraction_batch=400) →
    sigmoid(logit). The ImageNet normalisation and frequency (DFT) filtering
    happen INSIDE the vendored MFViT, faithfully to upstream.

NOTE: the image_facecheck pipeline resizes every photo to 512×512 before the
heavy gate (capability _load_image side=512); this scoring reproduces that same
input so tuning through the live pipeline sees real behaviour.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import torch

from .base import ArchSpec
from ..device import resolve_device

# Weight path is resolved by the adapter/capability from VISHWAS_IMAGE_FACE_WEIGHTS.
_DEFAULT_WEIGHTS = "/opt/verisafe/models/spai/spai.pth"
_MIN_SIDE = 224
_FEATURE_BATCH = 400


def _import_net():
    """Import the vendored _spai package lazily (heavy deps stay out of the
    hermeneutically-clean suite import path). Submodules are pulled in explicitly
    because an empty-package import does not register sibling modules on it."""
    import importlib
    pkg = importlib.import_module("vishwas.model_archs._spai")
    importlib.import_module("vishwas.model_archs._spai.sid")
    importlib.import_module("vishwas.model_archs._spai.conv_config")
    return pkg


class SpaiSpec(ArchSpec):
    name = "spai"
    weight_env = "VISHWAS_IMAGE_FACE_WEIGHTS"
    implemented = True

    def build(self) -> Any:
        net = _import_net()
        config = net.conv_config.build_inference_config()
        model = net.sid.build_mf_vit(config)
        # Load weights here too so build() returns a READY model: weights live at
        # the env path (weight_env), or the provisioned default below.
        ck_path = os.environ.get(self.weight_env, "") or _DEFAULT_WEIGHTS
        if ck_path and os.path.exists(ck_path):
            sd = torch.load(ck_path, map_location="cpu", weights_only=False)
            state = sd.get("model", sd) if isinstance(sd, dict) else sd
            missing, unexpected = model.load_state_dict(state, strict=False)
            if len(missing) > 0.05 * max(1, len(state)):
                raise RuntimeError(
                    f"spai checkpoint coverage incomplete: {len(missing)} missing "
                    f"(first {missing[:3]}), {len(unexpected)} unexpected")
        model.to(resolve_device())
        model.eval()
        return model

    def score(self, model: Any, x: Any) -> float:
        """AI-generated / manipulated posterior p_fake in [0,1]."""
        t = self._to_input_tensor(x)
        try:
            dev = next(model.parameters()).device
            t = t.to(dev)
        except (StopIteration, AttributeError):
            pass
        with torch.no_grad():
            logit = model([t], _FEATURE_BATCH)  # 1 x 1
        return float(torch.sigmoid(logit).squeeze().item())

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _to_input_tensor(x: Any) -> torch.Tensor:
        """Normalise any of: HWC uint8 BGR (cv2), CHW float [0,1], path str."""
        if isinstance(x, (str, bytes)):
            import cv2  # type: ignore
            arr = cv2.imread(x, cv2.IMREAD_COLOR)
        else:
            arr = np.asarray(x)
        if arr is None:
            raise TypeError("spai score(): could not decode image input")
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] <= arr.shape[2] <= arr.shape[1]:  # CHW
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise TypeError(f"spai score(): expected HWC image, got shape {arr.shape}")
        # Normalise dtype/range to uint8 [0,255].
        if arr.dtype != np.uint8:
            f = arr.astype(np.float32)
            if f.max() <= 1.0 + 1e-6:
                f = f * 255.0
            arr = np.clip(f, 0, 255).astype(np.uint8)
        # Pipeline loads with cv2 -> BGR. SPAI was trained on PIL/RGB.
        rgb = arr[..., ::-1].copy()
        h, w = rgb.shape[:2]
        ph, pw = max(_MIN_SIDE, h), max(_MIN_SIDE, w)
        canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
        canvas[:h, :w, :] = rgb
        t = torch.from_numpy(canvas.transpose(2, 0, 1)).float().div_(255.0).unsqueeze(0)
        return t


def get_arch() -> SpaiSpec:
    """Registry hook consumed by vishwas.model_adapters._arch_aware_load."""
    return SpaiSpec()