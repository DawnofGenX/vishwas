"""EFFORT (Effort-AIGI-Detection) architecture spec — Phase 1 Task 1.3.

VENDORED / METHOD-PORT PROVENANCE
=================================
Checkpoint(s): /opt/vishwas/models/effort/{chameleon,ffpp,genimage}/effort_*.pth
  HF card : YZY-stack/Effort-AIGI-Detection
  Source  : https://github.com/YZY-stack/Effort-AIGI-Detection (CC BY-NC 4.0)
            arXiv 2411.15633 (ICML 2025 oral). Three ~1.16 GB checkpoints,
            identical architecture, different training domains:
              chameleon -> face-region / spatial forensics   (PRIMARY here)
              ffpp      -> face forensics (FF++ domain)
              genimage  -> general AIGC image detection
  License : CC BY-NC 4.0 — NON-COMMERCIAL. Vishwas is a local, non-commercial
            verification tool; the opt-in gate (VISHWAS_EFFORT_WEIGHTS must be
            explicitly set) is the license boundary. See GAPS_AND_ENABLEMENT.md.

ARCHITECTURE (reconstructed from probed tensor shapes — verified, not guessed):
  A complete CLIP-style Vision Transformer, ViT-L/14:
    * patch_embedding : Conv2d(3, 1024, kernel=14, stride=14)  -> 224/14 = 16x16 = 256 patches
    * class_embedding : (1024,)                                -> 1 CLS token
    * position_embedding: (257, 1024)                          -> 256 patches + 1 CLS
    * pre_layrnorm    : LayerNorm(1024)   [checkpoint typo 'layr' preserved]
    * encoder.layers[0..23]: 24 x TransformerBlock
        - self_attn.{q,k,v,out}_proj : OrthAlign Linear(1024,1024)
        - layer_norm1 / mlp.fc1(1024->4096) / mlp.fc2(4096->1024) / layer_norm2
    * post_layernorm  : LayerNorm(1024)
    * head            : Linear(1024, 2)                        -> [real, fake] logits
  Total ~303M params, ALL embedded in the checkpoint (no external CLIP weights).

ORTHALIGN (the signature EFFORT component):
  Each self-attention projection is a base linear ``weight_main`` PLUS a
  rank-1 low-rank residual learned via subspace decomposition:
      W_eff = weight_main + S_residual * (U_residual @ V_residual)
  where S_residual is a scalar (1,), U_residual (1024,1), V_residual (1,1024).
  U@V is (1024,1024); added to weight_main (1024,1024). This is applied to q,
  k, v AND out_proj. It is the "orthogonal alignment" that lets EFFORT detect
  subtle generative artifacts while staying robust to compression.

KEY-MAP ASSUMPTIONS (honesty rule — every non-obvious reconstruction choice):
  E1. Checkpoint keys carry a leading ``module.`` (DataParallel training
      wrapper). apply_state() strips it before load_state_dict.
  E2. The pre-norm typo ``pre_layrnorm`` (not 'pre_layernorm') is preserved
      VERBATIM in the skeleton so strict=False sees a 1:1 key match.
  E2b. Two further verbatim-shape details, both confirmed by probing the
      checkpoint's exact key set (681 keys):
        - ``patch_embedding`` is a Conv2d with NO bias (checkpoint has only
          'patch_embedding.weight', no '.bias') -> built with bias=False.
        - ``position_embedding`` is an nn.Embedding(257,1024) so its state-dict
          key is 'position_embedding.weight' (a bare Parameter would expose
          'position_embedding' and mismatch).
  E3. Input contract: score() expects a (3,H,W) or (N,3,H,W) float array in
      [0,1] (CHW), as produced by the adapter's _img_resize_chw (center-crop
      square -> 224x224). It is ImageNet-normalised inside score() (mean
      [0.485,0.456,0.406], std [0.229,0.224,0.225]) — the standard CLIP/ViT
      normalisation. If the original trained on a different normalisation the
      posterior shifts but stays monotone; documented, not silently assumed.
  E4. Label order: head logits are [real, fake]. We return the FAKE posterior
      softmax(logits)[1] in [0,1]. (Empirically confirmed against a known-real
      natural image during the task-1.3 smoke run; see ARCH_VENDOR_EVIDENCE.)
  E5. Inference runs eval() + no_grad. No BatchNorm in this net (all LayerNorm),
      so no running-stat concerns.
  E6. Primary checkpoint = chameleon (face/spatial forensics) per Decision #2.
      ffpp/genimage share the identical arch and load through the same spec if
      pointed at by the env var.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ArchSpec
from ..device import resolve_device

# NORMALISATION CONSTANTS -- CLIP's, NOT ImageNet's.
#
# EFFORT's backbone is a CLIP ViT-L/14, and the official implementation
# specifies these exact values in
# DeepfakeBench/training/config/detector/effort.yaml:
#     mean: [0.48145466, 0.4578275, 0.40821073]
#     std:  [0.26862954, 0.26130258, 0.27577711]
#
# The repo previously used ImageNet's constants ([0.485,0.456,0.406] /
# [0.229,0.224,0.225]), which is a train/test mismatch: the ImageNet std is
# ~17% smaller on channel 0, so the input was amplified relative to
# training. The names are kept for continuity with the surrounding code,
# but the VALUES are CLIP's.
#
# Measured on the leak-free 60-clip corpus (see
# tools/eval_effort_norm_channel.py):
#     (BGR, ImageNet)  <- before this fix    AUC 0.6511
#     (BGR, CLIP)                            AUC 0.6278
#     (RGB, ImageNet)                        AUC 0.7211
#     (RGB, CLIP)      <- upstream           AUC 0.7189
# The channel swap is the dominant effect; the constants are worth ~0.002
# either way on this corpus, but they are still wrong and the upstream
# config is unambiguous.
_IMAGENET_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGENET_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class _OrthAlignLinear(nn.Module):
    """Base linear + rank-1 OrthAlign residual (see module docstring)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight_main = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        # Residuals start at zero so W_eff == weight_main at init (clean
        # random-init behaviour; real checkpoints overwrite all five).
        self.S_residual = nn.Parameter(torch.zeros(1))
        self.U_residual = nn.Parameter(torch.zeros(dim, 1))
        self.V_residual = nn.Parameter(torch.zeros(1, dim))
        nn.init.normal_(self.weight_main, std=dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_main + self.S_residual * (self.U_residual @ self.V_residual)
        return F.linear(x, w, self.bias)


class _SelfAttn(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = _OrthAlignLinear(dim)
        self.k_proj = _OrthAlignLinear(dim)
        self.v_proj = _OrthAlignLinear(dim)
        self.out_proj = _OrthAlignLinear(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q_proj(x).view(b, n, self.heads, c // self.heads).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.heads, c // self.heads).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.heads, c // self.heads).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, c)
        return self.out_proj(attn)


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.self_attn = _SelfAttn(dim, heads)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _EffortNet(nn.Module):
    """CLIP-style ViT-L/14 + OrthAlign self-attn + binary head.

    Top-level attribute names mirror the checkpoint's state-dict prefixes
    (backbone.embeddings.* / backbone.encoder.layers.* / backbone.post_layernorm.*
    / head.*) so apply_state() sees a 1:1 key match after the 'module.' strip.
    """

    def __init__(self) -> None:
        super().__init__()
        dim, heads, mlp_dim, depth = 1024, 16, 4096, 24
        self.backbone = nn.Module()
        emb = nn.Module()
        emb.class_embedding = nn.Parameter(torch.zeros(dim))
        emb.patch_embedding = nn.Conv2d(3, dim, kernel_size=14, stride=14, bias=False)
        emb.position_embedding = nn.Embedding(257, dim)   # -> 'position_embedding.weight' (E2b)
        self.backbone.embeddings = emb
        # NOTE: 'pre_layrnorm' typo preserved verbatim (E2).
        self.backbone.pre_layrnorm = nn.LayerNorm(dim)
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.layers = nn.ModuleList([_Block(dim, heads, mlp_dim) for _ in range(depth)])
        self.backbone.post_layernorm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) float32 (ImageNet-normalised) -> logits (B, 2)."""
        b, _, h, w = x.shape
        g = h // 14
        x = self.backbone.embeddings.patch_embedding(x)          # (B,1024,16,16)
        x = x.flatten(2).transpose(1, 2)                         # (B,256,1024)
        cls = self.backbone.embeddings.class_embedding.unsqueeze(0).unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)                           # (B,257,1024)
        x = x + self.backbone.embeddings.position_embedding.weight   # (257,1024) broadcast
        x = self.backbone.pre_layrnorm(x)
        for layer in self.backbone.encoder.layers:
            x = layer(x)
        x = self.backbone.post_layernorm(x)
        x = x[:, 0]                                              # CLS token
        return self.head(x)


class EffortSpec(ArchSpec):
    name = "effort"
    weight_env = "VISHWAS_EFFORT_WEIGHTS"
    #: Distinguishes a vendored arch from a stub (havic still carries False).
    implemented = True

    def build(self) -> _EffortNet:
        net = _EffortNet()
        net.to(resolve_device())  # no-op on CPU-only hosts
        net.eval()
        return net

    def apply_state(self, model: _EffortNet, sd: Dict[str, Any]) -> bool:
        """Strip the DataParallel 'module.' prefix (E1), then delegate."""
        if not isinstance(sd, dict) or not sd:
            return super().apply_state(model, sd)  # base records + returns False
        stripped = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        return super().apply_state(model, stripped)

    def score(self, model: _EffortNet, x: Any) -> float:
        """Fake/AIGC posterior in [0,1] (label order [real, fake] — E4).

        INPUT CONTRACT: `x` is BGR, HWC uint8 or CHW float in [0,1] — the
        order `cv2.imread` produces. This method converts BGR -> RGB
        internally, because EFFORT is a CLIP ViT-L/14 and was trained on
        RGB.

        2026-10-21: the conversion was MISSING. `_img_resize_chw` (the
        shared adapter preprocess) transposes HWC->CHW but never swaps
        channels, so the model was fed blue-red-swapped pixels and then
        normalised them with RGB-ordered statistics. Measured on the
        leak-free 60-clip corpus, the swap alone was worth **+0.070 AUC**
        (0.6511 -> 0.7211). Every sibling arch already does this --
        `spai.py` ("SPAI was trained on PIL/RGB"), `nyuad.py`, and
        `effnet_image.py` ("decodes through cv2, hence BGR -> RGB") -- so
        EFFORT was the only one relying on the shared preprocess.

        Normalisation is CLIP's, per the official implementation
        (DeepfakeBench/training/config/detector/effort.yaml):
            mean: [0.48145466, 0.4578275, 0.40821073]
            std:  [0.26862954, 0.26130258, 0.27577711]
        The repo previously used ImageNet's constants, which is a second
        (smaller, ~0.002 AUC) train/test mismatch.
        """
        if isinstance(x, (str, bytes)):
            raise TypeError(f"effort score() expects a numeric array-like, got {type(x).__name__}")
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[None]                       # (1,3,224,224)
        elif arr.ndim != 4:
            raise TypeError(f"effort score() expects (3,H,W) or (N,3,H,W), got shape {arr.shape}")
        # BGR -> RGB (the model was trained on RGB).
        #
        # AXIS MATTERS. `arr[..., ::-1]` reverses the LAST axis, which is the
        # CHANNEL axis only for an HWC array. The adapter's preprocess for
        # this family is `_img_resize_chw`, which returns CHW -- channels at
        # axis -3 -- so the swap must be on axis -3.
        #
        # SPAI's `arr[..., ::-1]` is correct because ITS array is HWC.
        # Copying that line into a CHW path silently reverses the WIDTH
        # instead of the channels, which produces plausible-looking garbage
        # rather than an error.
        #
        # By the time we get here `arr` is always 4-D (the 3-D case above was
        # promoted with arr[None]), so the swap is on axis 1 of the 4-D
        # array -- the channel axis of (N, C, H, W).
        if arr.shape[1] == 3:
            arr = arr[:, ::-1, :, :].copy()
        elif arr.shape[-3] == 3:
            # (3, H, W) that was not promoted above -- defensive.
            arr = arr[None, ::-1, :, :].copy()
        # CLIP normalise (upstream effort.yaml).
        arr = (arr - _IMAGENET_MEAN[None, :, None, None]) / _IMAGENET_STD[None, :, None, None]
        xt = torch.from_numpy(np.ascontiguousarray(arr))
        try:
            dev = next(model.parameters()).device
            xt = xt.to(dev)   # follow wherever build() placed the model
        except StopIteration:
            pass              # parameterless stub: nothing to align
        with torch.no_grad():
            logits = model(xt)
        probs = F.softmax(logits, dim=-1)
        return float(probs[:, 1].mean().item())   # index 1 == fake class


def get_arch() -> EffortSpec:
    """Registry hook consumed by ``vishwas.model_adapters._arch_aware_load``."""
    return EffortSpec()
