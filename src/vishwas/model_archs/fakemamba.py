"""RawBMamba architecture spec — audio anti-spoofing (arXiv:2406.06086v2).

VENDORED / METHOD-PORT PROVENANCE
=================================
Checkpoint : /opt/vishwas/models/rawbmamba/rawbmamba_bset.pt
             sha256 8536477b86d16f4df281b2bda2cdd3046b7a2dd7b62d845f3541ea187fd289bd
Source     : https://github.com/cyjie429/RawBMamba (NO LICENSE FILE — evaluation
             use only until the author grants one; see PROVENANCE.md next to
             the checkpoint).
Paper      : RawBMamba — End-to-End Bidirectional State Space Model for Audio
             Deepfake Detection (2024). SNN-style raw-waveform frontend +
             bidirectional Mamba SSM backbone + linear head.

KEY-MAP / CONTRACT NOTES (honesty rule)
  A1. Vendored arch lives in ._rawbmamba_vendor/ (verbatim copy of the repo's
      mamba_ssm model files) with pure-PyTorch CPU shims replacing the CUDA
      kernels (selective_scan_cuda, causal_conv1d, triton layernorm). All
      deviations commented "CPU vendored:" in place.
  A2. Checkpoint loads STRICT (284/284 tensors, no remapping): top-level keys
      backbone.* + lm_head.* match MambaLMHeadModel attribute names exactly.
  A3. Input contract: raw waveform (B, 1, 64600); the repo pads/truncates to
      64600 samples at ~8 kHz (~8 s). score() repeat-pad/truncates to that.
  A4. Label order: repo score = logit[1] - logit[0] with class 1 = spoof in
      ASVspoof protocol (bonafide=0/spoof=1 as trained). We return the SPOOF
      posterior softmax(logits)[1] per the ArchSpec contract.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base import ArchSpec

_INPUT_SAMPLES = 64600


def _build_net() -> Any:
    """Lazy import so a broken vendor tree degrades to ArchNotImplementedError."""
    import torch.nn as nn  # noqa: F401  (ensures torch present)
    from ._rawbmamba_vendor.ac_mamba import MambaLMHeadModel
    from ._rawbmamba_vendor.config_mamba import MambaConfig

    return MambaLMHeadModel(config=MambaConfig(), device="cpu", dtype=torch.float32)


class RawBMambaSpec(ArchSpec):
    """Fills the 'fakemamba' adapter family (the pipeline's Mamba slot) with
    the one Mamba-family detector whose weights are actually obtainable."""

    name = "fakemamba"
    weight_env = "VISHWAS_FAKEMAMBA_WEIGHTS"
    implemented = True

    def build(self) -> Any:
        net = _build_net()
        net.eval()
        return net

    def apply_state(self, model: Any, sd: dict) -> bool:
        try:
            missing, unexpected = model.load_state_dict(sd, strict=True), None
            return True
        except Exception:
            return False

    def score(self, model: Any, x: Any) -> float:
        """Spoof posterior in [0,1] from a raw waveform (any sample rate the
        caller decoded; the model was trained on 8 kHz but is fed whatever the
        shared _waveform_preprocess produced, consistent with AASIST usage)."""
        if isinstance(x, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        elif isinstance(x, torch.Tensor):
            wav = x.float()
        else:
            raise TypeError(f"rawbmamba score() expects a waveform array, got {type(x).__name__}")
        wav = wav.reshape(-1)
        if wav.numel() < _INPUT_SAMPLES:
            reps = _INPUT_SAMPLES // wav.numel() + 1
            wav = wav.repeat(reps)[:_INPUT_SAMPLES]
        else:
            wav = wav[:_INPUT_SAMPLES]
        wav = wav.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        with torch.no_grad():
            out = model(wav)
        logits = out[0] if isinstance(out, tuple) else out
        probs = F.softmax(logits, dim=-1)
        return float(probs[0, 1].item())  # index 1 == spoof


def get_arch() -> RawBMambaSpec:
    """Registry hook for vishwas.model_adapters._arch_aware_load."""
    return RawBMambaSpec()
