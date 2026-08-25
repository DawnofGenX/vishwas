"""AASIST (HABLA_WavLM_AASIST variant) architecture spec — Phase 1 Task 1.2.

VENDORED / METHOD-PORT PROVENANCE
=================================
Checkpoint: /opt/vishwas/models/aasist/best_model.pth
  HF card : DeepFense/HABLA_WavLM_AASIST_NoAug_Seed42
  Source  : https://github.com/Yaselley/deepfense-framework (Apache-2.0)
            - models/frontends/wavlm.py        -> WavLMWrapper (source="unil")
            - models/modules/wavlm/{wavlm,modules}.py -> fairseq-style WavLM
            - models/backends/aasist.py        -> AASIST (HtrgGAT trunk)
            - models/losses/cross_entropy.py   -> CrossEntropy (fc 160->2)
  Config  : HF repo config.yaml (exp_name Hable_wavlm_AASIST_NoAug_seed42)
            sampling_rate 16000; label_map {bonafide: 1, spoof: 0};
            train/val transform: pad max_len 64000 pad_type repeat;
            frontend wavlm ckpt WavLM-Large.pt freeze=false source unil;
            backend AASIST (default dims); loss CrossEntropy emb 160 n_classes 2.

NOTE ON THE PLAN'S ASSUMPTION: the plan described "classic CNN-BiLSTM AASIST"
with a mel front-end. The provisioned checkpoint is NOT that model — it is the
DeepFense HABLA variant: fairseq-style WavLM-Large (raw waveform in) + custom
HtrgGAT spectro-temporal graph-attention backend + Linear(160->2) head. The
mel preprocessor (_mel_aasist_preprocess) is therefore the WRONG front-end for
this checkpoint; the correct one is raw 16 kHz waveform repeat-padded to
64000 samples (4 s), exactly as trained. See KEY-MAP ASSUMPTIONS below.

KEY-MAP ASSUMPTIONS (honesty rule — every non-obvious reconstruction choice):
  A1. Frontend = verbatim copy of deepfense models/modules/wavlm (Apache-2.0),
      re-homed under vishwas.model_archs._wavlm with only the import path
      changed. No behavioural modification.
  A2. WavLM-Large hyperparameters are taken from the checkpoint shapes
      (encoder_layers=24, embed=1024, ffn=4096, heads=16, conv_pos=128/16
      groups, gru_rel_pos=True, num_buckets=320, max_distance=1280,
      conv_feature_layers=[(512,10,5)]+[(512,3,2)]*4+[(512,2,2)]*2) — verified
      against probed tensor shapes, not guessed.
  A3. Backend = verbatim copy of deepfense models/backends/aasist.py
      (Apache-2.0) with framework imports stripped; default dims
      (filts [70,[1,32],[32,32],[32,64],[64,64]], gat_dims [64,32],
      pool_ratios [0.5,0.7,0.5,0.5], temperatures [2,2,100,100],
      input_dim 1024) match the checkpoint exactly.
  A4. Head = nn.Linear(160, 2) mounted as self.losses[0].fc to mirror the
      training wrapper's 'losses.0.fc.*' keys (checkpoint prefix preserved).
  A5. Input contract: score() expects a 1-D float32 waveform (any length);
      it is repeat-padded/truncated to exactly 64000 samples (4 s @ 16 kHz),
      matching the training transform (pad max_len 64000, pad_type repeat).
      The adapter preprocessor (_aasist_wav_preprocess) does this up front.
  A6. Label order from config.yaml: bonafide=1, spoof=0. The head's
      get_score() returns logits[:, bonafide] (an LLR), but per the ArchSpec
      contract we return the SPOOF posterior: softmax(logits)[0] in [0,1].
  A7. Inference runs eval() + no_grad; BatchNorm running stats come from the
      checkpoint (num_batches_tracked present for all BN layers).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ArchSpec
from ..device import resolve_device
from ._aasist_backend import AASIST as _AASISTBackend
from ._wavlm.wavlm import WavLM, WavLMConfig

# Training-time input length: 4 s @ 16 kHz (config.yaml pad max_len 64000).
_INPUT_SAMPLES = 64000


def _wavlm_large_config() -> WavLMConfig:
    """WavLM-Large hyperparameters, verified against checkpoint shapes (A2)."""
    return WavLMConfig({
        "extractor_mode": "default",
        "encoder_layers": 24,
        "encoder_embed_dim": 1024,
        "encoder_ffn_embed_dim": 4096,
        "encoder_attention_heads": 16,
        "activation_fn": "gelu",
        "layer_norm_first": False,
        "conv_feature_layers": "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        "conv_bias": False,
        "feature_grad_mult": 1.0,
        "normalize": False,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "encoder_layerdrop": 0.0,
        "dropout_input": 0.0,
        "dropout_features": 0.0,
        "mask_length": 10,
        "mask_prob": 0.0,          # inference-only usage; keep masking off
        "mask_selection": "static",
        "mask_other": 0,
        "no_mask_overlap": False,
        "mask_min_space": 1,
        "mask_channel_length": 10,
        "mask_channel_prob": 0.0,
        "mask_channel_selection": "static",
        "mask_channel_other": 0,
        "no_mask_channel_overlap": False,
        "mask_channel_min_space": 1,
        "conv_pos": 128,
        "conv_pos_groups": 16,
        "relative_position_embedding": True,
        "num_buckets": 320,
        "max_distance": 1280,
        "gru_rel_pos": True,
    })


class _Head(nn.Module):
    """Mirrors the training wrapper's 'losses.0.fc.*' checkpoint keys."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(160, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _AASISTNet(nn.Module):
    """frontend(WavLM-Large) -> backend(HtrgGAT) -> losses[0].fc(160->2).

    Top-level attribute names mirror the checkpoint's state-dict prefixes
    (frontend.model.* / backend.* / losses.0.fc.*) so apply_state() sees a
    1:1 key match.
    """

    def __init__(self) -> None:
        super().__init__()
        # WavLMWrapper exposes only self.model -> checkpoint 'frontend.model.*'
        self.frontend = nn.ModuleDict({"model": WavLM(_wavlm_large_config())})
        self.backend = _AASISTBackend({})  # default dims == checkpoint dims (A3)
        self.losses = nn.ModuleList([_Head()])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T_samples) float32 @16 kHz -> logits (B, 2)."""
        feats, _ = self.frontend["model"].extract_features(wav, mask=False)
        emb = self.backend(feats)
        return self.losses[0](emb)


class AASISTSpec(ArchSpec):
    name = "aasist"
    weight_env = "VISHWAS_AASIST_WEIGHTS"
    #: Distinguishes a vendored arch from a stub (effort/havic carry False).
    implemented = True

    def build(self) -> _AASISTNet:
        net = _AASISTNet()
        net.to(resolve_device())  # no-op on CPU-only hosts
        net.eval()
        return net

    def score(self, model: _AASISTNet, x: Any) -> float:
        """Spoof posterior in [0,1] (label_map: bonafide=1, spoof=0 — A6)."""
        if isinstance(x, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        elif isinstance(x, torch.Tensor):
            wav = x.float()
        else:
            raise TypeError(f"aasist score() expects a waveform array, got {type(x).__name__}")
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        # Repeat-pad / truncate to the trained 4 s window (A5).
        if wav.numel() < _INPUT_SAMPLES:
            reps = _INPUT_SAMPLES // wav.numel() + 1
            wav = wav.repeat(reps)[:_INPUT_SAMPLES]
        else:
            wav = wav[:_INPUT_SAMPLES]
        wav = wav.unsqueeze(0)  # (1, 64000)
        try:
            dev = next(model.parameters()).device
            wav = wav.to(dev)   # follow wherever build() placed the model
        except StopIteration:
            pass                # parameterless stub: nothing to align
        with torch.no_grad():
            logits = model(wav)
        probs = F.softmax(logits, dim=-1)
        return float(probs[0, 0].item())  # index 0 == spoof class


def _aasist_wav_preprocess(raw: Any) -> np.ndarray:
    """Adapter preprocessor: WAV path or waveform -> 1-D float32 @16 kHz.

    Reuses the shared _waveform_preprocess (ffmpeg decode to 16 kHz mono PCM,
    [-1,1]); the fixed-length repeat-pad to 64000 samples happens inside
    AASISTSpec.score (single source of truth for the input contract).
    """
    from ..model_adapters import _waveform_preprocess
    return _waveform_preprocess(raw)


def get_arch() -> AASISTSpec:
    """Registry hook consumed by ``vishwas.model_adapters._arch_aware_load``."""
    return AASISTSpec()
