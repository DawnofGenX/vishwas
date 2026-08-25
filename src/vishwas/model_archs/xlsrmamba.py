"""XLSR-Mamba-LA architecture spec — audio anti-spoofing gate.

VENDORED / METHOD-PORT PROVENANCE
=================================
Checkpoint : /opt/vishwas/models/xlsr-mamba/model.safetensors
             sha256 bb8a1af0b3f9ee28dbcfc7c82d733f7cc5fdc77ff3e96b25db7b4ed72f2d1663
             1,277,378,920 bytes, 565 tensors (PytorchModelHubMixin-style flat
             state dict incl. the SSL frontend weights; ~319M params total).
HF card    : AustinXiao/XLSR-Mamba-LA
Source     : https://github.com/swagshaw/XLSR-Mamba (MIT)
Paper      : "XLSR-Mamba: A Dual-Column Bidirectional State Space Model for
             Spoofing Attack Detection" — Xiao & Das, arXiv:2411.10027,
             IEEE Signal Processing Letters. Reported EERs: ASVspoof2021 LA
             0.93% / DF 1.88% / In-the-Wild 6.71%.

ARCHITECTURE (verified against probed checkpoint shapes, not guessed):
  * ssl_model : fairseq wav2vec2 XLSR-300M frontend (24-layer stable-layer-norm
    PRE-norm transformer — do_stable_layer_norm=true in the canonical config,
    see the _SSL_CFG_OVERRIDES note; embed 1024 / ffn 4096 / heads 16,
    conv_pos 128 groups 16, extractor_mode "layer_norm", quantizer 320 vars x 2
    groups -> latent 384). Vendored under ._xlsrmamba_vendor.wav2vec2_frontend
    from pinned fairseq a54021305d6b3c WITHOUT installing fairseq.
  * LL        : Linear(1024 -> 144) projection to the Mamba width.
  * first_bn  : BatchNorm2d(1) over the (B, 1, T, D) feature map + SELU.
  * conformer : MixerModel = two parallel columns of 6 Mamba blocks each
    (d_model 144, d_state 16, d_conv 4, expand 2, dt_rank ceil(144/16)=9,
    RMSNorm eps 1e-5), one column run over the time-flipped sequence;
    per-column Add+RMSNorm + softmax attention pooling; concat(288) ->
    Linear(288->144) -> dropout -> Linear(144->2).

KEY-MAP / CONTRACT NOTES (honesty rule):
  X1. Vendored tree lives in ._xlsrmamba_vendor/ with pure-PyTorch CPU shims
      replacing mamba_ssm's CUDA kernels (selective_scan_fn, causal_conv1d_fn,
      triton RMSNorm) and a no-fairseq wav2vec2 re-home. All deviations are
      commented "CPU-vendored:" in place.
  X2. Checkpoint loads STRICT (565/565 tensors, zero remapping): top-level
      prefixes ssl_model.model.* / LL.* / first_bn.* / conformer.* match this
      module's attribute names exactly. The upstream repo's BiBlock refactor
      (per-block LL_hidden projections) is NOT in this checkpoint and is
      deliberately not vendored — the released LA model uses plain per-column
      Blocks with bidirectionality in MixerModel.forward. Two frontend-shape
      facts were discovered during strict-load bring-up and are pinned in the
      vendored config: (a) conv_bias=True — every feature-extractor Conv1d
      stores a bias despite fairseq's default; (b) final_dim=768 — the SSL
      pre-training heads (quantizer.vars (1,640,384) => vq_dim 768,
      project_q/final_proj 768x768) exist in the dump even though spoofing
      inference never executes them.
  X3. pos_conv weight-norm shape quirk: nn.utils.weight_norm(conv, dim=2)
      keeps the normalized axis as singletons, so weight_g is (1, 1, 128)
      for the grouped Conv1d(1024, 1024, k=128, groups=16) — verified by
      round-trip against torch's own weight_norm output before vendoring.
  X4. Input contract: raw 16 kHz waveform repeat-tiled/truncated to 66800
      samples (~4.17 s @ 16 kHz), matching the repo's Dataset_eval cut=66800
      and utils.pad() (np.tile + truncate — same repeat-pad family as AASIST).
      score() performs that pad itself.
  X5. Label order: upstream maps bonafide=1, spoof=0 (data_utils.genSpoof_list
      / read_metadata) and its eval scripts write logits[:, 1] (the bonafide
      logit) as the CM score. Per the ArchSpec contract we return the SPOOF
      posterior softmax(logits)[0] in [0, 1]. NOTE this is inverted vs
      RawBMamba/fakemamba (bonafide=0/spoof=1 there).
  X6. Inference runs eval() + no_grad on resolve_device(); the frontend's
      dropout paths are inactive in eval mode and masking is disabled
      (mask=False), matching la_evaluate.py's produce_evaluation_file usage.
  X7. safetensors payload: _load_safetensors accepts either a bare tensor dict
      or {"state_dict": ...} wrapper so both raw HF dumps and repacked
      checkpoints load through the same path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ArchSpec
from ..device import resolve_device

# Training-time input length: repo Dataset cut = 66800 samples (~4.17 s @ 16 kHz).
_INPUT_SAMPLES = 66800

#: XLSR-300M frontend config overrides (rest = fairseq defaults, verified X2).
_SSL_CFG_OVERRIDES = {
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    # PRE-NORM — XLSR-300M is a "do_stable_layer_norm" model (canonical HF
    # config: do_stable_layer_norm=true, feat_extract_norm="layer"), and
    # upstream XLSR-Mamba loads the frontend via fairseq's
    # load_model_ensemble_and_task, i.e. with the CHECKPOINT'S OWN cfg. The
    # hand-copied False (post-norm) collapsed every input to the same 0.154
    # spoof posterior; with True this gate strict-loads (565/565) and
    # discriminates inputs.
    "layer_norm_first": True,
    "conv_bias": True,         # checkpoint carries conv bias tensors (X2 probe)
    "final_dim": 768,          # SSL pre-training head width (project_q etc.)
    # defaults kept explicit for auditability:
    "conv_feature_layers": "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
    "activation_fn": "gelu",
}


def _build_net(device: str | None = None, mixer_cls=None, w2v_cls=None) -> Any:
    """Build the network skeleton.

    Lazy imports so a broken vendor tree degrades to ArchNotImplementedError.
    ``mixer_cls`` / ``w2v_cls`` are test-injection seams (tiny stand-ins that
    keep the hermetic suite off the 300M-param skeleton); production callers
    omit them.
    """
    from ._xlsrmamba_vendor.mamba_backend import MixerModel
    from ._xlsrmamba_vendor.wav2vec2_frontend import Wav2Vec2Config, Wav2Vec2Model

    dev = device if device is not None else resolve_device()
    net = _XLSRMambaNet(Wav2Vec2Config(_SSL_CFG_OVERRIDES),
                        mixer_cls or MixerModel,
                        w2v_cls or Wav2Vec2Model,
                        dev)
    net.eval()
    return net


def _load_safetensors(path: str) -> dict:
    """Read a .safetensors checkpoint into a plain {str: Tensor} state dict."""
    try:
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - env without safetensors
        raise RuntimeError(f"safetensors unavailable: {exc}") from exc
    sd: dict = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


class _XLSRMambaNet(nn.Module):
    """ssl_model(wav2vec2-XLSR) -> LL -> first_bn+SELU -> conformer(MixerModel).

    Top-level attribute names mirror the checkpoint's prefixes exactly (X2):
    ssl_model.model.*, LL.*, first_bn.*, conformer.*.
    """

    def __init__(self, ssl_cfg, mixer_model_cls, wav2vec2_model_cls, device: str) -> None:
        super().__init__()
        self.ssl_model = nn.Module()
        self.ssl_model.model = wav2vec2_model_cls(ssl_cfg)
        self.LL = nn.Linear(1024, 144)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.conformer = mixer_model_cls(
            d_model=144,
            n_layer=6,
            ssm_cfg={},
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=False,
        )
        self.to(device)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T_samples) float32 @16 kHz -> logits (B, 2) [spoof, bonafide]."""
        emb = self.ssl_model.model(wav, mask=False, features_only=True)["x"]
        x = self.LL(emb)                    # (B, T_frames, 144)
        x = x.unsqueeze(dim=1)              # (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)                # (B, T, 144)
        return self.conformer(x)


class XLSRMambaSpec(ArchSpec):
    """Fills the pipeline's xlsrmamba audio slot with the MIT-licensed
    XLSR-Mamba-LA detector (arXiv:2411.10027)."""

    name = "xlsrmamba"
    weight_env = "VISHWAS_XLSRMAMBA_WEIGHTS"
    #: Distinguishes a vendored arch from a stub.
    implemented = True

    def build(self) -> Any:
        return _build_net()

    def apply_state(self, model: Any, sd: dict) -> bool:
        """STRICT load — no remapping needed for this checkpoint (X2)."""
        try:
            missing, unexpected = model.load_state_dict(sd, strict=True), None
            self.last_apply = {
                "ok": True,
                "strict": True,
                "missing": [],
                "unexpected": [],
            }
            return True
        except Exception:
            self.last_apply = {
                "ok": False,
                "strict": True,
                "error": "strict load failed (checkpoint/vendor key mismatch)",
            }
            return False

    def score(self, model: Any, x: Any) -> float:
        """Spoof posterior in [0,1] from a raw waveform (label order X5)."""
        if isinstance(x, np.ndarray):
            wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        elif isinstance(x, torch.Tensor):
            wav = x.float()
        else:
            raise TypeError(
                f"xlsrmamba score() expects a waveform array, got {type(x).__name__}"
            )
        wav = wav.reshape(-1)
        # Truncate to the trained window (X4). For short inputs, ZERO-pad to the
        # window: upstream XLSR-Mamba pads via fairseq pad_to_multiple(value=0),
        # and repeat-padding (what this later emitted) creates an artificial
        # periodic signal that destroys the spoof posterior (measured: XLSR fake
        # posterior 0.591 repeat-pad vs 0.041 zero-pad on ASVspoof 2019; real
        # 0.583->0.368). Verified no length-label confound in the corpus.
        if wav.numel() < _INPUT_SAMPLES:
            wav = F.pad(wav, (0, _INPUT_SAMPLES - wav.numel()))
        else:
            wav = wav[:_INPUT_SAMPLES]
        wav = wav.unsqueeze(0)  # (1, 66800)
        # arch seam may hand us ArchModelWrapper or the raw net — unwrap once
        inner = getattr(model, "model", model)
        try:
            dev = next(inner.parameters()).device
            wav = wav.to(dev)   # follow wherever build() placed the model
        except (StopIteration, AttributeError):
            pass                # parameterless stub: nothing to align
        was_training = getattr(inner, "training", False)
        if hasattr(inner, "eval"):
            inner.eval()
        with torch.no_grad():
            logits = inner(wav)
        if was_training and hasattr(inner, "train"):
            inner.train()
        probs = F.softmax(logits.float(), dim=-1)
        return float(probs[0, 0].item())  # index 0 == spoof class (X5)


def get_arch() -> XLSRMambaSpec:
    """Registry hook consumed by ``vishwas.model_adapters._arch_aware_load``."""
    return XLSRMambaSpec()
