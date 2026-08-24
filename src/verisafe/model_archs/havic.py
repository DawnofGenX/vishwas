"""HAVIC holistic audio-visual coherence — vendored arch spec (roadmap 1.4a).

Provenance / licensing
======================
VENDORED (copy-adapted) from the MIT-licensed reference implementation:
  * repo    : https://github.com/JielunPeng/HAVIC (``src/models/*``)
  * paper   : arXiv:2603.23960
  * license : MIT (upstream headers preserved in the vendored files)
The backend lives in ``verisafe.model_archs._havic``.  The ONLY local edits
vs upstream are:
  (a) timm imports replaced by a minimal in-package shim
      (``_havic/_timm_shim.py``) — timm is not installed in the real-torch
      tree; the shim reproduces the classic timm parameter layouts the
      checkpoint was trained with (fused ``qkv`` + ``proj`` Attention,
      ``fc1``/``fc2`` Mlp).
  (b) one kwarg fix in ``HAVIC.forward``: upstream passed ``use_mask=False``
      to encoder forwards whose signatures say ``use_hierarchical`` (latent
      TypeError); vendored copy passes ``use_hierarchical=True``.

Checkpoint fallback chain
=========================
``CHECKPOINT_CHAIN = ("best_ft", "pt200")``

  * ``best_ft`` (primary) — finetune payload, 456 tensors, FLAT top-level
    dict: audio_encoder(164), visual_encoder(163),
    AudioVisualInteractionModule(69), classifier / classifier_audio /
    classifier_visual (6 each), Audio|VisualTokenReducer_{3,6,9,12,AVI}
    (4 each), pool_a, pool_v.
  * ``pt200`` (secondary) — pretrain payload, 676 tensors: same encoders +
    AVIM plus audio_decoder / visual_decoder / A2V / V2A.  HONEST
    LIMITATION: pt200 carries NO classifier / reducer / pool keys, so
    loading it into the full scoring skeleton leaves far more than 5% of
    expected keys missing and :meth:`HavicArch.apply_state` returns False.
    The chain entry is therefore NOT a working scoring fallback — a real
    fallback would need a dedicated head-less skeleton.  Documented here so
    no caller assumes encoder-only half-loads are usable (they never are).

Key-map summary (verified against tests/fixtures/havic_best_ft_key_shapes.txt)
==============================================================================
  * AVIM self-attn : classic timm Attention — fused ``qkv`` Linear (with
    bias) + ``proj``  -> provided by the shim.
  * AVIM mlp       : timm ``fc1``/``fc2`` -> shim ``Mlp``.
  * audio encoder  : upstream-local Attention with separate ``q_bias`` /
    ``v_bias`` Parameters and a fused bias-less ``qkv`` weight (NOT timm).
  * visual encoder : upstream-local Block3d / PatchEmbedding3d (Conv3d
    ``768x3x2x16x16``) — no timm.
  * heads          : FlexibleMLP (``layers.0/1`` + ``output_layer``),
    TokenWise_TokenReducer (``mlp.0``/``mlp.2``),
    LearnableWeightedPool (``weights``).

Input conventions (reference dataloader + HAVIC_FT.forward comments)
====================================================================
  * audio  : kaldi-style log-fbank interpolated to 1024 x 128
    (target_length=1024 time frames, 128 mel bins), float32; model input
    ``(B, 1024, 128)`` — ``AudioEncoder.forward`` unsqueezes the channel dim
    itself (HAVIC_FT's inline "(B, 1, 1024, 128)" comment is stale; caught
    by the task-1.4a smoke forward).  ``score()`` applies NO extra
    normalisation — callers supply features preprocessed exactly as in
    training.
  * visual : 16 RGB face frames (``face_00``..``face_15``), 224x224,
    float32; model input ``(B, 3, 16, 224, 224)``.  ``tubelet_size=2`` ->
    8 temporal segments; the net hardcodes ``n_segments=8`` throughout.
  * forward: ``model(audio, video, is_training=False)`` -> overall logits
    ``(B, 1)`` (training mode additionally returns the two aux logits).

Assumption H5 (polarity, unverified): ``score()`` returns the batch-mean
``sigmoid(overall logit)`` in [0, 1] following the reference training
convention.  The real/fake pole has NOT been checked against real weights
(no real-weight inference in task 1.4a) — the wiring task MUST smoke-check
polarity before this score is used for gating.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import ArchNotImplementedError, ArchSpec
from ..device import resolve_device

#: Checkpoint fallback chain (best_ft primary, pt200 secondary).
#: See module docstring: pt200 lacks the heads, so it cannot satisfy the
#: full skeleton — apply_state() honestly returns False for it.
CHECKPOINT_CHAIN = ("best_ft", "pt200")

#: Payload wrappers a checkpoint may nest its real state dict under.
_WRAPPER_KEYS = ("model", "state_dict", "net", "weights")


class HavicArch(ArchSpec):
    """HAVIC arch spec — dual ViT-L encoders + AVIM + weighted-pool heads."""

    name = "havic"
    weight_env = "VERISAFE_HAVIC_WEIGHTS"
    implemented = True

    def build(self):
        """Construct the HAVIC_FT skeleton with the reference defaults.

        Defaults (img_size=224, patch_size=16, n_frames=16, audio_length=1024,
        mel_bins=128, embed 768, depth 12, heads 12, tubelet 2) reproduce the
        best_ft checkpoint shapes key-for-key (see the fixture test).  The
        import is lazy so the hermetic tree (torch stub, no ``torch.nn``)
        degrades cleanly instead of breaking package import.
        """
        try:
            from ._havic.HAVIC import HAVIC_FT
        except Exception as exc:  # ImportError under the hermetic torch stub
            raise ArchNotImplementedError(
                f"havic backend requires real torch + the _havic package "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        model = HAVIC_FT()
        model.to(resolve_device())  # no-op on CPU-only hosts
        model.eval()
        return model

    def apply_state(self, model: Any, sd: Dict[str, Any]) -> bool:
        """Unwrap nested payloads, strip the DataParallel ``module.`` prefix,
        then delegate coverage checking to the base class (strict=False,
        >5% missing/mismatched -> False).

        The probed best_ft payload is a flat top-level dict, but single-key
        wrappers (``model`` / ``state_dict`` / ``net`` / ``weights``) are
        unwrapped defensively so re-saved payloads load unchanged.
        """
        if not isinstance(sd, dict) or not sd:
            return super().apply_state(model, sd)  # base records + returns False
        payload = sd
        if len(payload) == 1:
            only_key = next(iter(payload))
            inner = payload[only_key]
            if only_key in _WRAPPER_KEYS and isinstance(inner, dict) and inner:
                payload = inner
        stripped = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in payload.items()
        }
        return super().apply_state(model, stripped)

    def score(self, model: Any, x: Any) -> float:
        """Mean ``sigmoid(overall logit)`` in [0, 1] (assumption H5).

        *x* is an ``(audio_features, visual_frames)`` pair — numpy arrays or
        torch tensors.  Audio accepts ``(1024, 128)``, ``(B, 1024, 128)`` or
        ``(B, 1, 1024, 128)`` (channel squeezed); video accepts
        ``(3, 16, 224, 224)`` or ``(B, 3, 16, 224, 224)``.  Runs
        ``is_training=False`` under ``no_grad`` and eval() (restored
        afterwards).
        """
        import numpy as np
        import torch

        if isinstance(x, (str, bytes)) or not isinstance(x, (tuple, list)) or len(x) != 2:
            raise TypeError(
                "havic score() expects an (audio_features, visual_frames) "
                f"tuple, got {type(x).__name__}"
            )
        a_t = self._as_float_tensor(x[0])
        v_t = self._as_float_tensor(x[1])
        # audio -> (B, 1024, 128); AudioEncoder unsqueezes its own channel dim
        if a_t.ndim == 2:
            a_t = a_t[None]
        elif a_t.ndim == 3:
            pass
        elif a_t.ndim == 4 and a_t.shape[1] == 1:
            a_t = a_t.squeeze(1)  # tolerate the stale (B,1,1024,128) form
        else:
            raise TypeError(
                f"havic audio features must be (1024,128)/(B,1024,128)"
                f"[/(B,1,1024,128)], got {tuple(a_t.shape)}"
            )
        # video: (3,16,224,224) | (B,3,16,224,224) -> (B,3,16,224,224)
        if v_t.ndim == 4:
            v_t = v_t[None]
        elif v_t.ndim != 5:
            raise TypeError(
                f"havic visual frames must be (3,T,H,W) or (B,3,T,H,W), "
                f"got {tuple(v_t.shape)}"
            )
        was_training = model.training
        model.eval()
        try:
            try:
                dev = next(model.parameters()).device
                a_t = a_t.to(dev)
                v_t = v_t.to(dev)   # follow wherever build() placed the model
            except StopIteration:
                pass                # parameterless stub: nothing to align
            with torch.no_grad():
                logits = model(a_t, v_t, is_training=False)
        finally:
            if was_training:
                model.train()
        probs = torch.sigmoid(logits)
        return float(probs.mean().item())

    @staticmethod
    def _as_float_tensor(arr: Any):
        """numpy array / torch tensor -> float32 torch tensor (no copy when
        already a float32 tensor)."""
        import numpy as np
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().to(torch.float32)
        a = np.asarray(arr)
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(a))


def get_arch() -> HavicArch:
    """Registry hook consumed by ``verisafe.model_adapters._arch_aware_load``."""
    return HavicArch()
