"""EFFORT's INPUT CONTRACT — it was being fed BGR with the wrong normalisation.

THE DEFECT (2026-10-21)
-----------------------
EFFORT is a CLIP ViT-L/14 (YZY-stack/Effort-AIGI-Detection, ICML 2025
Oral). The serving path is:

    cv2.imread(...)        -> HWC uint8, BGR
    _img_resize_chw        -> centre-crop, resize, /255, transpose -> CHW
    EffortSpec.score()     -> normalise, then softmax(logits)[..., 1]

TWO independent errors, both of which the model was never trained for:

1. CHANNEL ORDER. `cv2.imread` returns BGR. Neither `_img_resize_chw` nor
   `EffortSpec.score` converts to RGB. EFFORT is a CLIP model, trained on
   RGB. The sibling archs all get this right --
   `spai.py` ("SPAI was trained on PIL/RGB" -> `rgb = arr[..., ::-1]`),
   `nyuad.py`, and `effnet_image.py` ("decodes through cv2, hence BGR -> RGB")
   all do the swap. EFFORT was the only one using the shared
   `_img_resize_chw`, which does not.

2. NORMALISATION CONSTANTS. The official implementation
   (DeepfakeBench/training/config/detector/effort.yaml) specifies
       mean: [0.48145466, 0.4578275, 0.40821073]
       std:  [0.26862954, 0.26130258, 0.27577711]
   which are CLIP's constants. The repo used ImageNet's
       mean: [0.485, 0.456, 0.406]
       std:  [0.229, 0.224, 0.225]

MEASURED on the leak-free corpus (60 real files, production code paths):

    (BGR, ImageNet)  <- repo today        AUC 0.6511
    (BGR, CLIP)                            AUC 0.6278
    (RGB, ImageNet)                        AUC 0.7211
    (RGB, CLIP)      <- upstream           AUC 0.7189

The channel swap is worth +0.070 AUC on its own. The normalisation
constants are worth ~0.002 either way on this corpus -- but they are still
wrong, and the upstream config is unambiguous, so both are fixed.

The -0.55 affine offset in the fusion calibration was fitted to the OLD
(bgr_imnet) score distribution and MUST be re-derived: with the fix the
scores shift down (real median 0.6326 -> 0.4762), so -0.55 would flatten
the separation to nothing.

These tests pin the INPUT CONTRACT (what the model is fed) rather than a
measured AUC, because the contract is checkable without a corpus and
without weights.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vishwas.model_archs import effort as eff


class _CaptureModel(torch.nn.Module):
    """A stand-in for the EFFORT net that records the tensor it was given.

    It must accept the input and return (N, 2) logits so `score()` can run
    to completion.
    """

    def __init__(self):
        super().__init__()
        self.seen: torch.Tensor | None = None
        # A parameter so `next(model.parameters())` works and the device
        # lookup in score() does not raise StopIteration.
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.seen = x.detach().clone()
        return torch.zeros((x.shape[0], 2))


def test_effort_normalisation_uses_the_upstream_clip_constants():
    """The official EFFORT config (DeepfakeBench effort.yaml) specifies
    CLIP's constants, not ImageNet's. ImageNet's std is ~17% smaller, so the
    repo's normalisation amplified the input relative to training."""
    assert np.allclose(eff._IMAGENET_MEAN,
                       [0.48145466, 0.4578275, 0.40821073], atol=1e-6), (
        "the mean must be CLIP's, from the upstream effort.yaml")
    assert np.allclose(eff._IMAGENET_STD,
                       [0.26862954, 0.26130258, 0.27577711], atol=1e-6), (
        "the std must be CLIP's, from the upstream effort.yaml")


def test_effort_score_converts_bgr_input_to_rgb():
    """EFFORT is trained on RGB. The serving path hands it BGR (cv2.imread),
    so the CONVERSION MUST happen in score() -- the same place spai.py,
    nyuad.py and effnet_image.py do it.

    This is the regression: EFFORT was the only arch using the shared
    `_img_resize_chw`, which does not swap channels.

    THE PROBE. A channel carrying 1.0 normalises to (1-mean)/std, which is
    POSITIVE for all three CLIP means; a channel carrying 0.0 normalises to
    -mean/std, which is NEGATIVE. So the channel that holds the content is
    the one with the LARGEST value. The input below is CHW (what
    `_img_resize_chw` produces) with the content at index 2 -- i.e. BGR
    order. After a correct swap the model must see it at index 0.

    NOTE: comparing two DIFFERENT frames does not work here -- each frame
    has content in exactly one channel, so the red-minus-blue difference is
    identical with or without the swap. The single-frame channel profile
    is the only unambiguous probe.
    """
    model = _CaptureModel()
    chw = np.zeros((1, 3, 8, 8), dtype=np.float32)
    chw[0, 2] = 1.0        # content at index 2 == BGR order

    eff.EffortSpec().score(model, chw)
    got = model.seen.numpy()
    assert got is not None
    per_channel = [float(got[0, c].max()) for c in range(3)]
    assert int(max(range(3), key=lambda c: per_channel[c])) == 0, (
        "score() did not convert BGR to RGB: the model still sees the "
        f"content at channel {int(max(range(3), key=lambda c: per_channel[c]))}, "
        "so it is being fed blue-red-swapped pixels")


def test_effort_swap_is_on_the_channel_axis_not_the_width_axis():
    """`arr[..., ::-1]` reverses the LAST axis. For a CHW array that is the
    WIDTH, not the channel axis -- SPAI's copy of that line is correct only
    because SPAI's array is HWC.

    A width-reversal is silent: the tensor still has the right shape, so
    nothing raises and the model simply reads a mirrored image. This test
    pins the axis explicitly: a frame with an asymmetric horizontal pattern
    must come out with the pattern intact (not mirrored)."""
    model = _CaptureModel()
    chw = np.zeros((1, 3, 4, 4), dtype=np.float32)
    chw[0, :, :, 0] = 1.0      # content in the LEFT column only

    eff.EffortSpec().score(model, chw)
    got = model.seen.numpy()
    assert got is not None
    left = float(got[0, :, :, 0].max())
    right = float(got[0, :, :, -1].max())
    assert left > right, (
        "the frame was MIRRORED horizontally -- the BGR->RGB swap was "
        "applied to the width axis instead of the channel axis")


def test_effort_score_is_documented_as_taking_bgr():
    """The arch must document the expected channel order, so a caller that
    supplies RGB can see the contract. A silent mismatch is what made this
    bug invisible for months."""
    doc = (eff.EffortSpec.score.__doc__ or "")
    assert "BGR" in doc, (
        "EffortSpec.score must document that it takes BGR HWC/CHW input "
        "and converts to RGB internally")


def test_effort_score_still_returns_a_finite_probability():
    """The contract change must not break the return contract: a float in
    [0, 1], never NaN."""
    model = _CaptureModel()
    x = np.random.rand(1, 3, 16, 16).astype(np.float32)
    p = eff.EffortSpec().score(model, x)
    assert isinstance(p, float)
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0
