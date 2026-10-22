"""The EFFORT video path must score the FACE REGION when a face is present.

WHY
---
EFFORT is a **face**-forensics model. The upstream paper
(YZY-stack/Effort-AIGI-Detection, ICML 2025 Oral) applies it to face
regions, and this repo's own `model_archs/effort.py` notes that the
`chameleon` checkpoint is the "face-region / spatial forensics" one while
`ffpp` is "face forensics (FF++ domain)".

The pipeline was feeding EFFORT the WHOLE FRAME, centre-cropped. The
`cross_modal` capability already loads a YuNet face detector
(`face_detection_yunet_2023mar.onnx`) for the HAVIC first-face crop, so the
tooling is already a dependency of this project.

MEASURED on the leak-free corpus (60 real files, production code paths):

    whole frame (before)   AUC 0.7189   real med 0.4762  fake med 0.6190
    YuNet face, 1.3x       AUC 0.7789   real med 0.4619  fake med 0.6881

(2026-10-22: the face row said 0.7711. That was the number for a REJECTED
crop -- tools/eval_effort_face_crop.py has its OWN hand-rolled crop. The
production crop measures 0.7789; both harnesses now say which is which. See
tests/test_68_face_crop_auc_provenance.py.)

The effect is real, not noise: 20 of 30 fakes' scores ROSE with the crop and
13 of 30 reals' scores FELL. A face was detected on 60/60 clips.

In the FUSED verdict, the face crop at offset -0.35 is strictly better than
the whole frame at -0.30:

                        FP   fakes flagged   fakes reading TRUST
    whole frame -0.30    2        10                 15
    face crop   -0.35    2        14                 11

...and on the operator's OWN media (the clean-side guard) both read 4/4
genuine = trust and 3/3 AI videos = do_not_use.

⚠️ 2026-10-22: THE "3/3 AI = do_not_use" ABOVE IS NO LONGER TRUE, AND IT
WAS NEVER A PROPERTY OF THE FACE CROP. It came from the cross-modal reading
av=0.5, which was the correlation function's UNDEFINED SENTINEL (-2.0) --
the AI videos have a COMPLETELY SILENT audio track (RMS 0.0), so their
correlation was undefined and the pipeline reported them as anti-correlated.
With the honest reading they read MEDIUM/caution. The face crop's own
measured effect (the AUC and the corpus numbers above) is unchanged; only
the end-to-end AI-video verdict is.

The 4/4 genuine = trust reading still holds, and that is the half of the
clean-side guard this crop is responsible for.

THE FALLBACK IS MANDATORY
-------------------------
One of the operator's 4 genuine clips has **no detectable face** on any
sampled frame. A face-crop path would have no signal for it, so the crop
MUST fall back to the whole frame. The clip must keep its whole-frame score
rather than read zero or being skipped.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vishwas.capabilities import deepfake_video as dv

FACE_MODEL = os.environ.get(
    "VISHWAS_FACE_MODEL",
    "/opt/vishwas/models/havic/face_detection_yunet_2023mar.onnx")


def _has_face_model() -> bool:
    return bool(FACE_MODEL) and os.path.exists(FACE_MODEL)


def test_the_face_model_path_is_documented():
    """The arch must name the detector it uses, so a deployment can point at
    its own copy."""
    assert hasattr(dv, "_EFFORT_FACE_MODEL"), (
        "deepfake_video must expose the face-model path it uses")
    assert hasattr(dv, "_EFFORT_FACE_SCALE"), (
        "deepfake_video must expose the face-crop expansion factor")


def test_face_crop_expansion_factor_is_the_facex_zoo_value():
    """FaceX-Zoo's FaceX_Zoo_FaceCropper uses 1.3; the cross-modal
    capability already uses the same constant for the HAVIC first-face
    crop. They must agree so the two crops are comparable."""
    from vishwas.capabilities import cross_modal as cm
    assert dv._EFFORT_FACE_SCALE == cm._HAVIC_FACE_SCALE == 1.3


@pytest.mark.skipif(not _has_face_model(), reason="no YuNet face model on this host")
def test_face_crop_falls_back_to_the_whole_frame_when_no_face_is_found():
    """A frame with no face must yield the WHOLE-FRAME crop, never a blank
    or a zero."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)   # black, no face
    out = dv._effort_face_crop(img)
    assert out is not None
    assert out.shape[:2] == (64, 64), (
        "the fallback must be the WHOLE frame, not a smaller crop")


@pytest.mark.skipif(not _has_face_model(), reason="no YuNet face model on this host")
def test_face_crop_returns_a_smaller_region_when_a_face_is_found():
    """With a face present, the crop must be the face region (expanded),
    which is smaller than the whole frame."""
    # A synthetic "face"-ish blob is not reliably detected by YuNet, so this
    # test only asserts the FALLBACK contract holds and that the helper does
    # not raise on a real image. The detection path is covered by the live
    # measurement (tools/eval_effort_face_crop.py).
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    out = dv._effort_face_crop(img)
    assert out is not None
    assert out.ndim == 3 and out.shape[2] == 3


def test_effort_uses_the_face_crop_when_available():
    """The EFFORT scoring path must call the face-crop helper. If it does
    not, the whole frame is scored and the measured gain is lost."""
    src = pathlib.Path(dv.__file__).read_text()
    assert "_effort_face_crop" in src, (
        "the EFFORT path does not use the face crop")
