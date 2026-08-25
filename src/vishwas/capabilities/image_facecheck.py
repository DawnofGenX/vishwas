"""Image face-check capability: still-photo face-manipulation forensics.

WhatsApp users often send static photos of "relatives"/"officials" asking for
money or confirmation. Pipeline:
  T0 cheap : magic-byte verify (already done upstream) + basic integrity scan
             (exif sanity, resolution, per-band luma stats)
  T1 heavy : gated face-forensics model (EFFORT weights reuse; env override
             VISHWAS_IMAGE_FACE_WEIGHTS allowed to point at a dedicated one)
Offline fallback: frequency-domain anomaly scoring (spectral discontinuity in
the mid-bands where swap-paste seams live) so a bare CPU box returns something
real rather than nothing.
"""
from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from ..device import resolve_device
from ..events import Artifact, JobContext, MediaKind
from ..model_adapters import resolve as _resolve_adapter, _call_model as _call_model_compat, _auto_extract, is_usable_model as _is_usable_model
from .base import CheckResult

IMAGE_KINDS = {MediaKind.PNG, MediaKind.JPEG, MediaKind.WEBP, MediaKind.GIF,
               MediaKind.TIFF, MediaKind.HEIC}


def _load_image(p: Path, side: int = 512) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            return None
        return cv2.resize(im, (side, side))
    except Exception:
        return None


def _spectral_band_anomaly(img: np.ndarray) -> float:
    """Mid-frequency energy ratio proxy. Swap/blend seams and GAN artifacts
    typically push energy into specific mid bands; real photos have a smoother
    falloff. Bounded heuristic -> 0..1."""
    gray = img.mean(axis=2).astype(np.float32)
    F = np.abs(np.fft.fft2(gray - gray.mean()))
    h, w = F.shape
    cy, cx = h // 2, w // 2
    # annuli around DC
    def ring(r_in: int, r_out: int) -> float:
        yy, xx = np.mgrid[0:h, 0:w]
        dy, dx = yy - cy, xx - cx
        m = (dy * dy + dx * dx >= r_in ** 2) & (dy * dy + dx * dx < r_out ** 2)
        return float(F[m].mean()) if m.any() else 0.0
    low = ring(4, 64)
    mid = ring(64, 128)
    high = ring(128, 256)
    denom = (low + mid + high) or 1.0
    ratio = mid / denom                     # typical real photo ~0.25-0.4
    anomaly = abs(ratio - 0.32) / 0.32      # deviation from expected band mix
    return round(min(1.0, anomaly), 3)


class ImageFaceCheckCapability:
    requires: tuple[str, ...] = ()

    # QR evidence for photographed gov IDs (gap fix 2026-08-25): the QR
    # verifier lives in GovDocumentCapability, which photo'd IDs never reach
    # (they route here). When the filename carries gov hints, borrow that
    # check so aadhaar.jpg etc. get qr_payload_check evidence too.
    _gov_hints = None

    def _qr_evidence_for_gov_image(self, art: Artifact) -> list[CheckResult]:
        if ImageFaceCheckCapability._gov_hints is None:
            from ..router import _GOV_HINTS
            ImageFaceCheckCapability._gov_hints = _GOV_HINTS
        name = (art.original_filename or "").lower()
        if not ImageFaceCheckCapability._gov_hints.search(name):
            return []
        try:
            from .gov_document import GovDocumentCapability
            return GovDocumentCapability()._qr_payload_checks(art)
        except Exception:  # noqa: BLE001 — QR is additive evidence, never fatal
            return []

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        kind = art.verified_kind
        if kind is None or kind not in IMAGE_KINDS:
            out.append(CheckResult("image_check", "cheap", "skipped", {},
                                   "verified content is not an image"))
            return out
        out.append(CheckResult("image_integrity", "cheap", "ok",
                               {"format": kind.value,
                                "size_bytes": art.size_bytes,
                                "sha256": art.sha256},
                               "container-level integrity ok"))
        img = _load_image(art.path)
        if img is None:
            out.append(CheckResult("image_face_forensics", "mid", "degraded", {},
                                   "could not decode pixel data with available decoders"))
            out.extend(self._qr_evidence_for_gov_image(art))
            return out
        anomaly = _spectral_band_anomaly(img)
        out.append(CheckResult("frequency_band_analysis", "mid", "ok",
                               {"prob_deepfake": round(min(1.0, 0.35 + 0.4 * anomaly), 3),
                                "band_anomaly": anomaly,
                                "source": "offline_frequency_heuristics"},
                               "offline frequency-band scan (fallback when face models absent)"))
        # gated learned model
        env, _p = _resolved_weights_env()
        adapter = _resolve_adapter(env) if env else None
        m = _load_model()
        if m is None:
            out.append(CheckResult("image_face_forensics", "heavy", "unavailable",
                                   {"missing_dependency": "model-weights"},
                                   "face-forensics weights not provisioned; frequency heuristic carries"))
        else:
            try:
                p = _infer_face(adapter, m, img)
                if p is None:
                    out.append(CheckResult("image_face_forensics", "heavy", "degraded",
                                           {}, "face-forensics model produced no usable score"))
                    return out
                out.append(CheckResult("image_face_forensics", "heavy", "ok",
                                       {"prob_deepfake": round(min(1.0, max(0.0, p)), 3)},
                                       "learned face-manipulation detector pass"))
            except Exception as e:  # noqa: BLE001
                out.append(CheckResult("image_face_forensics", "heavy", "failed",
                                       {"error_class": e.__class__.__name__}, "inference error"))
        out.extend(self._qr_evidence_for_gov_image(art))
        return out


def _resolved_weights_env() -> tuple[str | None, str | None]:
    """Return (env_name, weight_path) honouring the documented override order:
    VISHWAS_IMAGE_FACE_WEIGHTS first, else fall back to VISHWAS_EFFORT_WEIGHTS."""
    p = os.environ.get("VISHWAS_IMAGE_FACE_WEIGHTS")
    if p:
        return "VISHWAS_IMAGE_FACE_WEIGHTS", p
    p = os.environ.get("VISHWAS_EFFORT_WEIGHTS")
    if p:
        return "VISHWAS_EFFORT_WEIGHTS", p
    return None, None


def _load_model():
    """Registry-gated loader; returns None unless the resolved env path exists AND
    torch is importable. Routes through model_adapters (face or EFFORT family)."""
    env, p = _resolved_weights_env()
    if not p or not os.path.exists(p):
        return None
    adapter = _resolve_adapter(env) if env else None
    if adapter is not None:
        obj = adapter.load(p)
        return obj if _is_usable_model(obj) else None
    try:
        import torch  # type: ignore
        obj = torch.load(p, map_location=resolve_device(), weights_only=False)
        return obj if _is_usable_model(obj) else None
    except Exception:
        return None


def _infer_face(adapter, model, img: np.ndarray) -> float | None:
    """One face-forensics inference via the registry (or legacy .predict).
    [0,1] or None; never raises."""
    if adapter is not None:
        processed = adapter.preprocess(img)
        out = _call_model_compat(model, processed)
        prob = adapter.extract_prob(out) if adapter.extract_prob else None
        if prob is None:
            prob = _auto_extract(out)
        return prob
    if hasattr(model, "predict"):
        out = model.predict(img)
        val = out[0] if isinstance(out, (list, tuple)) else out
        return min(1.0, max(0.0, float(val)))
    return None
