"""QR/barcode decoding: zxing-cpp primary, OpenCV QRCodeDetector fallback.

Decoder choice follows the empirical bake-off recorded in the docs/research
QR scouting report: the zxing-cpp v3.x wheel (with try_downscale +
try_rotate) decoded 16/16 degraded Indian-ID-style fixtures while
cv2.QRCodeDetector managed only 13/16, so cv2 is fallback-only.

Accepts a filesystem path, a PIL Image, or a numpy ndarray. Never raises —
a decode failure is an empty list, per the package tamper discipline.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import zxingcpp

__all__ = ["decode_image"]


def _to_ndarray(source) -> np.ndarray | None:
    """Normalise path / PIL Image / ndarray input to an image ndarray."""
    if isinstance(source, np.ndarray):
        return source
    # PIL Image duck-typing (avoid a hard PIL import here)
    if hasattr(source, "convert") and hasattr(source, "size") and hasattr(source, "width"):
        rgb = np.asarray(source.convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR, contiguous
    try:
        data = np.fromfile(str(Path(source)), dtype=np.uint8)  # unicode-safe
    except (OSError, TypeError, ValueError):
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def decode_image(source) -> list[str]:
    """Decode QR payloads from an image; return unique payload strings in order.

    Primary: zxing-cpp with downscale+rotate retries (best-in-class on
    degraded photo-like captures). Fallback when zxing sees nothing:
    cv2.QRCodeDetector on the grayscale frame, then on an Otsu-binarised
    copy. Any decoder exception is swallowed and reported as no result.
    """
    texts: list[str] = []
    img = _to_ndarray(source)
    if img is None or img.size == 0:
        return texts
    try:
        for res in zxingcpp.read_barcodes(img, try_downscale=True, try_rotate=True):
            t = getattr(res, "text", "")
            if t and t not in texts:
                texts.append(t)
    except Exception:
        texts = []
    if texts:
        return texts
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        det = cv2.QRCodeDetector()
        for frame in (gray,
                      cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]):
            txt, _pts, _ = det.detectAndDecode(frame)
            if txt and txt not in texts:
                texts.append(txt)
                break
    except Exception:
        pass
    return texts
