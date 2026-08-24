"""Single device-resolution seam for learned-model inference.

Resolution order:
  1. VERISAFE_DEVICE env ("cpu"/"cuda", case-insensitive) — operator override
  2. cuda when torch reports it available
  3. cpu fallback (never raises)

Never raises: any torch import/inspection failure degrades to "cpu". This
deliberately does NOT swallow downstream errors — callers keep their own
error semantics; this seam only answers "where".
"""
from __future__ import annotations

import os

__all__ = ["resolve_device"]


def resolve_device() -> str:
    override = os.environ.get("VERISAFE_DEVICE", "").strip().lower()
    if override in ("cpu", "cuda"):
        return override
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
