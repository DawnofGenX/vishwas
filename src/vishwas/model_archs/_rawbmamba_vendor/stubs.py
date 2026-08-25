"""Minimal stubs for vendored RawBMamba modules (CPU inference only)."""
import torch.nn as nn


class GenerationMixin:
    """Verisafe never generates; only .forward() is used by the adapter."""
    pass
