"""AASIST (wavLm) audio anti-spoofing — arch-class STUB (roadmap Phase 1).

STATUS: NOT YET IMPLEMENTED (Task 1.2 / B1 fills this in).

The 733-tensor checkpoint at ``VERISAFE_AASIST_WEIGHTS`` (top-level key
``model_state``; frontend./backend./losses. prefixes; see docs/research/
WEIGHT_KEY_MAPS.md) is on disk but its network class is not yet vendored.
Until 1.2 lands, :func:`get_arch("aasist")` returns an ArchSpec whose
``build()``/``score()`` raise ``ArchNotImplementedError`` so the adapter seam
honestly degrades to None + reason 'weight file loaded but architecture
unavailable' — never a half-loaded model.

Reference: Jung et al., "AASIST" (arXiv:2110.01200, ASVspoof lineage).
"""
from __future__ import annotations

from .base import ArchNotImplementedError, ArchSpec


class AasistArch(ArchSpec):
    """AASIST arch spec — placeholder until task 1.2 vendors the real net."""

    name = "aasist"
    weight_env = "VERISAFE_AASIST_WEIGHTS"
    implemented = False

    def build(self):
        # Honest stub: raise, do not return a skeleton pretending to work.
        raise ArchNotImplementedError(
            "aasist architecture not vendored yet (task 1.2)"
        )

    def score(self, model, x):
        raise ArchNotImplementedError(
            "aasist scoring not vendored yet (task 1.2)"
        )


def get_arch() -> AasistArch:
    """Registry hook consumed by ``verisafe.model_adapters._arch_aware_load``."""
    return AasistArch()
