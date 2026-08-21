"""HAVIC holistic audio-visual coherence — arch-class STUB (roadmap Phase 1).

STATUS: NOT YET IMPLEMENTED (Task 1.4 / B3 fills this in).

Checkpoints on disk under ``VERISAFE_HAVIC_WEIGHTS``:
  * primary  ``havic/best_ft/best_ft_model.pth`` — 456 tensors; roots
    audio_encoder(164), visual_encoder(163), Audio/VisualTokenReducer_{3,6,9,
    12,AVI}, AudioVisualInteractionModule(69), classifier/classifier_audio/
    classifier_visual(6 each), pool_a, pool_v.
  * fallback ``havic/pt200/pt_model.200.pth`` (~1 GB) when best_ft is missing/
    unreadable — secondary fallback chain to be implemented here in 1.4.

Design reference: arXiv:2603.23960 (JielunPeng/HAVIC); ``score()`` will return
the inconsistency probability.

Until 1.4 lands, :func:`get_arch("havic")` returns an ArchSpec whose
``build()``/``score()`` raise ``ArchNotImplementedError`` so the adapter seam
honestly degrades to None + reason 'weight file loaded but architecture
unavailable' — never a half-loaded model.
"""
from __future__ import annotations

from .base import ArchNotImplementedError, ArchSpec

#: Checkpoint fallback chain (best_ft primary, pt200 secondary).
CHECKPOINT_CHAIN = ("best_ft", "pt200")


class HavicArch(ArchSpec):
    """HAVIC arch spec — placeholder until task 1.4 vendors the real net."""

    name = "havic"
    weight_env = "VERISAFE_HAVIC_WEIGHTS"
    implemented = False

    def build(self):
        # Honest stub: raise, do not return a skeleton pretending to work.
        raise ArchNotImplementedError(
            "havic architecture not vendored yet (task 1.4)"
        )

    def score(self, model, x):
        raise ArchNotImplementedError(
            "havic scoring not vendored yet (task 1.4)"
        )


def get_arch() -> HavicArch:
    """Registry hook consumed by ``verisafe.model_adapters._arch_aware_load``."""
    return HavicArch()
