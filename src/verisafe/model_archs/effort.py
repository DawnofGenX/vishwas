"""EFFORT spatial face/AIGI detector — arch-class STUB (roadmap Phase 1).

License: CC BY-NC 4.0 (YZY-stack/Effort-AIGI-Detection). Port of method, not
copy of repo code. Non-commercial use only.

STATUS: NOT YET IMPLEMENTED (Task 1.3 / B2 fills this in).

Checkpoints on disk under ``VERISAFE_EFFORT_WEIGHTS`` (chameleon primary per
locked Decision #2; ffpp/genimage lazy fallback chain): OrderedDict of 681
tensors, all ``module.``-prefixed; full ViT-L trunk inside the checkpoint
(embeddings.patch_embedding (1024,3,14,14), pos 257, OrthAlign self-attn with
k_proj.weight_main/.bias/.S_residual; head.weight (2,1024)) — no external
CLIP backbone needed. See docs/research/WEIGHT_KEY_MAPS.md for the key-map
provenance.

Until 1.3 lands, :func:`get_arch("effort")` returns an ArchSpec whose
``build()``/``score()`` raise ``ArchNotImplementedError`` so the adapter seam
honestly degrades to None + reason 'weight file loaded but architecture
unavailable' — never a half-loaded model. When enabled, the evidence record
will carry ``"license": "CC-BY-NC-4.0"`` (added in 1.3).
"""
from __future__ import annotations

from .base import ArchNotImplementedError, ArchSpec

#: Checkpoint fallback chain when chameleon is missing/unreadable (logic will
#: live here once the real arch lands in 1.3).
CHECKPOINT_CHAIN = ("chameleon", "ffpp", "genimage")


class EffortArch(ArchSpec):
    """EFFORT arch spec — placeholder until task 1.3 vendors the real net."""

    name = "effort"
    weight_env = "VERISAFE_EFFORT_WEIGHTS"
    implemented = False
    license = "CC-BY-NC-4.0"

    def build(self):
        # Honest stub: raise, do not return a skeleton pretending to work.
        raise ArchNotImplementedError(
            "effort architecture not vendored yet (task 1.3)"
        )

    def score(self, model, x):
        raise ArchNotImplementedError(
            "effort scoring not vendored yet (task 1.3)"
        )


def get_arch() -> EffortArch:
    """Registry hook consumed by ``verisafe.model_adapters._arch_aware_load``."""
    return EffortArch()
