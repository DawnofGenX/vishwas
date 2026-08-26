"""Vendored SPAI net (CVPR'25 "Any-Resolution AI-Generated Image Detection by
Spectral Learning", mever-team/spai, Apache-2.0). See PROVENANCE.md.

This package is imported LAZILY from vishwas.model_archs.spai (only when a SPAI
weight path is provisioned), never at image_facecheck import time, so the heavy
dependencies (torchvision/einops) never break the hermetic suite.
"""