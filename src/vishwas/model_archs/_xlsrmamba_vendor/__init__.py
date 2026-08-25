"""CPU-vendored XLSR-Mamba model tree (MIT).

Contents
--------
- ``wav2vec2_frontend`` : fairseq-style wav2vec 2.0 (XLSR-300M class) frontend,
  vendored from facebookresearch/fairseq @ a54021305d6b3c (the commit the
  upstream repo pins).
- ``mamba_backend``     : dual-column bidirectional Mamba trunk + attention
  pooling + linear head, vendored from swagshaw/XLSR-Mamba mamba_blocks.py +
  state-spaces/mamba mamba_simple.py.
- ``cpu_shims``         : pure-PyTorch replacements for the CUDA-only deps
  (selective_scan / causal_conv1d / triton RMSNorm), same recipe as
  ``_rawbmamba_vendor.cpu_ops``.

Upstream licences: fairseq MIT; XLSR-Mamba MIT; mamba (state-spaces) Apache-2.0.
See ../xlsrmamba.py for the full provenance header and key-map notes.
"""
