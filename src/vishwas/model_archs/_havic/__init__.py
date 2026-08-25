"""Vendored HAVIC backend (Vishwas task 1.4a).

Copy-adapted from the MIT-licensed reference implementation
(JielunPeng/HAVIC, arXiv:2603.23960).  Modules are imported lazily by
``vishwas.model_archs.havic.HavicArch.build()`` because they require REAL
torch (``torch.nn``); the hermetic test tree's torch stub cannot import them.

Local edits vs upstream (all documented in havic.py):
  * timm imports replaced by the local ``_timm_shim`` (timm absent in-tree).
  * ``HAVIC.forward`` passed ``use_mask=`` to encoder forwards whose signatures
    say ``use_hierarchical`` (upstream latent bug) — fixed to
    ``use_hierarchical=True``.
"""
