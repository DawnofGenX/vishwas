"""Model-architecture registry (roadmap Phase 1, B0).

:func:`get_arch` maps a family name to its :class:`~verisafe.model_archs.base.ArchSpec`
by LAZILY importing the matching sibling module.  Unknown families return
``None``; import failures also return ``None`` (adapter treats both as
'architecture unavailable' — honest degradation, never half-load).
"""
from __future__ import annotations

from typing import Optional

from .base import ArchNotImplementedError, ArchSpec

#: family -> module name inside this package
_FAMILIES = {
    "aasist": "aasist",
    "effort": "effort",
    "havic": "havic",
    # RawBMamba fills the pipeline's Mamba slot (deepfake_audio calls it
    # 'fakemamba'); see model_archs/fakemamba.py provenance header.
    "fakemamba": "fakemamba",
    # XLSR-Mamba-LA (arXiv 2411.10027, MIT) — wav2vec2-XLSR frontend +
    # bidirectional Mamba backend; see model_archs/xlsrmamba.py provenance
    # header.
    "xlsrmamba": "xlsrmamba",
}


def get_arch(family: str) -> Optional[ArchSpec]:
    """Lazily import *family*'s arch module and return its ArchSpec.

    Returns None for unknown families or importable-but-broken modules.
    The returned spec may still be an unimplemented stub — callers must treat
    ``build()`` raising :class:`ArchNotImplementedError` as 'not ready'.
    """
    if not isinstance(family, str):
        return None
    mod_name = _FAMILIES.get(family)
    if mod_name is None:
        return None
    try:
        import importlib
        mod = importlib.import_module(f".{mod_name}", __package__)
        factory = getattr(mod, "get_arch", None)
        if factory is None:
            return None
        spec = factory()
        return spec if isinstance(spec, ArchSpec) else None
    except Exception:
        return None
