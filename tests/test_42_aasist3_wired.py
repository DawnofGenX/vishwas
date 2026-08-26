"""aasist3 (Spectra-AASIST3) wiring tests — registry + production polarity.

MUST stay green in the hermetic suite (bare ``PYTHONPATH=src``, no real torch /
no weights). Therefore: no top-level torch / aasist3 / safetensors imports —
everything heavy is imported lazily inside a guarded branch or a weights-gated
test. Real-torch assertions use the same honest-degradation contract as
``test_14_model_adapters.test_get_arch_lazy_registry``: under a stub-torch tree
the ``'aasist3'`` module (which ``import torch.nn.functional``) fails to import,
so ``get_arch`` honestly returns None.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from vishwas.model_archs import _FAMILIES
import vishwas.model_adapters as ma


def _real_torch() -> bool:
    try:
        import torch.nn  # noqa: F401
        return True
    except Exception:
        return False


def _weights_available() -> bool:
    p = os.environ.get("VISHWAS_AASIST_WEIGHTS", "")
    return bool(p and Path(p).exists())


def test_aasist3_registered_in_family_registry():
    """(a) family registry maps 'aasist3' -> its owning module (pure dict, no import)."""
    assert _FAMILIES.get("aasist3") == "aasist3"


def test_aasist3_routes_from_weights_env():
    """(b) VISHWAS_AASIST_WEIGHTS routes to arch family 'aasist3' (was 'aasist')."""
    assert ma._ARCH_FAMILIES.get("VISHWAS_AASIST_WEIGHTS") == "aasist3"


def test_aasist3_get_arch_returns_spectra_spec():
    """(a) get_arch('aasist3') resolves to a SpectraAASIST3Spec when torch is real."""
    from vishwas.model_archs import get_arch
    from vishwas.model_archs.base import ArchSpec

    got = get_arch("aasist3")
    if _real_torch():
        from vishwas.model_archs.aasist3 import SpectraAASIST3Spec
        assert isinstance(got, SpectraAASIST3Spec)
        assert got.name == "aasist3"
        assert got.weight_env == "VISHWAS_AASIST_WEIGHTS"
        assert got.implemented is True
        # vendored build() overrides the base stub (not a placeholder)
        assert got.build.__func__ is not ArchSpec.build
    else:
        # hermetic stub-torch: aasist3 module fails to import -> honest None
        # (identical degradation to the aasist/effort families)
        assert got is None


@pytest.mark.skipif(
    not _weights_available() or not _real_torch(),
    reason="VISHWAS_AASIST_WEIGHTS not provisioned or real torch absent",
)
def test_aasist3_weights_gated_load_and_finite_score():
    """(c) weights-gated: loads via resolve to a READY ArchModelWrapper, full key
    coverage, and a finite in-range score on a tiny tile-windowed waveform."""
    import numpy as np
    from safetensors.torch import load_file

    env = os.environ["VISHWAS_AASIST_WEIGHTS"]
    ad = ma.resolve("VISHWAS_AASIST_WEIGHTS")
    assert ad is not None
    m = ad.load(env)
    assert m is not None
    assert isinstance(m, ma.ArchModelWrapper)

    net = m.model
    sd = load_file(env)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0
    assert len(sd) == 1022  # documented full checkpoint coverage

    wav = np.zeros(16000, dtype=np.float32)
    wav[::137] = 0.1
    p = m.predict(wav)  # ArchModelWrapper.predict -> aasist3 spec .score
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0