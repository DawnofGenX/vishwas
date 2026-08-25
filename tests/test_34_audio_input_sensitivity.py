"""Input-sensitivity regression tests for the learned audio gates.

2026-08-25 incident: both vendored audio detectors were FULLY INPUT-INVARIANT —
AASIST returned the same 0.997152 spoof posterior for silence, white noise and
a 440 Hz sine, and XLSR-Mamba returned 0.154 for everything. Root cause: the
hand-copied frontend configs set ``layer_norm_first=False`` (post-norm), while
both checkpoints were trained with stable-layer-norm PRE-norm frontends
(WavLM-Large.pt and XLSR-300M are do_stable_layer_norm=true models). Under
post-norm, each layer's LN-after-residual erases the input-dependent part of
the residual stream faster than attention re-injects it; after ~24 layers the
encoder output is input-independent to float32 precision.

The config flags themselves are pinned below (they cannot be derived from
checkpoint tensor shapes — booleans are shape-invisible). The weights-gated
E2E tests prove actual input sensitivity whenever a checkpoint is provisioned
and auto-skip otherwise (hermetic suite stays green).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Pre-bound placeholders so nothing below can hit NameError in degraded envs.
_wavlm_large_config = None
_SSL_CFG_OVERRIDES = None

try:  # pragma: no cover - environment-dependent (hermetic torch stub)
    import torch
    import torch.nn as nn  # noqa: F401  (the stub lacks torch.nn)
    from vishwas.model_archs.aasist import _wavlm_large_config
    from vishwas.model_archs.xlsrmamba import _SSL_CFG_OVERRIDES
    _IMPORT_ERROR = None
except Exception as e:
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"


def test_audio_frontends_are_prenorm():
    """Both WavLM-Large / XLSR-300M frontends must build in prenorm mode.

    Post-norm (False) is the collapsed configuration: LN-after-residual kills
    the input signal through the stack (see module docstring). This pin is the
    whole bug class in one assertion.
    """
    if _IMPORT_ERROR is not None:
        pytest.skip(f"real torch / vendor tree unavailable ({_IMPORT_ERROR})")
    assert _wavlm_large_config().layer_norm_first is True
    assert _SSL_CFG_OVERRIDES["layer_norm_first"] is True


# ---------------------------------------------------------------------------
# Weights-gated E2E: real checkpoint + real forward pass, three distinct inputs.
# ---------------------------------------------------------------------------

def _three_inputs(samples: int) -> dict[str, np.ndarray]:
    t = np.arange(samples) / 16000.0
    rng = np.random.default_rng(0)
    return {
        "silence": np.zeros(samples, dtype=np.float32),
        "noise": (rng.standard_normal(samples) * 0.05).astype(np.float32),
        "sine440": (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
    }


def _weights_available(env: str) -> str | None:
    if os.environ.get("VISHWAS_SKIP_WEIGHTED"):
        return None
    path = os.environ.get(env)
    return path if path and os.path.exists(path) else None


@pytest.mark.skipif(_weights_available("VISHWAS_AASIST_WEIGHTS") is None,
                    reason="VISHWAS_AASIST_WEIGHTS not provisioned")
def test_aasist_distinguishes_real_inputs_end_to_end():
    """The loaded detector must give DIFFERENT outputs for different inputs.

    The 2026-08-25 failure mode was bit-identical logits for every waveform;
    this test fails if that ever regresses. Goes through the production seam
    (adapter load -> ArchModelWrapper.predict -> repeat-pad -> softmax).
    """
    from vishwas.model_adapters import resolve

    gate = resolve("VISHWAS_AASIST_WEIGHTS")
    model = gate.load(_weights_available("VISHWAS_AASIST_WEIGHTS"))
    assert model is not None, "AASIST weights present but load() failed"

    probs = {n: float(model.predict(w))
             for n, w in _three_inputs(64000).items()}
    vals = list(probs.values())
    spread = max(vals) - min(vals)
    assert len(set(round(v, 6) for v in vals)) == 3, \
        f"input-invariant outputs (regression): {probs}"
    assert spread > 1e-3, f"suspiciously small spread {spread}: {probs}"


@pytest.mark.skipif(_weights_available("VISHWAS_XLSRMAMBA_WEIGHTS") is None,
                    reason="VISHWAS_XLSRMAMBA_WEIGHTS not provisioned")
def test_xlsrmamba_distinguishes_real_inputs_end_to_end():
    """XLSR-Mamba must separate silence/noise/sine (was constant 0.154)."""
    from vishwas.model_adapters import resolve

    gate = resolve("VISHWAS_XLSRMAMBA_WEIGHTS")
    model = gate.load(_weights_available("VISHWAS_XLSRMAMBA_WEIGHTS"))
    assert model is not None, "XLSR weights present but load() failed"

    probs = {n: float(model.predict(w))
             for n, w in _three_inputs(66800).items()}
    vals = list(probs.values())
    spread = max(vals) - min(vals)
    assert len(set(round(v, 6) for v in vals)) == 3, \
        f"input-invariant outputs (regression): {probs}"
    assert spread > 1e-3, f"suspiciously small spread {spread}: {probs}"
