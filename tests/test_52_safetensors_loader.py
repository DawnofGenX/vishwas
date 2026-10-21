"""SAFETENSORS CHECKPOINT LOADING — the loader must read .safetensors (2026-10-21).

REGRESSION CONTEXT
------------------
`deploy/vishwas-secrets.env` points the two audio gates at `.safetensors`
files:

    VISHWAS_AASIST_WEIGHTS=/opt/vishwas/models/aasist3/model.safetensors
    VISHWAS_XLSRMAMBA_WEIGHTS=/opt/vishwas/models/xlsr-mamba/model.safetensors

but `model_adapters._default_load()` read every checkpoint with
`torch.load(path)`, which CANNOT read the safetensors format. Measured
failure: `torch.load('.../aasist3/model.safetensors')` raises
`UnpicklingError: unpickling stack underflow` (the file's header magic is
read as a pickle stream).

The per-family ArchSpecs DO know how to read safetensors
(`xlsrmamba._load_safetensors`, `aasist3` loads it inside `build()`), but
those readers are only reachable via `spec.build()`, and `_arch_aware_load`
calls `spec.build()` only AFTER it has already tried `torch.load` and bailed.
So the safetensors path was dead code on the serving path: the audio gates
could never load, on ANY interpreter, with ANY dependency tree.

These tests pin the FIX at the lowest seam (`_default_load`), because that is
where the format decision belongs. They are hermetic: no real weights, no
network, no torch required for the suffix-dispatch assertions.

TDD ORDER: these were written and observed RED before `_default_load` was
changed, then GREEN after.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "/home/hermes/vishwas/src")
from vishwas import model_adapters as ma


def test_default_load_reads_safetensors_by_suffix(tmp_path):
    """A .safetensors checkpoint must be readable by _default_load.

    Uses the real `safetensors` package when available (it is installed for
    /usr/bin/python3 on this host) to write a tiny checkpoint, then asserts
    the loader returns its state dict. If safetensors is unavailable the test
    skips rather than failing — the SUFFIX DISPATCH itself is asserted
    separately below and never needs the package.
    """
    # NOTE: `safetensors.torch` needs TORCH, and the bare CI env has
    # `safetensors` without `torch`. Guarding on "safetensors" alone let the
    # import of torch blow up INSIDE the test body, so the job failed with
    # ModuleNotFoundError: No module named 'torch' instead of skipping.
    pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    from safetensors.torch import save_file
    import torch

    ck = tmp_path / "tiny.safetensors"
    payload = {"layer.weight": torch.zeros(2, 3), "layer.bias": torch.zeros(3)}
    save_file(payload, str(ck))

    raw = ma._default_load(str(ck))
    assert raw is not None, "safetensors checkpoint must load, not return None"
    assert isinstance(raw, dict)
    assert set(raw.keys()) == {"layer.weight", "layer.bias"}


def test_default_load_dispatches_on_suffix_not_torch_load(tmp_path, monkeypatch):
    """The loader must NOT call torch.load for a .safetensors path.

    This is the regression pin: torch.load on a safetensors file raises
    UnpicklingError, which _default_load used to swallow into a silent None.
    We assert the dispatch by poisoning torch.load and requiring the real
    reader to be used instead.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    import torch
    from safetensors.torch import save_file

    ck = tmp_path / "tiny.safetensors"
    save_file({"w": torch.zeros(1, 1)}, str(ck))

    def _boom(*_a, **_k):
        raise AssertionError("torch.load must not be used for .safetensors")

    monkeypatch.setattr(torch, "load", _boom, raising=False)
    raw = ma._default_load(str(ck))
    assert raw is not None, "must not route .safetensors through torch.load"
    assert "w" in raw


def test_default_load_still_uses_torch_load_for_pth(tmp_path, monkeypatch):
    """A .pth checkpoint must keep going through torch.load (no regression)."""
    torch = pytest.importorskip("torch")

    ck = tmp_path / "tiny.pth"
    torch.save({"w": torch.zeros(1, 1)}, str(ck))

    called = {"torch_load": False}
    real_load = torch.load

    def _spy(*a, **k):
        called["torch_load"] = True
        return real_load(*a, **k)

    monkeypatch.setattr(torch, "load", _spy, raising=False)
    raw = ma._default_load(str(ck))
    assert raw is not None
    assert called["torch_load"] is True, ".pth must still use torch.load"


def test_default_load_returns_none_for_missing_file(tmp_path):
    """A missing path must return None (unchanged contract)."""
    assert ma._default_load(str(tmp_path / "nope.safetensors")) is None
    assert ma._default_load(str(tmp_path / "nope.pth")) is None


def test_default_load_returns_none_when_reader_raises(tmp_path, monkeypatch):
    """A corrupt checkpoint must return None, never raise (unchanged contract)."""
    st = pytest.importorskip("safetensors.torch")

    bad = tmp_path / "corrupt.safetensors"
    bad.write_bytes(b"not a real safetensors file")

    def _boom(*_a, **_k):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(st, "load_file", _boom, raising=False)
    assert ma._default_load(str(bad)) is None


def test_safetensors_suffix_is_detected_case_insensitively():
    """The suffix check must not be fooled by case or by a path that merely
    CONTAINS '.safetensors' in a directory name."""
    assert ma._is_safetensors_path("/opt/vishwas/models/aasist3/model.safetensors")
    assert ma._is_safetensors_path("/opt/MODELS/Model.SAFETENSORS")
    assert not ma._is_safetensors_path("/opt/vishwas/models/aasist3/best_model.pth")
    assert not ma._is_safetensors_path("/opt/safetensors.backup/model.pth")


def test_every_configured_safetensors_gate_is_readable():
    """REGRESSION PIN on the real deployment config.

    `deploy/vishwas-secrets.env` points the two audio gates at .safetensors
    files. This test reads THAT config (not a hardcoded path) and asserts every
    gate whose file exists is actually LOADABLE by the seam. Before the fix
    these two returned None (torch.load cannot read safetensors), which
    downgraded the audio channel to missing_dependency and made real deepfake
    audio score LOW.

    Skips when the weights are not provisioned on this host, or when the
    `safetensors` package itself is missing from the running interpreter (the
    hermetic bare-env case) — it is a deployment assertion, not a unit one.
    """
    pytest.importorskip("safetensors.torch")
    secrets = Path("/home/hermes/vishwas/deploy/vishwas-secrets.env")
    if not secrets.exists():
        pytest.skip("vishwas-secrets.env not present")
    gates = {}
    for line in secrets.read_text().splitlines():
        line = line.strip()
        if not line.startswith("VISHWAS_") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        if v and not v.startswith("#"):
            gates[k.strip()] = v
    if not gates:
        pytest.skip("no gates in secrets env")
    checked = 0
    for name, path in sorted(gates.items()):
        if not path.endswith(".safetensors"):
            continue
        p = Path(path)
        if not p.exists():
            continue
        raw = ma._default_load(path)
        assert raw is not None, (
            f"{name} -> {path} is a .safetensors checkpoint but the loader "
            f"returned None; the audio gate would silently fail open"
        )
        assert isinstance(raw, dict) and raw, f"{name} loaded an empty payload"
        checked += 1
    if checked == 0:
        pytest.skip("no .safetensors gate weights provisioned on this host")
