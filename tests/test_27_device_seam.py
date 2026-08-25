"""test_27_device_seam.py — CUDA device-resolution seam (hermetic).

Task 1 of the GPU-rebuild plan: one seam (vishwas.device.resolve_device)
governs where learned-model weights load and tensors run.

Resolution contract:
  1. VISHWAS_DEVICE env ("cpu"/"cuda", case-insensitive) — operator override
  2. "cuda" when torch reports cuda available
  3. "cpu" fallback — including when torch import fails entirely; never raises

Stubbing pattern mirrors the repo's hermetic style (see test_26): no real GPU
or torch install is required — sys.modules is patched with a MagicMock torch.
"""
import os
from unittest import mock

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("VISHWAS_DEVICE", raising=False)


def test_resolve_device_prefers_cuda_when_available(clean_env, monkeypatch):
    from vishwas.device import resolve_device
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert resolve_device() == "cuda"


def test_resolve_env_override_wins_even_with_cuda(clean_env, monkeypatch):
    from vishwas.device import resolve_device
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setenv("VISHWAS_DEVICE", "cpu")
    assert resolve_device() == "cpu"   # explicit override beats autodetect


def test_resolve_env_override_case_insensitive_cuda(clean_env, monkeypatch):
    from vishwas.device import resolve_device
    monkeypatch.setenv("VISHWAS_DEVICE", "  CUDA ")
    # strict: with torch reporting no cuda, the override must still say cuda
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert resolve_device() == "cuda"


def test_resolve_falls_back_to_cpu_when_torch_missing(clean_env):
    from vishwas.device import resolve_device
    import sys
    with mock.patch.dict(sys.modules, {"torch": None}):
        assert resolve_device() == "cpu"


def test_resolve_falls_back_to_cpu_when_cuda_unavailable(clean_env, monkeypatch):
    from vishwas.device import resolve_device
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = False
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert resolve_device() == "cpu"


def test_resolve_never_raises_on_torch_import_error(clean_env):
    """Pitfall guard: the seam must not add new silent/raising failure paths."""
    from vishwas.device import resolve_device
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "torch":
            raise RuntimeError("torch exploded")
        return real_import(name, *a, **k)

    with mock.patch.object(builtins, "__import__", side_effect=boom):
        assert resolve_device() == "cpu"
