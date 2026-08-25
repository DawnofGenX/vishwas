"""test_29_health_device.py — /health reports the resolved inference device (hermetic)."""
import importlib
import sys
import types


def _fresh_app_module():
    """Import app.py fresh with a stubbed quarantine module (no HERMES_HOME writes)."""
    for m in [m for m in list(sys.modules) if m.startswith("vishwas.app")]:
        del sys.modules[m]
    if "vishwas.quarantine" not in sys.modules:
        stub = types.ModuleType("vishwas.quarantine")
        stub.__dict__["count_open_quarantines"] = lambda: 0
        sys.modules["vishwas.quarantine"] = stub
    app_mod = importlib.import_module("vishwas.app")
    return importlib.reload(app_mod)


def test_health_snapshot_reports_cpu(monkeypatch):
    monkeypatch.setenv("VISHWAS_DEVICE", "cpu")
    app = _fresh_app_module()
    snap = app.health_snapshot(processor=None, deps=["cv2"])
    assert snap["device"] == "cpu"


def test_health_snapshot_schema_unchanged_plus_device(monkeypatch):
    """device is ADDITIVE: legacy keys still present (backward-compat contract)."""
    monkeypatch.setenv("VISHWAS_DEVICE", "cpu")
    app = _fresh_app_module()
    snap = app.health_snapshot(processor=None, deps=[])
    for key in ("status", "uptime_s", "quarantines_open", "deps", "deps_available"):
        assert key in snap
