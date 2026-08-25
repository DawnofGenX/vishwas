"""test_32_vt_cache.py — VT TTL cache: quota economics + repeat-forward speedup.

Free tier = 500 req/day; identical forwards must not re-spend quota.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vishwas import vt_client
from vishwas.vt_client import VtResult


def test_cache_hit_does_not_respend(monkeypatch):
    vt_client._vt_cache.clear()
    calls = {"n": 0}

    def fake_check_hash(self, sha256):  # noqa: ANN001
        calls["n"] += 1
        return VtResult(status="ok", counts={"malicious": 60}, raw_status=200)

    monkeypatch.setattr(vt_client.VtClient, "check_hash", fake_check_hash)
    r1 = vt_client.check_hash("a" * 64)
    r2 = vt_client.check_hash("a" * 64)
    assert calls["n"] == 1, "second identical lookup must come from cache"
    assert r2.counts == {"malicious": 60}
    assert r1 is r2 or (r1.status == r2.status == "ok")


def test_failures_are_not_cached(monkeypatch):
    vt_client._vt_cache.clear()
    calls = {"n": 0}

    def flaky(self, sha256):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            return VtResult(status="unavailable", note="transport down")
        return VtResult(status="ok", counts={"malicious": 0}, raw_status=200)

    monkeypatch.setattr(vt_client.VtClient, "check_hash", flaky)
    vt_client.check_hash("b" * 64)   # fails -> NOT cached
    r2 = vt_client.check_hash("b" * 64)
    assert calls["n"] == 2, "failed lookups must retry, never cache the failure"
    assert r2.status == "ok"


def test_ttl_expiry(monkeypatch):
    vt_client._vt_cache.clear()
    import time as _t
    vt_client._vt_cache["hash:" + "c" * 64] = (_t.time() - 10_000_000,
                                               VtResult(status="ok"))
    calls = {"n": 0}

    def fresh(self, sha256):  # noqa: ANN001
        calls["n"] += 1
        return VtResult(status="ok", counts={}, raw_status=200)

    monkeypatch.setattr(vt_client.VtClient, "check_hash", fresh)
    vt_client.check_hash("c" * 64)
    assert calls["n"] == 1, "expired entry must be refetched"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
