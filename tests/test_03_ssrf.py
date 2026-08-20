"""SSRF-safe fetch guard: inbound URLs must never reach non-public IPs.

Pins _ip_categories / dns_resolve_safe semantics so an attacker cannot route a
fetch to loopback/private/link-local/CGNAT via DNS rebinding or direct IP hosts.
These tests use literal-IP URL objects (no real network) where possible.
"""
from __future__ import annotations

import ipaddress

import pytest

from verisafe.url_guard import (
    _ip_categories,
    SsrfBlocked,
    normalize_url,
    NormalizedUrl,
)


def cats(ip_str):
    return _ip_categories(ipaddress.ip_address(ip_str))


@pytest.mark.parametrize("ip,expect", [
    ("127.0.0.1", "loopback"),
    ("10.0.0.8", "private"),
    ("192.168.1.1", "private"),
    ("169.254.169.254", "link_local"),   # cloud metadata endpoint — the classic SSRF target
    ("172.16.0.1", "private"),
    ("::1", "loopback"),
    ("::ffff:127.0.0.1", "loopback"),     # IPv4-mapped IPv6 must NOT hide a private addr
])
def test_private_range_detection(ip, expect):
    assert expect in cats(ip), f"{ip} should be classified as {expect}: {cats(ip)}"


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
def test_public_ip_not_blocked(ip):
    c = cats(ip)
    assert not (c & {"private", "loopback", "link_local", "unspecified"})


def test_cgnat_100_64_flagged():
    assert "bad_range" in cats("100.64.0.1")


def test_zero_range_flagged():
    assert "bad_range" in cats("0.0.0.1")


# ------------------------------------------------------------ normalization --
def test_normalize_strips_tracking_params_and_schemes():
    n = normalize_url("https://Example.COM/path?utm_source=wa&x=1#frag")
    assert isinstance(n, NormalizedUrl)
    assert n.scheme == "https"
    assert "example.com" in (n.host or "").lower()
    # tracking params dropped, path preserved (NormalizedUrl stores composed url)
    from urllib.parse import urlparse
    u = urlparse(n.url)
    assert "utm_source" not in u.query
    assert u.path == "/path"


def test_normalize_rejects_empty_or_javascript():
    assert normalize_url("") is None
    assert normalize_url("javascript:alert(1)") is None
    assert normalize_url("data:text/html;base64,AAAA") is None


def test_normalize_keeps_port_and_path():
    n = normalize_url("http://a.b:8080/x/y?z=1")
    assert n.port == 8080
    from urllib.parse import urlparse
    assert urlparse(n.url).path == "/x/y"


# ---------------------------------------------------------- resolve-time block --
def test_dns_resolve_safe_blocks_all_loopback(monkeypatch):
    """If *every* resolved address is non-public, SsrfBlocked must be raised."""
    import socket
    import verisafe.url_guard as ug

    def fake_getaddrinfo(host, port=None, *a, **k):
        if host == "evil-loop.internal":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
        return []
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(SsrfBlocked):
        ug.dns_resolve_safe("evil-loop.internal")


def test_dns_resolve_safe_passes_public(monkeypatch):
    import socket
    import verisafe.url_guard as ug
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k:
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 80))])
    ips = ug.dns_resolve_safe("pub.example.com")
    assert "93.184.216.34" in ips
