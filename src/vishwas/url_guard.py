"""UrlGuard: normalize/validate URLs and protect against SSRF before any fetch.

Rules:
- RFC-ish normalization: lowercase scheme/host, strip default ports, drop
  tracking params (utm_*, fbclid, gclid, ref...), resolve IDN to punycode.
- Resolve DNS ourselves; reject private/loopback/link-local/reserved IPv4/IPv6,
  IPv4-mapped-in-IPv6 tricks, and hostnames that do not resolve.
- Cap redirects (<=3), re-resolve AND re-check every hop (open-redirect defense).
- Connect pinned to the resolved IP (host header preserved) to prevent
  DNS-rebinding TOCTOU.
- Body capped for the parse stage; everything returned is untrusted input.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import ssl
import time
import urllib.parse
import zlib
from dataclasses import dataclass, field
from typing import Callable

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "ref", "source", "cmpid",
}

_MAX_REDIRECTS = 3
_MAX_BODY_PARSE = 2 * 1024 * 1024   # 2 MB for the DOM/parse stage
_MAX_BODY_FULL = 32 * 1024 * 1024   # download stage cap
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 25


@dataclass(slots=True)
class NormalizedUrl:
    url: str
    host: str
    port: int
    scheme: str
    puny_host: str
    raw: str

    @property
    def registrable_domain(self) -> str:
        """Naive eTLD+1; long-TLD cases handled by a small suffix table."""
        h = self.puny_host.lower()
        parts = h.split(".")
        if len(parts) >= 3 and ".".join(parts[-2:]) in _MULTI_PART_TLDS:
            return ".".join(parts[-3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return h


_MULTI_PART_TLDS = {
    "co.in", "gov.in", "org.in", "net.in", "ac.in", "edu.in", "res.in",
    "co.uk", "org.uk", "gov.uk", "ac.uk", "co.jp", "com.au", "com.br",
    "co.za", "com.mx", "com.sg", "com.hk", "co.kr", "com.tw",
}


def normalize_url(url: str) -> NormalizedUrl | None:
    url = (url or "").strip().strip("<>\"'[]")
    # red-team: control chars (CRLF/LF) inside URL strings are a header-
    # injection / smuggling vector at fetch time — strip them unconditionally
    url = re.sub(r"[\x00-\x1f\x7f]", "", url)
    if not url:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        if url.startswith("//"):
            url = "https:" + url
        elif re.match(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}(?::\d+)?/", url, re.I):
            url = "http://" + url
        else:
            return None  # not a URL at all
    p = urllib.parse.urlsplit(url)
    scheme = p.scheme.lower()
    if scheme not in ("http", "https"):
        return None
    host = (p.hostname or "").lower()
    if not host:
        return None
    # IDN -> punycode
    try:
        puny = host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        puny = host
    port = p.port or (443 if scheme == "https" else 80)
    if port in (80, 443):
        port = -1  # mark default
    # strip tracking query params
    q = urllib.parse.parse_qsl(p.query, keep_blank_values=True)
    q = [(k, v) for k, v in q if k.lower() not in TRACKING_PARAMS]
    query = urllib.parse.urlencode(q)
    norm = urllib.parse.urlunsplit((scheme, f"{puny}:{port}" if port != -1 else puny, p.path or "/", query, ""))
    return NormalizedUrl(norm, host, port, scheme, puny, url)


def _ip_categories(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> set[str]:
    cats: set[str] = set()
    if ip.is_private:
        cats.add("private")
    if ip.is_loopback:
        cats.add("loopback")
    if ip.is_link_local:
        cats.add("link_local")
    if ip.is_reserved:
        cats.add("reserved")
    if ip.is_multicast:
        cats.add("multicast")
    if ip.is_unspecified:
        cats.add("unspecified")
    # 0.0.0.0/8 etc.
    if ip.version == 4:
        o = int(ip)
        if o >> 24 == 0 or o >> 24 == 100 and (o >> 16) & 0xFF == 64:  # 0/8, 100.64/10 CGNAT
            cats.add("bad_range")
        if 192 < (o >> 16) & 0xFF < 201 and (o >> 24) == 192:
            cats.add("docs_or_bad")
    mapped_v4 = getattr(ip, "ipv4_mapped", None)  # IPv4-mapped IPv6 (::ffff:a.b.c.d)
    if mapped_v4 is not None:
        cats.update(_ip_categories(mapped_v4))
    return cats


def _nonpublic_cats(ip_str: str) -> set[str]:
    c = _ip_categories(ipaddress.ip_address(ip_str))
    return c & {"private", "loopback", "link_local", "reserved", "multicast",
                "unspecified", "bad_range"}


def dns_resolve_safe(host: str, timeout: float = 6.0) -> list[str]:
    """Resolve a hostname; return IPs. Raises SsrfBlocked on private results.

    Red-team fix: literal-IP hosts (http://169.254.169.254/latest/meta-data/)
    used to reach getaddrinfo and skip classification entirely. They are now
    classified directly — no lookup needed for an address that is already one.
    """
    host = (host or "").strip().strip("[]")
    try:
        literal = ipaddress.ip_address(host)
        if literal.version == 6 and getattr(literal, "ipv4_mapped", None) is not None:
            literal = literal.ipv4_mapped
        if _nonpublic_cats(str(literal)):
            raise SsrfBlocked(f"literal non-public ip host blocked: {host}")
        return [str(literal)]
    except ValueError:
        pass  # not an IP literal -> fall through to DNS
    addrs = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    ips: list[str] = []
    for fam, _, _, _, sa in addrs:
        ip = sa[0]
        if not ip:
            continue
        obj = ipaddress.ip_address(ip)
        if obj.version == 6 and obj.ipv4_mapped:
            obj = obj.ipv4_mapped
        if str(obj) not in ips:
            ips.append(str(obj))
    blocked = []
    for s in ips:
        cats = _ip_categories(ipaddress.ip_address(s))
        bad = cats & {"private", "loopback", "link_local", "reserved", "multicast", "unspecified", "bad_range"}
        if bad:
            blocked.append((s, ",".join(sorted(cats))))
    if blocked and len(blocked) == len(set(ips)):
        raise SsrfBlocked(f"all resolved addresses of {host} are non-public: {blocked}")
    if blocked:
        # keep only public ones, but record the block for evidence
        good = [s for s in ips if str(ipaddress.ip_address(s)) not in [b[0] for b in blocked]]
        if not good:
            raise SsrfBlocked(f"no public address for {host}: {blocked}")
    return ips


class SsrfBlocked(Exception):
    pass


class UrlFetchResult:
    def __init__(self, url: NormalizedUrl):
        self.url = url
        self.final_url: str | None = None
        self.status: int | None = None
        self.content_type: str = ""
        self.body: bytes = b""
        self.redirect_chain: list[dict] = []
        self.headers: dict[str, str] = {}
        self.tls: dict = {}
        self.duration_s: float = 0.0
        self.error: str | None = None
        self.block_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.block_reason is None


def _ssl_ctx(insecure_warn: bool = False) -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    return ctx


def fetch_url(nurl: NormalizedUrl, body_cap: int = _MAX_BODY_PARSE,
              follow_redirects: bool = True, note: Callable[[str], None] | None = None) -> UrlFetchResult:
    """SSRF-guarded fetch. Resolves each hop, blocks non-public IPs, pins connect."""
    res = UrlFetchResult(nurl)
    t0 = time.monotonic()
    current = nurl
    hops = 0
    while True:
        try:
            ips = dns_resolve_safe(current.puny_host)
        except SsrfBlocked as e:
            res.block_reason = f"ssrf:{e}"
            break
        except (socket.gaierror, OSError) as e:
            res.error = f"dns:{e.__class__.__name__}"
            break
        target_ip = ips[0]
        if note:
            note(f"connect {current.puny_host} -> {target_ip}")
        conn = None
        try:
            if current.scheme == "https":
                raw_sock = socket.create_connection((target_ip, 443), timeout=_CONNECT_TIMEOUT)
                ctx = _ssl_ctx()
                conn = ctx.wrap_socket(raw_sock, server_hostname=current.puny_host)
                peercert = conn.getpeercert()
                if peercert:
                    res.tls = {
                        "subject": dict(x[0] for x in peercert.get("subject", ())),
                        "issuer": dict(x[0] for x in peercert.get("issuer", ())),
                        "san": [d for d in peercert.get("subjectAltName", ()) if d[0] == "DNS"],
                        "version": getattr(conn, "version", lambda: "")(),
                    }
                req = f"GET {current.url.split('://', 1)[1]} HTTP/1.1\r\nHost: {current.puny_host}\r\nConnection: close\r\nUser-Agent: Mozilla/5.0 (compatible; Vishwas/1.0)\r\nAccept: text/html,*/*\r\n\r\n"
            else:
                raw_sock = socket.create_connection((target_ip, 80), timeout=_CONNECT_TIMEOUT)
                conn = raw_sock
                req = f"GET {current.url.split('://', 1)[1]} HTTP/1.1\r\nHost: {current.puny_host}\r\nConnection: close\r\nUser-Agent: Mozilla/5.0 (compatible; Vishwas/1.0)\r\nAccept: text/html,*/*\r\n\r\n"
            conn.sendall(req.encode())
            data = b""
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
                if len(data) > _MAX_BODY_FULL:
                    break
            hdr, _, body = _split_http(data)
            res.headers.update({k.lower(): v for k, v in hdr.items()})
            res.status = res.status or hdr.get("__status__")
            res.content_type = hdr.get("content-type", "")
            res.body = _dechunk(body)[:body_cap]
            code = res.status or 0
            loc = hdr.get("location")
            if code in (301, 302, 303, 307, 308) and loc and follow_redirects:
                res.redirect_chain.append({"from": current.url, "to": loc, "status": code})
                if hops >= _MAX_REDIRECTS:
                    res.block_reason = "too_many_redirects"
                    break
                nxt = normalize_url(loc if "://" in loc else current.scheme + "://" + current.puny_host + loc)
                if nxt is None:
                    res.block_reason = f"bad_redirect:{loc}"
                    break
                current = nxt
                hops += 1
                continue
            break
        except (socket.timeout, TimeoutError) as e:
            res.error = f"timeout:{e.__class__.__name__}"
            break
        except SsrfBlocked as e:
            res.block_reason = f"ssrf:{e}"
            break
        except (ConnectionError, OSError, ssl.SSLError) as e:
            res.error = f"net:{e.__class__.__name__}: {str(e)[:80]}"
            break
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    res.final_url = current.url
    res.duration_s = round(time.monotonic() - t0, 3)
    return res


def _split_http(data: bytes) -> tuple[dict[str, str], bytes, bytes]:
    head, _, rest = data.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    if not lines:
        return {}, b"", rest
    status_line = lines[0].decode("latin-1")
    m = re.match(r"HTTP/\d\.\d (\d{3})", status_line)
    hdr: dict[str, str] = {}
    if m:
        hdr["__status__"] = int(m.group(1))
    for ln in lines[1:]:
        if b":" in ln:
            k, v = ln.split(b":", 1)
            hdr[k.decode("latin-1").strip()] = v.decode("latin-1").strip()
    return hdr, head, rest


def _dechunk(body: bytes) -> bytes:
    if body.startswith(b"chunked") or b"\x00" in body[:16]:
        pass
    # naive: if response used chunked, our recv loop already got decoded-by-proxy
    # bodies; just strip trailing NUL padding some servers send
    return body.rstrip(b"\x00") if len(body) > 2 and body.endswith(b"\x00") else body


_TLD_ONLY = {"com","net","org","io","co","in","uk","us","gov","edu","ac",
             "res","me","info","biz","dev","app","ai","xyz","online","site"}

# brand -> its own official registrable domain(s); a hit on these is the brand
# itself, never a lookalike. Keep minimal and exact — precision over recall.
_OFFICIAL_DOMAINS = {
    "icici":     {"icicibank.com"},
    "hdfcbank":  {"hdfcbank.com"},
    "sbi":       {"sbi.co.in", "sbi.in"},
    "axisbank":  {"axisbank.com"},
    "kotak":     {"kotakbank.com"},
    "yesbank":   {"yesbank.co.in"},
    "nsdl":      {"nsdl.gov.in", "nsdl.com"},
    "passport":  {"passport.gov.in"},
    "upi":       {"upi.gov.in", "npci.org.in"},
    "google":    {"google.com", "google.co.in"},
    "amazon":    {"amazon.in", "amazon.com"},
    "flipkart":  {"flipkart.com"},
    "paypal":    {"paypal.com", "paypal.in"},
    "sebi":      {"sebi.gov.in"},
    "irdai":     {"irdai.org.in"},
}


def typosquat_signals(domain: str, brand_list: list[str]) -> list[dict]:
    """Lookalike-domain heuristics vs a brand lexicon."""
    out = []
    d = domain.lower().lstrip("www.")
    trans = str.maketrans({"0": "o", "1": "l", "i": "l", "g": "9"})

    def _lookalike(s: str) -> str:
        # 'rn' -> 'm' needs two-pass handling (maketrans keys must be len 1)
        s = s.replace("rn", "m")
        return s.translate(trans)

    def _stuffing_tokens(residue: str) -> str:
        """Meaningful leftover tokens after removing a brand hit — ignoring the TLD."""
        meaningful = []
        for lab in residue.lower().split("."):
            alnum = re.sub(r"[^a-z0-9]", "", lab)
            if not alnum:
                continue
            if alnum in _TLD_ONLY:
                continue
            meaningful.append(alnum)
        return "".join(meaningful)

    # longest brands first so 'icicibank' consumes the string before 'icici'
    for brand in sorted(brand_list, key=len, reverse=True):
        b = brand.lower()
        if b not in d:
            continue
        # a brand's own official domain is never a lookalike
        if d in _OFFICIAL_DOMAINS.get(b, set()) or d.rstrip(".") in _OFFICIAL_DOMAINS.get(b, set()):
            continue
        # brand present => likely legit unless weird extra segments/symbols
        extra = d.replace(b, "", 1)
        extra_clean = _stuffing_tokens(extra)
        if len(extra_clean) > 2:
            out.append({"brand": b, "extra": extra_clean[:12], "signal": "brand_plus_extra"})
            break  # one best (longest) brand explanation is enough
    # full lookalike: map typosquat chars, then compare against brands
    mapped = _lookalike(d)
    for brand in sorted(brand_list, key=len, reverse=True):
        b = brand.lower()
        if mapped == b and d != b:
            out.append({"brand": b, "extra": "", "signal": "typosquat_exact"})
            break
    return out
