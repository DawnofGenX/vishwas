"""VirusTotal v3 reputation client — auth, retry/backoff, graceful degradation.

Verdict-mapping rule (documented per build brief D6):
    malicious >= 1  OR  suspicious >= 3   ->  'high'
    suspicious in {1, 2}                  ->  'mid'
    otherwise                             ->  'low'

Design constraints:
- stdlib-only transport (urllib.request); injectable ``opener`` for hermetic tests.
- Auth via VERISAFE_VT_API_KEY (header ``x-apikey``). Absent => unavailable;
  callers keep their existing "not provisioned" phrasing verbatim.
- Rate limiting: honor HTTP 429 Retry-After (sleep capped at 30 s); max 3
  retries per call with exponential backoff (1 s / 4 s / 16 s). On exhaustion
  return a structured 'unavailable' VtResult — NEVER raise into the capability
  layer.
- Timeout 20 s per request; connection errors => 'unavailable'.
- Zero cloud runtime dependency: this module is inert unless a key is set.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

VT_BASE = "https://www.virustotal.com/api/v3"
TIMEOUT_S = 20
MAX_RETRIES = 3
BACKOFF_S = (1.0, 4.0, 16.0)
RETRY_AFTER_CAP_S = 30.0


def api_key() -> str | None:
    """Current VT API key from the environment, or None."""
    k = os.environ.get("VERISAFE_VT_API_KEY", "").strip()
    return k or None


def available() -> bool:
    """True iff a VT API key is provisioned."""
    return api_key() is not None


def base64url(vt_path_component: str) -> str:
    """Base64url-encode a URL or hash for use in a VT v3 path component."""
    return base64.urlsafe_b64encode(vt_path_component.encode("utf-8")).decode("ascii")


@dataclass
class VtResult:
    """Structured outcome of one VT lookup. Never raises; status is one of
    'ok' | 'unavailable' | 'error'."""
    status: str = "unavailable"
    counts: dict[str, int] = field(default_factory=dict)
    raw_status: int | None = None
    note: str = ""
    verdict: str = "low"          # 'high' | 'mid' | 'low' per mapping rule above
    category: str = ""

    @property
    def positives_ratio(self) -> float | None:
        total = sum(self.counts.values())
        if not total:
            return None
        pos = self.counts.get("malicious", 0) + self.counts.get("suspicious", 0)
        return pos / total


def map_verdict(counts: dict[str, int]) -> str:
    """Apply the documented verdict-mapping rule to a stats dict."""
    mal = counts.get("malicious", 0)
    sus = counts.get("suspicious", 0)
    if mal >= 1 or sus >= 3:
        return "high"
    if sus in (1, 2):
        return "mid"
    return "low"


class VtClient:
    """Thin VT v3 client. ``opener`` is injectable for tests; default is
    stdlib urllib.request.urlopen."""

    def __init__(self, opener: Callable[..., Any] | None = None,
                 sleep: Callable[[float], None] | None = None):
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    # ------------------------------------------------------------ public --
    def check_url(self, url: str) -> VtResult:
        return self._get(f"/urls/{base64url(url)}")

    def check_hash(self, sha256: str) -> VtResult:
        return self._get(f"/files/{sha256}")

    # ----------------------------------------------------------- internal --
    def _get(self, path: str) -> VtResult:
        key = api_key()
        if not key:
            return VtResult(status="unavailable", note="VirusTotal API key not provisioned")
        url = VT_BASE + path
        last_note = "unavailable"
        for attempt in range(MAX_RETRIES + 1):
            req = urllib.request.Request(
                url, headers={"x-apikey": key, "Accept": "application/json"})
            try:
                with self._opener(req, timeout=TIMEOUT_S) as r:
                    body = json.loads(r.read().decode())
                data = body.get("data") or {}
                attrs = data.get("attributes") or {}
                counts = _extract_counts(attrs)
                if not counts:
                    return VtResult(status="ok", raw_status=200,
                                    note="VT record exists but carries no analysis stats yet",
                                    verdict="low", category=str(attrs.get("category", "")))
                return VtResult(status="ok", counts=counts, raw_status=200,
                                verdict=map_verdict(counts),
                                category=str(attrs.get("category", "")),
                                note=f"{sum(counts.values())} engines reported")
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < MAX_RETRIES:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    delay = BACKOFF_S[attempt]
                    if ra:
                        try:
                            delay = min(RETRY_AFTER_CAP_S, max(0.0, float(ra)))
                        except ValueError:
                            pass
                    last_note = f"rate-limited (HTTP 429), retried {attempt + 1}x"
                    self._sleep(delay)
                    continue
                if e.code == 429:
                    # rate-limit retry budget exhausted -> structured unavailable
                    return VtResult(status="unavailable", raw_status=429,
                                    note="rate-limited (HTTP 429), retry budget exhausted",
                                    verdict="low")
                if e.code == 404:
                    return VtResult(status="ok", raw_status=404,
                                    note="no VT record for this identifier",
                                    verdict="low")
                return VtResult(status="error", raw_status=e.code,
                                note=f"VirusTotal returned HTTP {e.code}",
                                verdict="low")
            except Exception as e:  # noqa: BLE001 — network/timeout/JSON all degrade
                last_note = f"{type(e).__name__}: {e}"
                if attempt < MAX_RETRIES:
                    self._sleep(BACKOFF_S[attempt])
                    continue
                break
        return VtResult(status="unavailable", note=last_note, verdict="low")


def _extract_counts(attrs: dict) -> dict[str, int]:
    """Pull engine-count stats from either files (last_analysis_stats) or
    urls (last_analysis_results.overview) attribute shapes."""
    stats = attrs.get("last_analysis_stats")
    if isinstance(stats, dict):
        return {k: int(v) for k, v in stats.items() if isinstance(v, (int, float))}
    overview = (attrs.get("last_analysis_results") or {}).get("overview")
    if isinstance(overview, dict):
        return {k: int(v) for k, v in overview.items() if isinstance(v, (int, float))}
    return {}


# --------------------------------------------------------------------------
# Module-level convenience API (capabilities use these)
# --------------------------------------------------------------------------

_default_client: VtClient | None = None


def get_client() -> VtClient:
    global _default_client
    if _default_client is None:
        _default_client = VtClient()
    return _default_client


def check_url(url: str) -> VtResult:
    return get_client().check_url(url)


def check_hash(sha256: str) -> VtResult:
    return get_client().check_hash(sha256)
