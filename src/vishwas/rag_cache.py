"""RAG template-cache reader (retrieval-cache signal only, never source of truth).

Pure, hermetic, no network. Reads the JSON index produced by
``scripts/build_rag_cache.py``. Contract mirrors the rest of Vishwas:
nothing here raises on bad input — malformed/absent cache degrades to an
empty view so callers can emit a silent "feature OFF" signal (mirroring the
docling/pades gate idiom).

Freshness contract: an entry older than ``staleness_days`` (or whose version
mismatches the active ``VISHWAS_RAG_VERSION``) must DEGRADE downstream
confidence, never block. Callers treat a stale/absent cache as "no signal",
not as a failure.

Cache-level freshness gate (``cache_freshness`` / ``cache_stale``): the whole
index is stale when its ``built_utc`` is missing/unparseable, older than
``CACHE_TTL_DAYS`` (14), or its recorded ``source_digest`` sha256 does not
match the latest ``apisetu_catalog_digest_*.json`` in the catalog dir
(``VISHWAS_RAG_DIGEST_DIR``, default ``<repo>/docs/research/data``).
Honest defaults, fail-closed: a cache with NO build timestamp is stale (we
cannot claim freshness we cannot see); a cache with NO recorded source digest
is likewise stale (pre-gate caches must be rebuilt once to gain provenance).
Staleness degrades — consumers drop their confidence contribution silently,
exactly like an absent cache; it is never an error surface.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path

_FALLBACK_DIR = "/home/hermes/rag-cache"
_FALLBACK_VERSION = "1"
CACHE_TTL_DAYS = 14          # cache-level TTL for built_utc (Phase 2 task 2.4)
_CATALOG_GLOB = "apisetu_catalog_digest_*.json"   # date-suffixed -> name-sorted = chronological


def _cache_dir() -> Path:
    """Resolve VISHWAS_RAG_CACHE lazily (per call, not at import)."""
    return Path(os.environ.get("VISHWAS_RAG_CACHE", _FALLBACK_DIR))


def _default_index() -> Path:
    return _cache_dir() / "cache-index.json"


def _default_version() -> str:
    return os.environ.get("VISHWAS_RAG_VERSION", _FALLBACK_VERSION)


def _parse_utc(s: str | None) -> _dt.datetime | None:
    if not s:
        return None
    try:
        return _dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc)
    except ValueError:
        try:
            return _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None


def load(path: str | Path | None = None) -> dict:
    """Load the cache index; returns {} when absent/unreadable/malformed."""
    p = Path(path) if path else _default_index()
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def version_of(index: dict) -> str:
    return str(index.get("version", ""))


def staleness_days(index: dict) -> int:
    try:
        return int(index.get("staleness_days", 90))
    except (TypeError, ValueError):
        return 90


def freshness(entry: dict, now: _dt.datetime | None = None,
              index: dict | None = None) -> bool:
    """True when *entry* is fresh enough to contribute a confidence signal.

    An entry is fresh when its ``fetched_utc`` is within ``staleness_days`` of
    *now*. Entries without a fetch timestamp (derived-only templates) are
    considered fresh by construction — they carry no live provenance to go
    stale. A missing/stale timestamp degrades (returns False), never raises.
    """
    fetched = _parse_utc(entry.get("fetched_utc"))
    if fetched is None:
        return True                       # derived entry: no live clock to age
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    days = staleness_days(index or {})
    age = (now - fetched).total_seconds() / 86400.0
    return age <= days


# ------------------------------------------------- cache-level freshness ----
def _default_digest_dir() -> Path:
    """Catalog dir: VISHWAS_RAG_DIGEST_DIR or <repo>/docs/research/data."""
    env = os.environ.get("VISHWAS_RAG_DIGEST_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "docs" / "research" / "data"


def built_at(index: dict) -> _dt.datetime | None:
    """Parse the builder's ``built_utc``; None when missing/unparseable."""
    return _parse_utc(index.get("built_utc"))


def source_digest_of(index: dict) -> str | None:
    """sha256 the builder recorded for the catalog file it consumed."""
    sd = index.get("source_digest")
    if isinstance(sd, dict):
        v = sd.get("sha256")
        return str(v) if v else None
    if isinstance(sd, str) and sd:
        return sd
    return None


def latest_catalog(digest_dir: str | Path | None = None) -> Path | None:
    """Newest ``apisetu_catalog_digest_*.json`` in the catalog dir, or None.

    Filenames carry ISO dates, so lexicographic order == chronological.
    """
    d = Path(digest_dir) if digest_dir else _default_digest_dir()
    try:
        cands = sorted(d.glob(_CATALOG_GLOB))
    except OSError:
        return None
    return cands[-1] if cands else None


def file_sha256(path: str | Path) -> str | None:
    """sha256 hex digest of a file; None when unreadable (never raises)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def cache_freshness(index: dict, now: _dt.datetime | None = None,
                    digest_dir: str | Path | None = None) -> tuple[bool, str]:
    """Cache-level freshness gate. Returns ``(fresh, reason)``.

    Fresh only when ALL hold:
      * ``built_utc`` present and parseable  -> else ``missing-build-timestamp``
      * built within ``CACHE_TTL_DAYS`` (14) -> else ``ttl-expired``
      * recorded ``source_digest`` sha256 equals the sha256 of the LATEST
        catalog file in the digest dir       -> else ``missing-source-digest``
        / ``catalog-unavailable`` / ``digest-mismatch``

    Honest fail-closed defaults: no timestamp, or no recorded digest, means
    stale — we never claim a freshness we cannot verify. Never raises.
    """
    built = built_at(index or {})
    if built is None:
        return False, "missing-build-timestamp"
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    if (now - built).total_seconds() > CACHE_TTL_DAYS * 86400.0:
        return False, "ttl-expired"
    recorded = source_digest_of(index)
    if not recorded:
        return False, "missing-source-digest"
    cat = latest_catalog(digest_dir)
    if cat is None:
        return False, "catalog-unavailable"
    actual = file_sha256(cat)
    if actual is None or actual != recorded:
        return False, "digest-mismatch"
    return True, "ok"


def cache_stale(index: dict, now: _dt.datetime | None = None,
                digest_dir: str | Path | None = None) -> bool:
    """True when the cache fails the freshness gate (see cache_freshness)."""
    fresh, _ = cache_freshness(index, now=now, digest_dir=digest_dir)
    return not fresh


def get_entries(cls: str, index: dict | None = None) -> list[dict]:
    """All entries under ``entries.<cls>``, as a stable list of dicts.

    Handles both mapping-valued classes (document_templates, issuer_trust ->
    list of dicts keyed by name) and list-valued classes (qr_schemes,
    official_content_baselines). Unknown/absent class -> [].
    """
    idx = index if index is not None else load()
    entries = idx.get("entries", {})
    node = entries.get(cls)
    if node is None:
        return []
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault("_key", k)
                out.append(item)
            else:
                out.append({"_key": k, "_value": v})
        return out
    if isinstance(node, list):
        return [x for x in node if isinstance(x, dict)]
    return []


def available(path: str | Path | None = None) -> bool:
    """Gate predicate: True when the index exists AND parses non-empty."""
    idx = load(path)
    return bool(idx.get("entries"))


def template_for(doc_type: str | None, index: dict | None = None) -> dict | None:
    """Legacy-view lookup used by gov_document._rag_cache.

    Returns the ``templates[doc_type]`` record (with ``required_fields`` +
    ``version``) or None when absent. Version-mismatch is NOT filtered here —
    the caller decides whether to degrade, keeping this a pure read.
    """
    idx = index if index is not None else load()
    t = idx.get("templates", {})
    e = t.get(doc_type or "")
    return e if isinstance(e, dict) else None
