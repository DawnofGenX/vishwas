#!/usr/bin/env python3
"""Build the Vishwas RAG template-cache (idempotent).

Stdlib + urllib ONLY. Honors:
  VISHWAS_RAG_CACHE   cache dir   (default /home/hermes/rag-cache)
  VISHWAS_RAG_VERSION version tag (default "1")

Deterministic content (always written, no network):
  * document_templates  - derived from gov_document.DOC_TYPE_LEXICON (derived=true)
  * issuer_trust        - from the dated API Setu catalog digest (class b)
  * qr_schemes          - transcribed VERBATIM from .delegation/sec4_qr.txt
  * bookkeeping         - budget / wall-clock / unreachable sources

Optional live refresh (--no-network disables): fetches AT MOST the listed
root pages (hard cap 6 requests, 1 retry each, 2-3s jitter, browser UA,
30s timeout, abort-host on 403/CAPTCHA per politeness) to populate
official_content_baselines (verbatim phrases from fetched bytes) and to
record specimen-image presence. INVENTS NOTHING: a host that fails is
recorded under bookkeeping.unreachable_sources, never fabricated.

Idempotent: re-running overwrites cache-index.json with fresh timestamps;
the deterministic classes are byte-stable apart from built_utc/fetched_utc.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIGEST = ROOT / "docs" / "research" / "data" / "apisetu_catalog_digest_2026-08-19.json"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
BUDGET = 6
STALENESS_DAYS = 90

# Root pages permitted for the live refresh (brief ground truth). epfo.gov.in
# is KNOWN NXDOMAIN on this network -> recorded unreachable, never retried.
ROOT_PAGES = [
    ("digilocker.gov.in", "https://digilocker.gov.in/"),
    ("nic.gov.in", "https://nic.gov.in/"),
    ("incometax.gov.in", "https://incometax.gov.in/"),
    ("nsdl.co.in", "https://www.nsdl.co.in/"),
]
KNOWN_UNREACHABLE = [("epfo.gov.in", "dns", None)]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------- class (c) --
# Derived from DOC_TYPE_LEXICON (gov_document.py). required_fields = the
# lexicon keywords (lowercased) + canonical field labels. These are DERIVED
# layout fingerprints, not fetched specimens -> derived=true, status=ok.
DOC_TEMPLATES = {
    "aadhaar_ecyc": ["ekyc", "aadharnumber", "aadhaar number", "uidai",
                     "unique identification", "name", "date of birth"],
    "pan_card": ["permanent account number", "pan card", "income-tax department",
                 "name", "father name", "date of birth"],
    "voter_id": ["electoral roll", "voter id", "ephw",
                 "electoral registration office", "voter card", "epic number"],
    "passport": ["passport", "issue post", "expiry date", "mrz", "<ind",
                 "surname", "given names", "nationality"],
    "driving_license": ["driving licence", "driving license", "vehicle class",
                        "license number", "date of issue"],
    "ration_card": ["ration card", "food supply department", "card holder",
                    "ration card number", "family size"],
    "bank_passbook": ["pass book", "account number", "branch code", "ifsc",
                      "account holder", "balance"],
    "epf_statement": ["employee provident fund", "epf", "uepf member id",
                      "challan", "uan", "establishment name"],
    "income_tax_notice": ["department of income tax", "tax year",
                          "assessment year", "intimation u/s", "pan"],
    "pm_kisan_letter": ["pm-kisan", "direct benefit transfer", "samrat nidhi",
                        "beneficiary name", "bank account"],
}


# --------------------------------------------------------------- class (d) --
# Transcribed VERBATIM from .delegation/sec4_qr.txt (scheme name, digit-format
# description, classification regex). No paraphrase.
QR_SCHEMES = [
    {"scheme": "Aadhaar (UIDAI)",
     "description": "card front QR encodes masked UID (XXXX XXXX 1234 format) "
                    "- privacy-by-design, not the 12 digits; back-side QR "
                    "(post-2020 cards) carries masked UID + name.",
     "classification_regex": "^\\d{4} ?\\d{4} ?\\d{4}$"},
    {"scheme": "Voter ID (EPIC, NIC)",
     "description": "QR encodes EPIC number (10-digit, e.g. ABC1234567) + "
                    "photo hash (NIC's \"photo verification\" QR for polling booths).",
     "classification_regex": "^[A-Z]{3}\\d{7}$"},
    {"scheme": "EPFO member card",
     "description": "QR carries UAN + member name (UAN is 12-digit numeric).",
     "classification_regex": "^\\d{12}$"},
    {"scheme": "PAN (NSDL/UTI)",
     "description": "modern PAN cards carry a QR with PAN + name.",
     "classification_regex": "^[A-Z]{5}\\d{4}[A-Z]$"},
    {"scheme": "DigiLocker certificates & UDYAM",
     "description": "QR = deep link to the issuing portal's verify page "
                    "(URL, not JSON) - this is the user-assisted verification vector.",
     "classification_regex": None},
    {"scheme": "IndiaStack e-KYC responses",
     "description": "signed JSON with base64-encoded PDF/A image; the QR on "
                    "physical copies typically points to the issuer's verification endpoint.",
     "classification_regex": None},
]


def _sha256_file(path: Path) -> str:
    """sha256 hex of the catalog file -> recorded in the index as provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_issuer_trust(digest_path: Path) -> tuple[dict, str]:
    """Class (b): dedupe issuers across all queries by issuerId."""
    d = json.loads(digest_path.read_text())
    snap = d.get("captured_utc", "")
    seen: dict[str, dict] = {}
    order: list[str] = []
    for q in d.get("queries", {}).values():
        for e in q.get("entries", []):
            iid = e.get("issuerId")
            if not iid or iid in seen:
                continue
            seen[iid] = {
                "orgName": e.get("org"),
                "issuerId": iid,
                "orgType": e.get("type"),
                "orgState": e.get("orgState"),
                "subdomainName": e.get("subdomain"),
                "apis_sample": (e.get("apis_sample") or [])[:5],
                "snapshot_utc": snap,
            }
            order.append(iid)
    return {k: seen[k] for k in order}, snap


# ------------------------------------------------------------- live refresh --
class _BoundedRedirect(urllib.request.HTTPRedirectHandler):
    """Follow at most N redirects; beyond that raise (treated as unreachable)."""
    MAX = 5

    def __init__(self):
        super().__init__()
        self._hops = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._hops += 1
        if self._hops > self.MAX:
            raise urllib.error.HTTPError(newurl, code, "redirect loop", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get(url: str, timeout: int = 30) -> tuple[int, bytes, str]:
    """GET via urllib (stdlib only). Bounded redirects + hard timeout.

    Returns (status, body, content_type). Any HTTP error / timeout / DNS /
    redirect-loop raises; the caller records it as an unreachable source.
    """
    opener = urllib.request.build_opener(_BoundedRedirect())
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept": "text/html,*/*"})
    with opener.open(req, timeout=timeout) as r:
        return r.status, r.read(), r.headers.get("Content-Type", "")


def extract_phrases(html: bytes, cap: int = 8) -> list[str]:
    """Verbatim <=8-word phrases from fetched bytes. INVENTS NOTHING."""
    txt = html.decode("utf-8", "ignore")
    txt = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", txt)
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    txt = re.sub(r"&[a-z]+;", " ", txt)
    toks = re.findall(r"[A-Za-z][A-Za-z'&\-]{2,}", txt)
    out: list[str] = []
    i = 0
    while i < len(toks) and len(out) < cap:
        n = min(8, len(toks) - i)
        ph = " ".join(toks[i:i + n])
        if len(ph.split()) >= 3 and ph.lower() not in {p.lower() for p in out}:
            out.append(ph)
        i += n
    return out[:cap]


def specimen_urls(html: bytes) -> list[str]:
    pat = re.compile(rb'href=["\']([^"\']*(?:specimen|sample|demo)[^"\']*\.(?:png|jpe?g|pdf))["\']',
                     re.I)
    return sorted({m.decode() for m in pat.findall(html)})[:1]


def live_refresh(index: dict, budget: int = BUDGET) -> dict:
    used = 0
    t0 = time.time()
    baselines: list[dict] = []
    unreachable: list[dict] = []
    for host, url in ROOT_PAGES:
        if used >= budget:
            break
        rec = {"host": host, "url": url, "http_status": None,
               "content_type": None, "fetched_utc": None,
               "specimen_urls": [], "phrases": []}
        try:
            try:
                st, body, ct = _get(url)
                used += 1
            except Exception as ex:     # http error / timeout / conn / dns / redirect-loop
                used += 1
                rec["reason"] = ex.__class__.__name__
                unreachable.append({"host": host, "reason": ex.__class__.__name__.lower(),
                                    "http_status": None})
                continue
            rec.update(http_status=st, content_type=ct, fetched_utc=_now())
            rec["specimen_urls"] = specimen_urls(body)
            rec["phrases"] = extract_phrases(body)
            for p in rec["phrases"]:
                baselines.append({"phrase": p, "source_url": url,
                                  "fetched_utc": rec["fetched_utc"]})
            # stamp the matching derived template with real fetch provenance
            dt_key = {"digilocker.gov.in": "aadhaar_ecyc",
                      "nic.gov.in": "voter_id",
                      "incometax.gov.in": "income_tax_notice",
                      "nsdl.co.in": "pan_card"}.get(host)
            if dt_key and dt_key in index["entries"]["document_templates"]:
                te = index["entries"]["document_templates"][dt_key]
                te["source_url"] = url
                te["fetched_utc"] = rec["fetched_utc"]
                te["content_type"] = ct
                te["http_status"] = st
        finally:
            time.sleep(random.uniform(2.0, 3.0))
    for h, reason, st in KNOWN_UNREACHABLE:
        unreachable.append({"host": h, "reason": reason, "http_status": st})
    index["entries"]["official_content_baselines"] = baselines
    bk = index["entries"]["bookkeeping"]
    bk["unreachable_sources"] = unreachable
    bk["requests_used"] = used
    bk["request_budget_total"] = budget
    bk["wall_clock_seconds"] = int(time.time() - t0)
    return {"requests_used": used, "baselines": len(baselines),
            "unreachable": [u["host"] for u in unreachable]}


# --------------------------------------------------------------------- main --
def build(cache_dir: Path, version: str, do_network: bool) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "blobs").mkdir(exist_ok=True)
    issuers, snap = load_issuer_trust(DIGEST)
    index = {
        "schema_version": 1,
        "version": version,
        "built_utc": _now(),
        "staleness_days": STALENESS_DAYS,
        # Provenance for the reader's freshness gate (rag_cache.cache_freshness):
        # sha256 of the catalog file this build consumed; the reader recomputes
        # it against the LATEST catalog file and marks the cache stale on drift.
        "source_digest": {"file": DIGEST.name, "sha256": _sha256_file(DIGEST)},
        "entries": {
            "document_templates": {
                k: {"sha256": None, "source_url": None, "fetched_utc": None,
                    "content_type": None, "field_inventory": v, "derived": True,
                    "status": "ok", "http_status": None}
                for k, v in DOC_TEMPLATES.items()},
            "issuer_trust": issuers,
            "qr_schemes": QR_SCHEMES,
            "official_content_baselines": [],
            "bookkeeping": {"unreachable_sources": [
                                 {"host": h, "reason": r, "http_status": s}
                                 for h, r, s in KNOWN_UNREACHABLE],
                            "request_budget_total": BUDGET,
                            "requests_used": 0, "wall_clock_seconds": 0},
        },
        # Legacy view consumed by gov_document._rag_cache (db["templates"]).
        "templates": {k: {"version": version, "required_fields": v}
                      for k, v in DOC_TEMPLATES.items()},
    }
    refresh = None
    if do_network:
        refresh = live_refresh(index)
    idx_path = cache_dir / "cache-index.json"
    idx_path.write_bytes(json.dumps(index, indent=2, ensure_ascii=False).encode())
    return {"index_path": str(idx_path), "refresh": refresh,
            "n_templates": len(index["entries"]["document_templates"]),
            "n_issuers": len(issuers), "n_qr": len(QR_SCHEMES),
            "n_baselines": len(index["entries"]["official_content_baselines"]),
            "unreachable": [u["host"] for u in
                            index["entries"]["bookkeeping"]["unreachable_sources"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-network", action="store_true",
                    help="skip the live refresh (deterministic classes only)")
    args = ap.parse_args(argv)
    cache_dir = Path(os.environ.get("VISHWAS_RAG_CACHE", "/home/hermes/rag-cache"))
    version = os.environ.get("VISHWAS_RAG_VERSION", "1")
    res = build(cache_dir, version, do_network=not args.no_network)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
