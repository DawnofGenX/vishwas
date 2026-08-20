"""RAG template-cache reader tests (brief .delegation/brief_rag.md item 5).

HERMETIC: temp-dir fixtures, zero network. Exercises the reader contract
(load / freshness / staleness-degradation / empty-cache-feature-off /
partial-version-mismatch) plus the capability's silent feature-OFF path.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
from pathlib import Path

import pytest

from verisafe import rag_cache as rc
from verisafe.capabilities import gov_document as gd
from verisafe.events import Artifact, InputType, JobContext, MediaKind


def _idx(tmp_path: Path, *, version="1", templates=None, staleness_days=90) -> Path:
    p = tmp_path / "cache-index.json"
    data = {
        "schema_version": 1,
        "version": version,
        "built_utc": "2026-08-20T00:00:00Z",
        "staleness_days": staleness_days,
        "entries": {
            "document_templates": {
                "pan_card": {"sha256": None, "source_url": None,
                             "fetched_utc": None, "content_type": None,
                             "field_inventory": ["pan card", "name"],
                             "derived": True, "status": "ok", "http_status": None},
            },
            "issuer_trust": {"in.gov.x": {"orgName": "X", "issuerId": "in.gov.x",
                                          "orgType": "Central", "orgState": None,
                                          "subdomainName": "x", "apis_sample": [],
                                          "snapshot_utc": "2026-08-19"}},
            "qr_schemes": [{"scheme": "PAN (NSDL/UTI)", "description": "d",
                            "classification_regex": "^[A-Z]{5}\\d{4}[A-Z]$"}],
            "official_content_baselines": [],
            "bookkeeping": {"unreachable_sources": [], "request_budget_total": 6,
                            "requests_used": 0, "wall_clock_seconds": 0},
        },
        "templates": (templates if templates is not None else {
            "pan_card": {"version": version,
                         "required_fields": ["pan card", "name", "date of birth"]},
        }),
    }
    p.write_text(json.dumps(data))
    return p


# ------------------------------------------------------------------ load ----
def test_load_returns_dict(tmp_path):
    p = _idx(tmp_path)
    d = rc.load(p)
    assert d["schema_version"] == 1
    assert "entries" in d and "templates" in d


def test_load_absent_returns_empty(tmp_path):
    assert rc.load(tmp_path / "nope.json") == {}


def test_load_malformed_returns_empty(tmp_path):
    p = tmp_path / "cache-index.json"
    p.write_text("{not valid json")
    assert rc.load(p) == {}


def test_available_gate(tmp_path):
    assert rc.available(tmp_path / "missing.json") is False
    assert rc.available(_idx(tmp_path)) is True


# --------------------------------------------------------------- freshness --
def test_freshness_derived_entry_is_fresh():
    """Derived-only template (no fetched_utc) never goes stale."""
    assert rc.freshness({"fetched_utc": None}) is True


def test_freshness_recent_entry_is_fresh():
    now = _dt.datetime(2026, 8, 20, tzinfo=_dt.timezone.utc)
    e = {"fetched_utc": "2026-08-19T00:00:00Z"}
    assert rc.freshness(e, now=now, index={"staleness_days": 90}) is True


def test_freshness_stale_entry_degrades():
    now = _dt.datetime(2026, 12, 1, tzinfo=_dt.timezone.utc)   # >90d after Aug
    e = {"fetched_utc": "2026-08-01T00:00:00Z"}
    assert rc.freshness(e, now=now, index={"staleness_days": 90}) is False


# ------------------------------------------------------------ get_entries ---
def test_get_entries_mapping_class(tmp_path):
    d = rc.load(_idx(tmp_path))
    out = rc.get_entries("document_templates", d)
    assert len(out) == 1
    assert out[0]["_key"] == "pan_card"
    assert out[0]["field_inventory"] == ["pan card", "name"]


def test_get_entries_list_class_and_unknown(tmp_path):
    d = rc.load(_idx(tmp_path))
    assert len(rc.get_entries("qr_schemes", d)) == 1
    assert rc.get_entries("does_not_exist", d) == []


# ------------------------------------------------------- capability wiring --
def _mk_ctx(tmp_path: Path, rag_cache_available: bool):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")
    art = Artifact(path=p, original_filename="doc.pdf",
                   declared_type=InputType.FILE, verified_kind=MediaKind.PDF)
    ctx = JobContext(job_id="t", artifact=art, quarantine_root=tmp_path,
                     deadline_mono=time.monotonic() + 300,
                     rag_cache_available=rag_cache_available)
    return art, ctx


def test_capability_empty_cache_is_silent_feature_off(tmp_path):
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = tmp_path / "absent" / "cache-index.json"
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=False)
    res = cap._rag_cache("PAN card ABCDE1234F name John", ctx, "pan_card", None)
    assert res[0].status == "skipped"
    assert "no local template cache" in res[0].notes


def test_capability_version_mismatch_skips(tmp_path):
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = _idx(tmp_path, version="7")
    cap.rag_version = 1                       # active version != cache version
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    res = cap._rag_cache("PAN card ABCDE1234F", ctx, "pan_card", None)
    assert res[0].status == "skipped"
    assert "doc type/version" in res[0].notes


def test_capability_matching_version_reports_layout(tmp_path):
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = _idx(tmp_path, version="1")
    cap.rag_version = 1
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    text = "PAN card number ABCDE1234F, name John Doe, date of birth 01/01/1990"
    res = cap._rag_cache(text, ctx, "pan_card", None)
    assert res[0].status == "ok"
    assert res[0].signals["required_fields_matched"] == 1.0
    assert res[0].signals["stale"] is False
    assert res[0].signals["source_of_truth"] is False


def test_capability_stale_entry_degrades_not_blocks(tmp_path):
    """Freshness contract: stale entry -> 'degraded', never an error/block."""
    cap = gd.GovDocumentCapability()
    p = _idx(tmp_path, version="1")
    # force a live-provenance entry far outside the staleness window
    d = json.loads(p.read_text())
    d["templates"]["pan_card"]["fetched_utc"] = "2025-01-01T00:00:00Z"
    p.write_text(json.dumps(d))
    cap.rag_cache_path = p
    cap.rag_version = 1
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    res = cap._rag_cache("PAN card ABCDE1234F name John date of birth",
                         ctx, "pan_card", None)
    assert res[0].status == "degraded"
    assert res[0].signals["stale"] is True
    assert "STALE" in res[0].notes
