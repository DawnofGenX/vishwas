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
from verisafe.capabilities.base import CheckResult
from verisafe.events import Artifact, InputType, JobContext, MediaKind, Verdict
from verisafe.report import ReportBuilder


def _catalog(tmp_path: Path) -> Path:
    """Fake apisetu catalog-digest file (the builder's source of truth)."""
    d = tmp_path / "digests"
    d.mkdir(exist_ok=True)
    p = d / "apisetu_catalog_digest_2026-08-19.json"
    p.write_text(json.dumps({"captured_utc": "2026-08-19", "queries": {}}))
    return p


def _digest_env(tmp_path: Path, monkeypatch) -> None:
    """Point the freshness gate's catalog dir at the hermetic fixture dir."""
    monkeypatch.setenv("VERISAFE_RAG_DIGEST_DIR", str(tmp_path / "digests"))


def _idx(tmp_path: Path, *, version="1", templates=None, staleness_days=90,
         built_utc=None, source_digest: str | None = "match") -> Path:
    """Write a cache index + matching catalog file (hermetic fixtures).

    built_utc:      default = now (fresh); pass an ISO string to backdate.
    source_digest:  "match" records the catalog file's real sha256;
                    any other str records that literal value (mismatch);
                    None omits the field entirely (pre-gate cache shape).
    """
    cat = _catalog(tmp_path)
    if source_digest == "match":
        source_digest = rc.file_sha256(cat)
    p = tmp_path / "cache-index.json"
    data = {
        "schema_version": 1,
        "version": version,
        "built_utc": (built_utc if built_utc is not None else
                      _dt.datetime.now(_dt.timezone.utc)
                      .strftime("%Y-%m-%dT%H:%M:%SZ")),
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
    if source_digest is not None:
        data["source_digest"] = {"file": cat.name, "sha256": source_digest}
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


# -------------------------------------------------- cache freshness gate ----
def test_cache_fresh_with_matching_digest(tmp_path):
    """Fresh build + recorded digest matches latest catalog file -> fresh."""
    idx = rc.load(_idx(tmp_path))
    fresh, reason = rc.cache_freshness(idx, digest_dir=tmp_path / "digests")
    assert fresh is True
    assert reason == "ok"
    assert rc.cache_stale(idx, digest_dir=tmp_path / "digests") is False


def test_cache_backdated_beyond_ttl_is_stale(tmp_path):
    """(a) built_utc older than the 14-day TTL -> stale, with reason."""
    old = (_dt.datetime.now(_dt.timezone.utc)
           - _dt.timedelta(days=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
    idx = rc.load(_idx(tmp_path, built_utc=old))
    fresh, reason = rc.cache_freshness(idx, digest_dir=tmp_path / "digests")
    assert fresh is False
    assert reason == "ttl-expired"


def test_cache_ttl_boundary_exact_14d_is_fresh(tmp_path):
    edge = (_dt.datetime.now(_dt.timezone.utc)
            - _dt.timedelta(days=rc.CACHE_TTL_DAYS, seconds=-1))
    idx = rc.load(_idx(tmp_path, built_utc=edge.strftime("%Y-%m-%dT%H:%M:%SZ")))
    fresh, _ = rc.cache_freshness(idx, digest_dir=tmp_path / "digests")
    assert fresh is True


def test_cache_missing_timestamp_is_stale(tmp_path):
    """Honest default: no parseable built_utc -> stale, never assumed fresh."""
    p = _idx(tmp_path)
    d = json.loads(p.read_text())
    del d["built_utc"]                       # pre-gate cache shape
    p.write_text(json.dumps(d))
    fresh, reason = rc.cache_freshness(rc.load(p),
                                       digest_dir=tmp_path / "digests")
    assert fresh is False
    assert reason == "missing-build-timestamp"


def test_cache_unparseable_timestamp_is_stale(tmp_path):
    p = _idx(tmp_path, built_utc="not-a-timestamp")
    fresh, reason = rc.cache_freshness(rc.load(p),
                                       digest_dir=tmp_path / "digests")
    assert fresh is False
    assert reason == "missing-build-timestamp"


def test_cache_digest_mismatch_is_stale(tmp_path):
    """(d) recorded sha256 no longer matches the catalog file -> stale."""
    idx = rc.load(_idx(tmp_path, source_digest="0" * 64))
    fresh, reason = rc.cache_freshness(idx, digest_dir=tmp_path / "digests")
    assert fresh is False
    assert reason == "digest-mismatch"


def test_cache_missing_source_digest_is_stale(tmp_path):
    """Backward-compat honesty: a cache with no recorded provenance is stale."""
    idx = rc.load(_idx(tmp_path, source_digest=None))
    fresh, reason = rc.cache_freshness(idx, digest_dir=tmp_path / "digests")
    assert fresh is False
    assert reason == "missing-source-digest"


def test_cache_catalog_file_absent_is_stale(tmp_path):
    """No catalog file to verify against -> cannot claim freshness."""
    idx = rc.load(_idx(tmp_path))
    fresh, reason = rc.cache_freshness(idx,
                                       digest_dir=tmp_path / "no-such-dir")
    assert fresh is False
    assert reason == "catalog-unavailable"


def test_cache_digest_checked_against_latest_catalog(tmp_path):
    """A NEWER catalog file in the dir wins; old recorded sha -> mismatch."""
    idx = rc.load(_idx(tmp_path))            # records sha of the 08-19 file
    d = tmp_path / "digests"
    newer = d / "apisetu_catalog_digest_2026-09-25.json"
    newer.write_text(json.dumps({"captured_utc": "2026-09-25"}))
    assert rc.latest_catalog(d) == newer
    fresh, reason = rc.cache_freshness(idx, digest_dir=d)
    assert fresh is False
    assert reason == "digest-mismatch"


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


def test_capability_version_mismatch_skips(tmp_path, monkeypatch):
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = _idx(tmp_path, version="7")
    cap.rag_version = 1                       # active version != cache version
    _digest_env(tmp_path, monkeypatch)
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    res = cap._rag_cache("PAN card ABCDE1234F", ctx, "pan_card", None)
    assert res[0].status == "skipped"
    assert "doc type/version" in res[0].notes


def test_capability_matching_version_reports_layout(tmp_path, monkeypatch):
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = _idx(tmp_path, version="1")
    cap.rag_version = 1
    _digest_env(tmp_path, monkeypatch)
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    text = "PAN card number ABCDE1234F, name John Doe, date of birth 01/01/1990"
    res = cap._rag_cache(text, ctx, "pan_card", None)
    assert res[0].status == "ok"
    assert res[0].signals["required_fields_matched"] == 1.0
    assert res[0].signals["stale"] is False
    assert res[0].signals["source_of_truth"] is False


def test_capability_stale_entry_degrades_not_blocks(tmp_path, monkeypatch):
    """Freshness contract: stale entry -> 'degraded', never an error/block."""
    cap = gd.GovDocumentCapability()
    p = _idx(tmp_path, version="1")
    # force a live-provenance entry far outside the staleness window
    d = json.loads(p.read_text())
    d["templates"]["pan_card"]["fetched_utc"] = "2025-01-01T00:00:00Z"
    p.write_text(json.dumps(d))
    cap.rag_cache_path = p
    cap.rag_version = 1
    _digest_env(tmp_path, monkeypatch)
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    res = cap._rag_cache("PAN card ABCDE1234F name John date of birth",
                         ctx, "pan_card", None)
    assert res[0].status == "degraded"
    assert res[0].signals["stale"] is True
    assert "STALE" in res[0].notes


# ------------------------------------------- cache-level gate at consumer ---
def test_capability_backdated_cache_drops_contribution_silently(
        tmp_path, monkeypatch):
    """(a) TTL-expired cache -> silent skip (same surface as absent cache),
    never an error; carries the evidence_gap token for the report layer."""
    old = (_dt.datetime.now(_dt.timezone.utc)
           - _dt.timedelta(days=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = _idx(tmp_path, built_utc=old)
    cap.rag_version = 1
    _digest_env(tmp_path, monkeypatch)
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    res = cap._rag_cache("PAN card ABCDE1234F name John date of birth",
                         ctx, "pan_card", None)
    assert res[0].status == "skipped"            # silent degradation
    assert res[0].signals["cache_stale"] is True
    assert res[0].signals["stale_reason"] == "ttl-expired"
    assert res[0].signals["evidence_gap"] == "gov-template-cache-stale"
    assert "freshness gate" in res[0].notes


def test_capability_digest_mismatch_drops_contribution(tmp_path, monkeypatch):
    """(d) at the consumer: catalog drift -> signal dropped silently."""
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = _idx(tmp_path, source_digest="f" * 64)
    cap.rag_version = 1
    _digest_env(tmp_path, monkeypatch)
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    res = cap._rag_cache("PAN card ABCDE1234F name John date of birth",
                         ctx, "pan_card", None)
    assert res[0].status == "skipped"
    assert res[0].signals["stale_reason"] == "digest-mismatch"


def test_capability_fresh_cache_still_contributes(tmp_path, monkeypatch):
    """(b) end-to-end: fresh cache + matching digest -> contribution intact."""
    cap = gd.GovDocumentCapability()
    cap.rag_cache_path = _idx(tmp_path)
    cap.rag_version = 1
    _digest_env(tmp_path, monkeypatch)
    art, ctx = _mk_ctx(tmp_path, rag_cache_available=True)
    res = cap._rag_cache("PAN card number ABCDE1234F, name John Doe",
                         ctx, "pan_card", None)
    assert res[0].status in ("ok", "degraded")   # contributed, not dropped
    assert "cache_stale" not in res[0].signals


# ------------------------------------------------------- report surfacing ---
def test_report_evidence_missing_lists_stale_token():
    """Staleness-caused drops are named on the evidence_missing line."""
    checks = [CheckResult("rag_template_cache", "cheap", "skipped",
                          {"cache_stale": True,
                           "evidence_gap": "gov-template-cache-stale"})]
    rep = ReportBuilder().build(target="gov_document", verdict=Verdict.CAUTION,
                                confidence=0.4, reasons=[], checks=checks)
    assert "gov-template-cache-stale" in rep.text


def test_report_evidence_missing_absent_without_gaps():
    checks = [CheckResult("rag_template_cache", "cheap", "ok", {})]
    rep = ReportBuilder().build(target="gov_document", verdict=Verdict.CAUTION,
                                confidence=0.4, reasons=[], checks=checks)
    assert "gov-template-cache-stale" not in rep.text


# ------------------------------------------------------ builder provenance --
def test_builder_embeds_source_digest_provenance(tmp_path):
    """build() records the consumed catalog file's sha256 for the gate."""
    import importlib.util
    script = (Path(__file__).resolve().parents[1] / "scripts"
              / "build_rag_cache.py")
    spec = importlib.util.spec_from_file_location("build_rag_cache", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "cache"
    mod.build(out, "1", do_network=False)
    idx = json.loads((out / "cache-index.json").read_text())
    assert idx["source_digest"]["file"] == Path(mod.DIGEST).name
    assert idx["source_digest"]["sha256"] == rc.file_sha256(mod.DIGEST)
    # and the freshly-built index passes its own freshness gate
    fresh, reason = rc.cache_freshness(idx, digest_dir=mod.DIGEST.parent)
    assert fresh is True
    assert reason == "ok"
