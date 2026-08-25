"""test_30_fusion_trust.py — clean-evidence TRUST reachability + total_w==0 guard.

Regression coverage for the 2026-08-25 fusion fixes:
  F-A: fully-clean scans on risk-weighted targets reach TRUST (was: CAUTION forever).
  F-B: partial coverage or elevated signals must NOT get the clean bonus.
  F-C: confirmed-bad evidence still lands DO_NOT_USE.
  F-D: total_w==0 targets (document_generic/unclassified) answer UNABLE, never
       TRUST-at-confidence-1.0 (latent overclaim found by adversarial wave 2A).

Hermetic: pure CheckResult/FusionEngine construction, no I/O.
"""
from __future__ import annotations

import pytest

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine


def _ck(name: str, status: str = "ok", sig: dict | None = None) -> CheckResult:
    return CheckResult(name=name, cost="mid", status=status,
                       signals=sig if sig else {},
                       notes="")


def _url_checks_clean() -> list[CheckResult]:
    return [
        _ck("vt_url_reputation", "ok", {"positives_ratio": 0.0, "vt_total_engines": 91}),
        _ck("phish_heuristics", "ok", {"score_norm": 0.01, "young_domain": False}),
        _ck("ssrf_guard", "ok", {"blocked": 0.0}),
        _ck("url_redirects", "ok", {"suspicious_hops": 0}),
        _ck("url_download_revalidated", "ok", {"ext_mismatch": False, "verified_kind": "text"}),
    ]


def test_url_all_clean_reaches_trust():
    d = FusionEngine().decide("url_phishing", _url_checks_clean())
    assert d.verdict is Verdict.TRUST
    assert d.raw <= 0.15


def test_url_bad_stays_do_not_use():
    checks = [
        _ck("vt_url_reputation", "ok", {"positives_ratio": 0.18, "vt_total_engines": 91}),
        _ck("phish_heuristics", "ok", {"score_norm": 1.0, "young_domain": True}),
        _ck("ssrf_guard", "ok", {"blocked": 0.0}),
    ]
    d = FusionEngine().decide("url_phishing", checks)
    assert d.verdict is Verdict.DO_NOT_USE


def test_partial_coverage_no_bonus():
    """Only one signal ran -> no clean bonus -> NOT trustable."""
    checks = [_ck("phish_heuristics", "ok", {"score_norm": 0.02, "young_domain": False})]
    d = FusionEngine().decide("url_phishing", checks)
    assert d.verdict is not Verdict.TRUST


def test_malicious_file_clean_trust_and_eicar_dnu():
    clean = [
        _ck("vt_reputation", "degraded", {"positives_ratio": 0.0, "vt_total_engines": 0}),
        _ck("clamscan", "ok", {"detected": 0.0}),
        _ck("yara_x", "ok", {"hits_norm": 0.0}),
        _ck("file_entropy", "ok", {"anomaly": 0.0}),
        _ck("pe_statics", "degraded", {"packed": 0.0, "prob_malicious": 0.0}),
        # realistic clean-run companions (mirrors malware_file.analyze output):
        _ck("dynamic_sandbox", "skipped", {}),          # known_gap: progressive skip
        _ck("apk_statics", "unavailable", {}),           # known_gap: not an APK
        _ck("mobsf_apk", "unavailable", {}),             # known_gap
        _ck("quark_engine", "unavailable", {}),          # known_gap
        _ck("ext_mismatch_flag", "ok", {}),              # const_true, no mismatch => gap
    ]
    d = FusionEngine().decide("malicious_file", clean)
    assert d.verdict is Verdict.TRUST
    eicar = [
        _ck("vt_reputation", "ok", {"positives_ratio": 0.87, "vt_total_engines": 75}),
        _ck("clamscan", "ok", {"detected": 1.0}),
        _ck("yara_x", "ok", {"hits_norm": 0.33}),
    ]
    d2 = FusionEngine().decide("malicious_file", eicar)
    assert d2.verdict is Verdict.DO_NOT_USE


def test_unmapped_target_never_trusts_at_full_confidence():
    """total_w==0 guard: usable-but-unmapped check must yield UNABLE, conf 0."""
    checks = [_ck("some_generic_check", "ok", {"prob_deepfake": 0.9})]
    for target in ("document_generic", "unclassified"):
        d = FusionEngine().decide(target, checks)
        assert d.verdict is Verdict.UNABLE_TO_VERIFY
        assert d.confidence == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
