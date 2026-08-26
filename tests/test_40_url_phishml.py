"""URL-phishml vendored xgboost model — hermetic-ish tests.

Covers the local ML URL-phishing evidence signal (malindard/phishing-checker-flask,
MIT; see src/vishwas/_urlphish_vendor/PROVENANCE.md). The model/scaler/xgboost are
ONLY loadable under the serving PYTHONPATH (xgboost/tldextract/pandas live in the
docling-python tree), so these tests skip when those deps are absent — they do NOT
need the network, VT, or any weight env var.
"""
from __future__ import annotations

import pytest

from vishwas.capabilities.url_mal_ml import UrlPhishMl, _extract_features, _predict


def _importable(name: str) -> bool:
    import importlib
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

_HAVE = _importable("xgboost") and _importable("tldextract") and _importable("pandas")

pytestmark = pytest.mark.skipif(
    not _HAVE,
    reason="url-phishml deps (xgboost/tldextract/pandas) not importable in this env",
)


def test_loader_available_with_serving_deps():
    assert UrlPhishMl().available is True


def test_benign_versus_phish_separation():
    """The model must rank a phish URL above a benign one (offline, 16 lexical)."""
    loader = UrlPhishMl()
    benign = _predict("https://www.wikipedia.org", budget_s=0.0, loader=loader)[0]
    phish = _predict(
        "http://paypa1-secure-login.verify-account.top/webscr?cmd=login",
        budget_s=0.0, loader=loader)[0]
    assert benign is not None and phish is not None
    assert benign < 0.5, f"benign should score low, got {benign}"
    assert phish > 0.7, f"phish should score high, got {phish}"
    assert phish > benign


def test_extract_features_16_lexical_gated_network():
    """Offline-first: 16 lexical features computed, 7 network ones are gaps."""
    feas, gaps = _extract_features("https://www.google.com/search?q=hi", budget_s=0.0)
    assert feas is not None
    # 23 total (16 lexical present + 7 network added as neutral defaults)
    assert len(feas) == 23
    assert set(gaps) == {
        "nb_hyperlinks", "ratio_intHyperlinks", "empty_title", "domain_in_title",
        "domain_age", "google_index", "page_rank",
    }
    # lexical features are real values, not gaps
    assert feas["length_url"] > 0
    assert feas["nb_dots"] >= 0


def test_garbage_input_is_conservative_not_phish():
    """A non-URL string yields a low feature row — NOT a confident phish call.
    The extractor is permissive (degrades to structural defaults, no raise).
    Note: only real http(s) URLs reach url_phishing via the router; this pins
    that structural garbage stays well below the model's phish-certain zone
    (real phish score >=0.87, garbage ~0.57)."""
    feas, gaps = _extract_features("not a url", budget_s=0.0)
    assert feas is not None
    prob = _predict("not a url", budget_s=0.0, loader=UrlPhishMl())[0]
    assert prob is not None
    assert prob < 0.7, f"garbage input must not read as certain-phish, got {prob}"