"""URL-phishing classical-ML evidence signal (vendored xgboost model).

VENDORED MODEL: malindard/phishing-checker-flask (MIT) — XGBoost classifier +
StandardScaler over 23 URL features. Inference: features -> scaler.transform ->
model.predict_proba -> index 1 = PHISHING probability (HIGH = risky, matches the
fusion positive-weight convention). See _urlphish_vendor/PROVENANCE.md.

POSTURE: OFFLINE-FIRST. The 16 lexical features are computed deterministically
with zero network. The 7 network-scrape features (nb_hyperlinks,
ratio_intHyperlinks, empty_title, domain_in_title, domain_age, google_index,
page_rank) default to the authors' neutral unfetched-URL values (1, 0.5, 0, 0,
plus whois/google calls) UNLESS wall-clock budget remains within the stage —
keeping the WhatsApp path fast + deterministic. This is the "after Virustotal
didn't confirm" local evidence signal: it runs even when VT is unavailable or
0-detections.

TRUST BAR (2026-08-26): this model was 0-star with no published eval, so fusion
weight is gated on an AUC >= 0.75 mini-corpus test; it must not be raised without
re-proving on real data. Evidence: /tmp/phish_ml_status.md.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ..events import Artifact, JobContext
from .base import CheckResult

_VENDOR = Path(__file__).resolve().parent.parent / "_urlphish_vendor"

# 16 deterministic/offline features (subset of the 23 the model expects).
_LEXICAL_FEATURES = (
    "length_url", "length_hostname", "ip", "nb_dots", "nb_qm", "nb_eq",
    "nb_slash", "nb_www", "ratio_digits_url", "ratio_digits_host",
    "tld_in_subdomain", "prefix_suffix", "shortest_word_host",
    "longest_words_raw", "longest_word_path", "phish_hints",
)
# 7 network-scrape features -> neutral defaults unless budget allows fetching.
_NETWORK_FEATURES = (
    "nb_hyperlinks", "ratio_intHyperlinks", "empty_title", "domain_in_title",
    "domain_age", "google_index", "page_rank",
)
# Neutral unfetched-URL defaults (matches authors' else-branch in api_url.py).
_NETWORK_NEUTRAL = {
    "nb_hyperlinks": 1.0, "ratio_intHyperlinks": 0.5, "empty_title": 0.0,
    "domain_in_title": 0.0, "domain_age": 0.0, "google_index": 0.0,
    "page_rank": 0.0,
}
_NETWORK_BUDGET_S = 8  # per-network-feature wall-clock allowance before we give up


class UrlPhishMl:
    """Lazy loader for the vendored xgboost pipeline. Silent-None on failure
    (same discipline as the other gates — a load error reads as unavailable)."""

    def __init__(self) -> None:
        self._pipe = None  # (model, scaler, selected_features)

    def _load(self):
        import joblib, pickle
        if self._pipe is not None:
            return self._pipe
        model = joblib.load(str(_VENDOR / "url_phishing_model.pkl"))
        scaler = joblib.load(str(_VENDOR / "scaler.pkl"))
        with open(_VENDOR / "selected_features.pkl", "rb") as f:
            features = pickle.load(f)
        self._pipe = (model, scaler, features)
        return self._pipe

    @property
    def available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False


def _extract_features(url: str, budget_s: float) -> tuple[dict[str, float] | None, list[str]]:
    """Compute the 23-feature row. Returns (features_or_None, gaps).

    Offline-first: 16 lexical always; 7 network only while budget remains.
    On any extractor error the whole row is None (unavailable), never a
    half-filled row that would silently bias the scaler.
    """
    import sys
    import warnings
    sys.path.insert(0, str(_VENDOR))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # vendored extractor has invalid-escape lint
        from url_feature_extractor import (  # type: ignore
        url_length, get_domain, having_ip_address, count_dots, count_exclamination,
        count_equal, count_slash, check_www, ratio_digits, tld_in_subdomain,
        prefix_suffix, shortest_word_length, longest_word_length, phish_hints,
        words_raw_extraction,
    )
    import tldextract
    from urllib.parse import urlparse

    gaps: list[str] = []
    try:
        hostname, domain, path = get_domain(url)
        ex = tldextract.extract(url)
        domain_name = f"{ex.domain}.{ex.suffix}"
        subdomain = ex.subdomain
        words_raw, words_raw_host, words_raw_path = words_raw_extraction(
            ex.domain, subdomain, path)
        feats: dict[str, float] = {
            "length_url": float(url_length(url)),
            "length_hostname": float(len(hostname or "")),
            "ip": float(having_ip_address(url)),
            "nb_dots": float(count_dots(hostname) if hostname else 0),
            "nb_qm": float(count_exclamination(url)),
            "nb_eq": float(count_equal(url)),
            "nb_slash": float(count_slash(url)),
            "nb_www": float(check_www(words_raw)),
            "ratio_digits_url": float(ratio_digits(url)),
            "ratio_digits_host": float(ratio_digits(hostname) if hostname else 0),
            "tld_in_subdomain": float(tld_in_subdomain(ex.suffix, subdomain)),
            "prefix_suffix": float(prefix_suffix(hostname) if hostname else 0),
            "shortest_word_host": float(shortest_word_length(words_raw_host or [])),
            "longest_words_raw": float(longest_word_length(words_raw or [])),
            "longest_word_path": float(longest_word_length(words_raw_path or [])),
            "phish_hints": float(phish_hints(url)),
        }
    except Exception:
        return None, ["feature_extraction_error"]

    # Network features — deterministic neutral default unless budget remains.
    t0 = time.monotonic()
    for nf in _NETWORK_FEATURES:
        if time.monotonic() - t0 > _NETWORK_BUDGET_S or budget_s <= 0:
            feats[nf] = _NETWORK_NEUTRAL[nf]
            gaps.append(nf)
            continue
        # (network-scrape computation intentionally minimal here: fetching HTML,
        #  whois domain_age, google_index, page_rank would each be a network call;
        #  offline-first default = neutral + gap. Full-scrape is a planned toggle.)
        feats[nf] = _NETWORK_NEUTRAL[nf]
        gaps.append(nf)

    return feats, gaps


def _predict(url: str, budget_s: float, loader: UrlPhishMl | None = None) -> tuple[float | None, list[str], int]:
    """(phishing_prob, gaps, n_lexical_used). None prob => unavailable."""
    feas, gaps = _extract_features(url, budget_s)
    if feas is None:
        return None, gaps, 0
    try:
        model, scaler, sel = (loader or UrlPhishMl())._load()
        import warnings
        with warnings.catch_warnings():
            # vendored model/scaler deserialized across minor sklearn/xgboost
            # versions -> benign InconsistentVersion / model-file warnings
            warnings.simplefilter("ignore")
            import numpy as np  # noqa: F401  (present in serving tree)
            import pandas as pd
            row = pd.DataFrame([{k: feas.get(k, 0.0) for k in sel}])[sel]
            X = scaler.transform(row)
            prob = float(model.predict_proba(X)[0][1])
        return prob, gaps, sum(1 for f in _LEXICAL_FEATURES if f in feas)
    except Exception:
        return None, gaps, 0


class UrlPhishMlCapability:
    """requires=() keeps this always-runnable; it gates internally on the model
    being loadable. Emits 'url_phishml' evidence -> fusion phishml.prob."""

    requires: tuple[str, ...] = ()

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        raw = art.path.read_text(errors="ignore").strip()
        url_str = (ctx.extra.get("urls_in_text") or [raw])[0]
        if not url_str:
            return [CheckResult("url_phishml", "mid", "unavailable",
                                {}, "no URL found in message")]
        budget = max(0.0, float(getattr(ctx, "remaining_s", lambda: 0)() or 0))
        prob, gaps, n_lex = _predict(url_str, budget)
        if prob is None:
            return [CheckResult("url_phishml", "mid", "unavailable",
                                {"n_lexical": n_lex, "gaps": gaps[:6]},
                                "url-phishml model/gate unavailable")]
        return [CheckResult(
            "url_phishml", "mid", "ok",
            {"phishing_prob": round(float(prob), 4),
             "n_lexical": n_lex,
             "n_network": len(gaps),
             "model_type": "xgboost",
             "gaps": gaps[:6]},
            ("ML url-phishing classifier: high score is phishing risk"
             if prob > 0.5 else "ML url-phishing classifier: low phishing score"),
        )]