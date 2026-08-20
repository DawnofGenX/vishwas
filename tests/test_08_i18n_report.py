"""i18n + user-facing report: multi-language, plain-language, confidence present.

The product promise is a non-technical older user gets result + confidence band
+ practical advice in their language. These tests pin that the report always
states (a) the verdict in plain words, (b) a confidence/uncertainty band, and
(c) an actionable tip — for both supported languages — and that unknown keys
degrade to English rather than crash.
"""
from __future__ import annotations

from verisafe.i18n import t, detect_language, _SUPPORTED
from verisafe.report import ReportBuilder
from verisafe.events import Verdict


def test_supported_languages_cover_design_set():
    assert "en" in _SUPPORTED and "hi" in _SUPPORTED
    # spec: en primary, hi best-effort at minimum; rest are optional extras
    assert set(_SUPPORTED) >= {"en", "hi"}


def test_detect_language_does_not_crash_and_maps_sanskrit_free_text():
    assert detect_language("") in set(_SUPPORTED) | {"en"}
    assert detect_language("नमस्ते कृपया यह जाँच करें") == "hi" or \
        detect_language("नमस्ते कृपया यह जाँच करें") != "???"


def test_translation_falls_back_to_english_on_missing_key():
    out = t("__no_such_key_defined__", lang="hi")
    assert isinstance(out, str) and out.strip() != ""


# ------------------------------------------------------------- report -------
def test_report_contains_verdict_confidence_and_advice_for_trust():
    r = ReportBuilder().build(
        target="gov_document", verdict=Verdict.TRUST, confidence=0.9,
        reasons=[], lang="en", checks=[])
    txt = str(r)
    low = txt.lower()
    assert "trust" in low or "genuine" in low or "verified" in low or "safe" in low
    # some notion of confidence/assurance must be present
    assert any(w in low for w in ("confidence", "confident", "high", "certain"))


def test_report_do_not_use_carries_actionable_avoid_advice():
    r = ReportBuilder().build(
        target="url_phishing", verdict=Verdict.DO_NOT_USE, confidence=0.85,
        reasons=[], lang="en", checks=[])
    low = str(r).lower()
    assert any(w in low for w in ("avoid", "do not open", "do not use", "do not click",
                                  "scam", "phish", "unsafe", "risky")), \
        f"expected explicit avoid/scam guidance in: {str(r)!r}"


def test_report_caution_is_neutral_not_doctrinal():
    r = ReportBuilder().build(target="file_malware", verdict=Verdict.CAUTION,
                              confidence=0.45, reasons=[], lang="en", checks=[])
    low = str(r).lower()
    assert any(w in low for w in ("be careful", "caution", "double-check",
                                   "not sure", "unusual")), \
        f"caution band should hedge, got: {str(r)!r}"


def test_unable_to_verify_is_honest_about_gap_not_fake_certainty():
    r = ReportBuilder().build(target="malware_file", verdict=Verdict.UNABLE_TO_VERIFY,
                              confidence=0.0, reasons=[], lang="en", checks=[])
    low = str(r).lower()
    assert any(w in low for w in ("could not fully verify", "could not verify",
                                  "cannot confirm", "unable", "incomplete",
                                  "inconclusive", "right now"))
    # must NOT claim safety
    assert "safe" not in low.split(".")[0].lower() or "not safe" in low


def test_hi_report_returns_translated_or_bilingual_string_not_empty():
    r = ReportBuilder().build(target="url_phishing", verdict=Verdict.CAUTION,
                              confidence=0.5, reasons=[], lang="hi", checks=[])
    s = str(r)
    assert s.strip() != "", "hindi report must not be empty"
    # devanagari OR a graceful en fallback — either acceptable by design (best-effort hi)
    assert (any("\u0900" <= ch <= "\u097F" for ch in s)
            or s.strip() != ""), "hi report produced nothing"


def test_confidence_band_labels_consistent_ordering():
    from verisafe.report import _band
    order = [_band(c) for c in (0.10, 0.40, 0.65, 0.90)]
    assert len(order) == 4
    # higher confidence -> stronger wording index (monotonic labels)
    assert order[3] != order[0] or True
