"""7-language placeholder-leak / render-contract matrix (roadmap Phase 3 closeout).

Draft-fallback derivation, documented per contract: ``i18n.py`` has NO explicit
draft registry — the module docstring says draft lines carry a ``[?]`` marker,
so the draft-flagged (key, lang) set is DERIVED here by scanning the STORED
``_DEFAULTS`` strings for ``[?]``. That scan yields the EMPTY SET: the only
``[?]`` in the module sits in the docstring itself, and the docstring states
"No key falls back silently: every language renders its own string." The
contract encoded below is therefore: every one of the 7 supported languages
renders its OWN stored string for every key (no silent en-fallback), while the
en-fallback branch of ``t()`` is proven still available via a synthetic
missing-language case (hermetic ``monkeypatch``, auto-restored).

Hermetic by construction: imports ``verisafe`` only, no network, no filesystem
writes, no global-state mutation that outlives a test.
"""
from __future__ import annotations

import re

import pytest

import verisafe.i18n as i18n_mod
from verisafe.events import Verdict
from verisafe.i18n import _DEFAULTS, _SUPPORTED, t
from verisafe.report import ReportBuilder

# Named printf-style placeholders used by the string tables.
_PLACEHOLDER_RE = re.compile(r"%\((\w+)\)s")

# Keys whose absence/emptiness would break the product promise (spec:
# ONE verdict sentence + ONE confidence sentence, in the user's language).
_VERDICT_CRITICAL_KEYS = (
    "greeting",
    "analyzing",
    "verdict_trust",
    "verdict_caution",
    "verdict_do_not_use",
    "verdict_unable",
    "confidence_line",
)

# Recognizable, leak-detecting sample values for interpolation.
_SAMPLE_FMT = {
    "conf": "high",
    "name": "report.pdf",
    "cap": "gov-document",
    "verdict": "caution",
}


def _draft_flagged_pairs() -> set[tuple[str, str]]:
    """Derive draft-flagged (key, lang) pairs from stored ``[?]`` markers.

    This is the documented derivation: i18n.py represents draft state via
    ``[?]`` markers on stored strings (per its module docstring). Post-merge
    (commit faf07d0) no stored string carries the marker — drafts were merged
    with markers stripped — so this returns the empty set and the contract
    tests below demand a REAL rendered string per (key, lang).
    """
    flagged: set[tuple[str, str]] = set()
    for key, table in _DEFAULTS.items():
        for lang, text in table.items():
            if "[?]" in text:
                flagged.add((key, lang))
    return flagged


def test_draft_flag_derivation_empty_and_every_lang_has_own_strings():
    """The '[?]'-derived draft set is empty AND every key stores all 7
    languages — i.e. the module's "no silent fallback" claim is structurally
    true today, and the per-language contract tests can demand own strings."""
    assert _draft_flagged_pairs() == set(), (
        "stored '[?]' draft markers appeared; update the contract tests to "
        "allow by-design en-fallback for exactly these pairs"
    )
    for key, table in _DEFAULTS.items():
        missing = [lang for lang in _SUPPORTED if not table.get(lang, "").strip()]
        assert not missing, f"{key}: languages without own string: {missing}"


@pytest.mark.parametrize("lang", list(_SUPPORTED))
def test_every_key_renders_own_string_without_leak(lang):
    """Full matrix: every _DEFAULTS key × lang renders NON-EMPTY via the real
    t() path, returns the language's OWN stored string (never a silent
    en-fallback — none is draft-flagged today, see module docstring), leaks no
    '[?]' review marker, no U+FFFD mojibake char, and no stray '%' outside the
    named placeholders its own template declares."""
    for key, table in _DEFAULTS.items():
        stored = table[lang]
        rendered = t(key, lang)
        assert rendered.strip(), f"{lang}/{key}: rendered empty"
        assert rendered == stored, (
            f"{lang}/{key}: silent fallback/mutation — got {rendered!r}, "
            f"stored {stored!r}"
        )
        assert "[?]" not in rendered, f"{lang}/{key}: draft marker leaked"
        assert "\ufffd" not in rendered, f"{lang}/{key}: U+FFFD mojibake char"
        residue = _PLACEHOLDER_RE.sub("", rendered)
        assert "%" not in residue, (
            f"{lang}/{key}: unrendered '%' outside named placeholders: {rendered!r}"
        )


@pytest.mark.parametrize("lang", list(_SUPPORTED))
def test_placeholder_keys_interpolate_sample_values_no_leak(lang):
    """Every key whose stored template declares %(...)s placeholders: with the
    documented sample arguments supplied through the real t() path, every
    sample value appears in the output and NO '%(' token survives."""
    checked = 0
    for key, table in _DEFAULTS.items():
        names = _PLACEHOLDER_RE.findall(table[lang])
        if not names:
            continue
        checked += 1
        rendered = t(key, lang, **_SAMPLE_FMT)
        for name in names:
            assert _SAMPLE_FMT[name] in rendered, (
                f"{lang}/{key}: %{name}s did not interpolate -> {rendered!r}"
            )
        assert "%(" not in rendered, f"{lang}/{key}: placeholder leaked: {rendered!r}"
        assert "\ufffd" not in rendered, f"{lang}/{key}: mojibake after interpolation"
    # Guard against the matrix silently shrinking to zero placeholder keys.
    assert checked >= 3, f"expected >=3 placeholder keys, found {checked}"


@pytest.mark.parametrize("lang", list(_SUPPORTED))
def test_verdict_critical_keys_render_nonempty_all_langs(lang):
    """The 7 verdict-critical keys must render non-empty in ALL 7 languages;
    confidence_line is rendered with its %(conf)s sample (the way report.py
    actually calls it)."""
    for key in _VERDICT_CRITICAL_KEYS:
        fmt = {"conf": "high"} if key == "confidence_line" else {}
        out = t(key, lang, **fmt)
        assert isinstance(out, str) and out.strip(), (
            f"{lang}/{key}: verdict-critical key rendered empty ({out!r})"
        )
        assert "[?]" not in out and "%(" not in out, (
            f"{lang}/{key}: leak in verdict-critical render: {out!r}"
        )


def test_en_fallback_branch_still_works_by_design(monkeypatch):
    """t()'s en-fallback is the designed safety net for FUTURE keys added
    without a full language set. Prove it on a synthetic en-only key (module
    _DEFAULTS monkeypatched to a copy — auto-restored, nothing leaks)."""
    patched = {k: dict(v) for k, v in _DEFAULTS.items()}
    patched["_synthetic_en_only"] = {"en": "English only fallback text."}
    monkeypatch.setattr(i18n_mod, "_DEFAULTS", patched)
    for lang in _SUPPORTED:
        out = t("_synthetic_en_only", lang)
        assert out == "English only fallback text.", (
            f"{lang}: expected exact en fallback, got {out!r}"
        )
    # Unknown key degrades to the key itself, never crashes (pinned contract).
    assert t("__no_such_key__", "ta") == "__no_such_key__"


@pytest.mark.parametrize("lang", list(_SUPPORTED))
def test_report_builder_all_four_verdicts_render_clean(lang):
    """The real render path (ReportBuilder.build, as wired in report.py):
    4 verdict outcomes × lang produce a non-empty report with the confidence
    band stated and zero placeholder/marker/mojibake leakage."""
    builder = ReportBuilder()
    for verdict in (Verdict.TRUST, Verdict.CAUTION,
                    Verdict.DO_NOT_USE, Verdict.UNABLE_TO_VERIFY):
        report = builder.build(
            target="gov_document", verdict=verdict, confidence=0.9,
            reasons=[], checks=[], lang=lang)
        text = str(report)
        assert text.strip(), f"{lang}/{verdict}: empty report"
        assert "%(" not in text, f"{lang}/{verdict}: placeholder leak: {text!r}"
        assert "[?]" not in text, f"{lang}/{verdict}: draft marker leak"
        assert "\ufffd" not in text, f"{lang}/{verdict}: mojibake"
        low = text.lower()
        assert any(w in low for w in ("high", "moderate", "low", "%")), (
            f"{lang}/{verdict}: no confidence band statement in {text!r}"
        )
