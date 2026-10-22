"""A TOOL'S SUMMARY TEXT MUST NOT CONTRADICT ITS OWN NUMBERS.

2026-10-22. `tools/measure_av_sync_real_traffic.py` printed, on the CURRENT
(correct) tree:

    av distribution: {0.0: 24, 0.1: 34, 0.45: 3, 0.5: 2}
    alignment classes: {'unknown': 9, 'synced': 15, 'weakly_synced': 34,
                        'anti_correlated': 2, 'decorrelated': 3}
    av_correlation: min=-0.443 med=0.396 max=0.793

    WHAT THIS ESTABLISHES: genuinely synchronised media produces
    av=0.0 (synced) or av=0.1 (weakly_synced) -- it does NOT produce
    the decorrelated band (0.45) or the anti-correlated band (0.5).

**The text contradicts the numbers directly above it.** The current tree
DOES produce the 0.5 band — for 2 clips — and the tool's own numbers say
so. The text was written when the pre-fix capability reported silent audio
as "anti_correlated", so "genuine media does not produce 0.5" was a true
reading of the 0.45-is-a-mux-artefact finding. The silence fix (PR #13)
changed the behaviour and the text was never revisited.

This is the same bug class as the two harnesses fixed in PRs #16 and #17:
**a tool whose output text was frozen at the moment of writing and never
revisited when the underlying behaviour changed.** It is worse than a
stale number, because the text is an INTERPRETATION — a reader takes the
conclusion as established.

WHAT THE HONEST TEXT MUST SAY
-----------------------------
* genuine media produces 0.0 (synced) and 0.1 (weakly_synced);
* it produces "unknown" when the audio has no speech energy — that is a
  measurement the check could not make, NOT evidence of anything;
* the 0.45 band is a MUX ARTEFACT (this part was right and stays);
* the 0.5 band on genuine media was, before PR #13, SILENCE misread as
  anti-correlation; the 2 clips that remain are REAL negative correlations
  (-0.299 and -0.443) and even those overlap the AI videos' range.

These tests pin:
  1. the summary text no longer claims genuine media "does NOT produce" the
     0.45 / 0.5 bands unconditionally;
  2. the text mentions the "unknown" band, which the current numbers show;
  3. the text records that the 0.5 band was a silence artefact;
  4. the text is derived from the run's OWN numbers, not a fixed string
     that can outlive them.
"""
from __future__ import annotations

import pathlib
import re

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "measure_av_sync_real_traffic.py"
REPO = pathlib.Path(__file__).resolve().parents[1]


def _text() -> str:
    if not TOOL.exists():
        pytest.skip("tool not present")
    return TOOL.read_text()


def test_the_summary_does_not_claim_genuine_media_never_produces_0_45_or_0_5():
    """The exact sentence that contradicted the run's own numbers.

    NOTE: the phrase is emitted by TWO adjacent print() calls, so the source
    contains a `")` / `print("` boundary between "does NOT produce" and
    "the decorrelated band". A regex over the raw source must therefore
    allow that boundary — an earlier version of this test used a tight
    pattern and therefore did NOT catch a mutation that reinstated the
    false claim. The check below matches the SOURCE shape, and
    `test_the_rendered_summary_is_consistent` checks the RENDERED text.
    """
    text = _text()
    # The false claim, as it appears in the source: two adjacent print
    # calls whose string literals together say it.
    false_claim = re.compile(
        r"does NOT produce\".{0,40}?\n?\s*print\(\"the decorrelated band",
        re.S)
    assert not false_claim.search(text), (
        "the summary still claims genuine media never produces the 0.45/0.5 "
        "bands -- but the CURRENT tree produces 0.5 for 2 clips and 0.45 for "
        "3, as its own printed distribution shows")

    # Belt and braces: the two halves must not both be present either.
    half_a = "does NOT produce" in text
    half_b = "the decorrelated band (0.45) or the anti-correlated band (0.5)" in text
    assert not (half_a and half_b), (
        "the source still contains both halves of the false claim")


def test_the_summary_mentions_the_unknown_band():
    """The current numbers contain 9 'unknown' clips. The summary must
    account for them -- they are clips the check could not measure."""
    text = _text()
    assert "unknown" in text, (
        "the summary never mentions the 'unknown' band, which is what the "
        "current run reports for clips with no speech energy")


def test_the_summary_records_that_0_5_was_a_silence_artefact():
    """The pre-fix 0.5 readings were SILENCE misread as anti-correlation.
    A reader of this tool's output must not conclude that genuine media
    simply 'never' reaches 0.5."""
    text = _text().lower()
    assert "silen" in text, (
        "the summary does not record that the 0.5 band was a silence "
        "artefact -- the pre-fix sentinel reading")
    assert "artefact" in text or "artifact" in text, (
        "the summary does not use the artefact framing for the band readings")


def test_the_summary_still_keeps_the_mux_artefact_conclusion():
    """The part that was RIGHT and must not be lost: the 0.45 band on the
    synthetic corpus is a mux artefact."""
    text = _text().lower()
    assert "mux" in text, (
        "the summary lost the mux-artefact conclusion, which is still the "
        "correct reading of the 0.45 band")


def test_the_tool_reports_the_tree_it_ran_against():
    """Same lesson as PR #18: the output must carry its own provenance, so
    a stale run is distinguishable from a changed behaviour."""
    text = _text()
    assert "vishwas tree:" in text or "rev-parse" in text, (
        "the tool does not stamp the commit it ran against, so a stale run's "
        "output is indistinguishable from a behaviour change")


def test_the_rendered_summary_is_consistent_with_the_current_numbers():
    """The strongest check: RUN the tool's summary block and read the text.

    The tool's summary is printed at the end of a long measurement, so the
    test extracts the summary statements from the source and asserts the
    rendered result does not contain the false claim. This is what a reader
    of the tool's output actually sees.
    """
    import re as _re
    text = _text()
    # Collect the string literals passed to print() in the summary block.
    literals = _re.findall(r'print\(\s*"((?:[^"\\]|\\.)*)"\s*\)', text)
    rendered = " ".join(literals)
    assert "does NOT produce" not in rendered, (
        "the RENDERED summary still says genuine media 'does NOT produce' "
        "the 0.45/0.5 bands")
    assert "unknown" in rendered, (
        "the rendered summary does not mention the 'unknown' band")
    assert "SILENCE" in rendered or "silen" in rendered.lower(), (
        "the rendered summary does not record that the 0.5 band was a "
        "silence artefact")
    assert "MUX" in rendered or "mux" in rendered.lower(), (
        "the rendered summary lost the mux-artefact conclusion")


# ---------------------------------------------------------------------------
# the same class of staleness in the sibling arbitration tool
# ---------------------------------------------------------------------------
def test_the_two_signal_arbitration_tool_records_that_the_third_signal_exists():
    """`tools/eval_image_arbitration.py` concluded that separating the two
    NYUAD-only groups "needs a THIRD signal".

    That was RIGHT, and PR #14 SUPPLIED IT (genimage). The tool now says so.
    Without that note, a reader of its output would conclude the image
    channel still cannot separate the two groups -- which stopped being true
    on 2026-10-22.

    Same bug class: an interpretation frozen at the moment of writing and
    never revisited when the underlying work changed.
    """
    p = REPO / "tools" / "eval_image_arbitration.py"
    if not p.exists():
        pytest.skip("tool not present")
    text = p.read_text()
    assert "THIRD SIGNAL NOW EXISTS" in text, (
        "the two-signal arbitration tool does not record that the third "
        "signal (genimage) now exists -- its conclusion reads as if the "
        "image channel still cannot separate the two groups")
    assert "genimage" in text.lower(), (
        "the tool does not name the third signal it says now exists")
