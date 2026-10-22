"""NO SHIPPED DOCSTRING MAY STATE A VERDICT THE PIPELINE NO LONGER PRODUCES.

2026-10-22. The silence fix (PR #13) changed the operator's AI videos from
HIGH/do_not_use to MEDIUM/caution -- because their HIGH had been carried by
the cross-modal reading av=0.5, which was the correlation function's
UNDEFINED sentinel (-2.0), i.e. their SILENT audio tracks reported as
"anti-correlated".

**Three shipped locations still asserted the old verdict**, and none of them
made a test fail:

  * `src/vishwas/fusion.py` (the weight-map comment that justifies the
    shipped effort offset) — "3/3 AI DO_NOT_USE, so the whole band
    satisfies the intent";
  * `tests/test_65_effort_face_crop.py` (docstring) — "3/3 AI videos =
    do_not_use";
  * `tests/test_61_video_calibration.py` (docstring) — "the real AI videos
    DO read DO_NOT_USE end-to-end ... 3/3 AI videos HIGH".

They are prose, so the suite stayed green — which is exactly why they
survived. And the `fusion.py` one was **load-bearing**: it is the stated
justification for the shipped -0.40 offset.

THE RULE
--------
A docstring that states an END-TO-END VERDICT must carry a date and say
which behaviour it describes, and the "what the pipeline actually produces
now" must be recorded alongside it. A claim that was true when written is
not evidence that it is still true.

These tests pin:
  1. the "3/3 AI = HIGH / DO_NOT_USE" claim, wherever it appears, is
     accompanied by a note that it was superseded;
  2. the load-bearing justification in `fusion.py` says the offset is
     justified on the measurement that actually chose it;
  3. the current end-to-end truth (MEDIUM for the AI videos, LOW for the
     genuine clips) is recorded in the repo, so the corrected claim has a
     home.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

# The locations that carried the superseded claim.
CLAIM_SITES = (
    "src/vishwas/fusion.py",
    "tests/test_65_effort_face_crop.py",
    "tests/test_61_video_calibration.py",
    # the user-facing document -- a stale claim there is the most visible
    # of all, and it is what an operator or a reviewer actually reads
    "README.md",
)

# Phrases that assert the AI videos reach DO_NOT_USE.
CLAIM_PATTERNS = (
    r"3/3 AI DO_NOT_USE",
    r"3/3 AI videos = do_not_use",
    r"3/3 AI videos HIGH",
    r"AI videos? (?:must )?still read DO_NOT_USE",
    r"the real AI videos\s*\n?\s*DO read DO_NOT_USE",
)


def _read(rel: str) -> str:
    p = REPO / rel
    if not p.exists():
        pytest.skip(f"{rel} not present")
    return p.read_text()


@pytest.mark.parametrize("rel", CLAIM_SITES)
def test_the_superseded_ai_verdict_claim_is_flagged_as_superseded(rel: str):
    """Wherever the old claim survives, it must be marked as superseded."""
    text = _read(rel)
    hits = []
    for pat in CLAIM_PATTERNS:
        for m in re.finditer(pat, text):
            hits.append(m.group(0))
    if not hits:
        pytest.skip(f"{rel} no longer carries the claim")
    # The correction must be within a reasonable distance of the claim, so
    # a note elsewhere in the file does not "cover" a claim here.
    for hit in set(hits):
        for m in re.finditer(re.escape(hit), text):
            window = text[max(0, m.start() - 1400): m.end() + 1400]
            assert ("2026-10-22" in window) and (
                "SILENCE" in window.upper() or "SENTINEL" in window.upper()
            ), (
                f"{rel} still states {hit!r} with no nearby note that it was "
                "superseded by the silence fix (PR #13). A claim that was true "
                "when written is not evidence that it is still true.")


def test_the_fusion_offset_justification_names_the_measurement_that_chose_it():
    """The weight-map comment is the JUSTIFICATION for the shipped offset.
    It must not rest on a claim the pipeline no longer produces."""
    text = _read("src/vishwas/fusion.py")
    m = re.search(r'"effort\.prob":\s*\(1\.0,\s*-0\.40\)', text)
    assert m, "the shipped effort offset is no longer (1.0, -0.40)"
    window = text[max(0, m.start() - 3000): m.end() + 200]
    assert "INTENT (b) IS NO LONGER SATISFIED" in window or \
           "NO LONGER SATISFIED" in window, (
        "the shipped offset's justification does not record that its "
        "intent (b) -- the operator's AI videos reaching DO_NOT_USE -- is "
        "no longer satisfied")
    assert "STILL JUSTIFIED" in window or "still justified" in window, (
        "the justification must say the offset is still justified on the "
        "measurement that actually chose it (the corpus numbers and the "
        "clean-side guard), so a reader is not left thinking the offset "
        "lost its basis")


def test_the_current_end_to_end_truth_is_recorded_in_the_repo():
    """The corrected claim needs a home, so the correction is not lost with
    the note that motivated it."""
    p = REPO / "docs" / "research" / "VIDEO_CHANNEL_MEASUREMENT.md"
    if p.exists():
        text = p.read_text()
    else:
        # fall back to any in-repo doc that records the current truth
        found = False
        for cand in REPO.rglob("*.md"):
            try:
                t = cand.read_text()
            except Exception:
                continue
            if "MEDIUM" in t and ("caution" in t.lower()) and \
               ("silence" in t.lower() or "sentinel" in t.lower()):
                found = True
                break
        assert found, (
            "no in-repo document records that the operator's AI videos now "
            "read MEDIUM/caution because their former HIGH came from a "
            "silence artefact -- the corrected claim has no home")


def test_the_regression_suite_does_not_assert_the_superseded_verdict():
    """No test may ASSERT the AI videos reach DO_NOT_USE.

    A docstring is prose; an assertion is a gate. This one must be a gate,
    because the AI videos' HIGH was the thing that most needed re-checking.
    """
    for p in (REPO / "tests").glob("test_*.py"):
        text = p.read_text()
        for m in re.finditer(
                r'assert[^#\n]*(?:DO_NOT_USE|DNU)[^#\n]*(?:ai_crf45|ai_fps12|'
                r'ai_resize50|operator)', text, re.I):
            pytest.fail(
                f"{p.name} asserts the superseded AI-video verdict: "
                f"{m.group(0)[:120]!r}")
