"""THE "(av=0.5 AND havic) PAIR RULE" IS REJECTED — and must STAY rejected.

2026-10-22. After the silence fix (PR #13) the av=0.5 band means "no speech
energy", and HAVIC is saturated on genuine traffic (>= 0.95 on 42 of 63
genuine clips, median 0.980, versus 0.995 on the AI videos).

That combination makes ONE rule look attractive:

    "flag when av_risk_addition == 0.5 AND havic >= 0.9"

It is a **cross-modal-only** rule — it needs no video-side signal — and it
is exactly the shape of the old (wrong) belief that "anti-correlated IS the
deepfake signature". So it WILL be proposed by someone reading the
recorded numbers.

MEASURED on the operator's real WhatsApp traffic:

    the 2 genuine av=0.5 clips BOTH have high HAVIC
      c1dfce01  corr=-0.299  havic=0.983
      b10342bc  corr=-0.443  havic=0.980
    => the pair fires on 2 of 63 genuine clips (3% FALSE POSITIVES)
    and it catches 0 of the 3 operator AI videos, because their av reading
    is 0.0 (unknown — silent audio), so the pair never fires for them.

=> IT COSTS 3% GENUINE FALSE POSITIVES AND BUYS NOTHING.

These tests pin the rejection and the measurement behind it, so the rule
cannot be reintroduced by someone who has only read the summary numbers.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
FUSION = REPO / "src" / "vishwas" / "fusion.py"
TOOL = REPO / "tools" / "measure_crossmodal_pair_rule.py"


def _fusion_text() -> str:
    if not FUSION.exists():
        pytest.skip("fusion.py not present")
    return FUSION.read_text()


def test_the_pair_rule_rejection_is_recorded_where_the_numbers_are():
    """The rejection must live next to the numbers that invite it."""
    text = _fusion_text()
    assert "PAIR RULE" in text, (
        "the rejection of the (av=0.5 AND havic) pair rule is not recorded "
        "in the weight map, where the numbers that invite it are")
    assert "3% GENUINE FALSE POSITIVES" in text or \
           "FALSE POSITIVES" in text, (
        "the rejection does not state the false-positive cost")


def test_the_rejection_records_both_halves_of_the_measurement():
    text = _fusion_text()
    # the genuine false positives, with their numbers
    assert "-0.299" in text and "-0.443" in text, (
        "the rejection does not record the two genuine clips' correlations")
    assert "0.983" in text and "0.980" in text, (
        "the rejection does not record the two genuine clips' HAVIC readings")
    # and that it catches NONE of the AI videos
    assert "0 of 3" in text or "catches 0" in text, (
        "the rejection does not record that the pair rule catches none of "
        "the AI videos — which is the half that makes it pointless")


def test_no_cross_modal_only_rule_can_carry_the_verdict():
    """The structural invariant behind the rejection.

    A cross-modal-only rule has no video-side signal to corroborate, and
    the two signals it would use are both saturated on genuine media. The
    gate in `decide()` must therefore never let an elevated cross-modal
    reading carry a verdict on its own.
    """
    text = _fusion_text()
    i = text.find("if target == \"deepfake_video\":")
    assert i != -1, "the deepfake_video branch is gone"
    j = text.find("if isinstance(_av, (int, float))", i)
    assert j != -1, "the cross-modal gate is gone from the video branch"
    window = text[i:j + 4000]
    # the gate must require a VIDEO-side signal to agree
    assert "_eff_cal" in window and "_fh_cal" in window, (
        "the cross-modal gate no longer requires a video-side signal to "
        "agree — an elevated cross-modal reading could carry the verdict")
    assert "0.40" in window, (
        "the gate's video-side agreement threshold is gone")


def test_the_measurement_tool_exists():
    """The rejection should be re-checkable, not taken on trust."""
    if not TOOL.exists():
        pytest.skip("tool not present")
    text = TOOL.read_text()
    assert "av_risk_addition == 0.5" in text or "AV_T = 0.5" in text, (
        "the measurement tool does not implement the rejected rule")
    assert "REJECTED" in text or "Rejected" in text, (
        "the measurement tool does not state the verdict it produces")


def test_the_measurement_tool_stamps_the_tree_it_ran_against():
    """2026-10-22. The PR #18 lesson: a measurement with no provenance is
    indistinguishable from a stale run. This tool justifies a REJECTION, so
    a reader must be able to tell which tree produced the numbers.

    MUTATION-TRAP FOUND HERE: the first version of this test only checked
    that the tool DEFINES a `_tree_stamp` helper. A mutation that removed
    the CALL (leaving the definition) left the test green — the tool would
    define the stamp and never print it. So the assertion is on the CALL
    SITE, not on the definition.
    """
    if not TOOL.exists():
        pytest.skip("tool not present")
    text = TOOL.read_text()
    assert "_tree_stamp" in text and "rev-parse" in text, (
        "the measurement tool does not stamp the commit it ran against")
    assert "vishwas tree:" in text, (
        "the measurement tool's stamp does not use the 'vishwas tree:' "
        "prefix that every other stamp in the repo uses")
    # the CALL SITE: the stamp must actually be PRINTED, not merely defined
    assert re.search(r"print\([^)]*_tree_stamp\(\)", text), (
        "the measurement tool DEFINES a tree-stamp helper but never prints "
        "it — the output carries no provenance")


def test_the_measurement_tool_prints_the_stamp_before_its_numbers():
    """The stamp must be visible in the tool's output, before the numbers
    it labels. A stamp printed at the very end still labels nothing.

    NOTE the anchoring: the strings are matched inside `print(...)` calls,
    NOT as bare text. The tool's DOCSTRING mentions "THE PAIR RULE", and a
    bare `text.find()` matched that line instead of the print call — which
    made this test fail for the wrong reason (a stale lookup, exactly the
    trap the runner's own test fell into).
    """
    if not TOOL.exists():
        pytest.skip("tool not present")
    text = TOOL.read_text()
    m = re.search(r"print\([^)]*_tree_stamp\(\)", text)
    assert m, "the tool never prints the stamp"

    def _first_print_with(needle: str) -> int:
        for pm in re.finditer(r"print\(", text):
            # take the call's argument text up to the closing paren
            end = text.find(")", pm.start())
            frag = text[pm.start():end if end != -1 else len(text)]
            if needle in frag:
                return pm.start()
        return -1

    first_number = min(
        (i for i in (_first_print_with("genuine traffic rows:"),
                     _first_print_with("THE PAIR RULE:")) if i != -1),
        default=-1)
    if first_number != -1:
        assert m.start() < first_number, (
            "the tool prints its numbers BEFORE the stamp that labels them")


def test_the_measurement_tool_does_not_hardcode_its_denominator():
    """2026-10-22. The PR #17 lesson: never hardcode the number a harness
    measures. The tool printed "catches {n} of 3 AI videos" — a literal 3.

    If an anchor clip is added to or removed from the seeded dir, the
    printed fraction must FOLLOW the directory, or the summary states a
    number the run did not produce.
    """
    if not TOOL.exists():
        pytest.skip("tool not present")
    text = TOOL.read_text()
    assert not re.search(r"of 3 AI videos", text), (
        "the measurement tool hardcodes its AI-video denominator — it must "
        "derive it from the anchor directory it actually scanned")
    assert "len(ai_clips)" in text or "len(ai_dir)" in text, (
        "the measurement tool does not derive the AI-video count from the "
        "clips it scanned")


def test_the_measurement_tool_does_not_hardcode_its_false_positive_rate():
    """The tool printed "IT COSTS 3% GENUINE FALSE POSITIVES" — a literal
    percentage of a denominator it had just computed. The rate must be
    derived, or it states a number the run did not produce."""
    if not TOOL.exists():
        pytest.skip("tool not present")
    text = TOOL.read_text()
    assert not re.search(r"COSTS 3% GENUINE", text), (
        "the measurement tool hardcodes its false-positive percentage — it "
        "must derive it from the clips it measured")


def test_the_rejection_is_not_contradicted_by_a_live_rule():
    """No code path may implement the pair rule."""
    for p in list((REPO / "src").rglob("*.py")):
        t = p.read_text()
        for m in re.finditer(
                r"av_risk_addition.{0,40}==.{0,10}0\.5.{0,80}havic|"
                r"havic.{0,40}>=.{0,10}0\.9.{0,80}av_risk", t, re.I | re.S):
            pytest.fail(f"{p.relative_to(REPO)} appears to implement the "
                        f"rejected pair rule: {m.group(0)[:120]!r}")
