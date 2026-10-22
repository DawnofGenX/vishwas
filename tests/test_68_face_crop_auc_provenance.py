"""The two face-crop AUCs are for DIFFERENT crops — pin which is which.

2026-10-22. A background run of `tools/eval_effort_face_crop.py` printed
`face13 AUC=0.7711`, while the shipped configuration's documented number is
**0.7789**. The medians were IDENTICAL (real 0.4619 / fake 0.6881), so the
discrepancy looked like an arithmetic error.

IT IS NOT. The two harnesses measure DIFFERENT crops:

  tools/score_video_signals_crop.py  -> 0.7789   the PRODUCTION crop
        (calls deepfake_video._effort_face_crop: LARGEST-AREA face
         selection, minimum face size as a FRACTION of the frame area)

  tools/eval_effort_face_crop.py    -> 0.7711   its OWN hand-rolled crop
        (orders faces by YuNet's raw 5th column, which reads ~120 and is
         not a 0..1 score; 16-pixel minimum face size)

Both numbers are CORRECT. The "full frame" row agrees at 0.7189 in both,
because that path is identical.

The trap was that the stale harness had NO warning that it measures a
different crop than production, and its output (0.7711) looks like a
sibling of the shipped number (0.7789). Anyone re-running it would
conclude the shipped crop is worse, or that the numbers disagree.

These tests pin:
  1. the production crop's documented AUC is 0.7789, not 0.7711;
  2. the production crop selects by AREA, not by YuNet's raw score column;
  3. the production crop's minimum face size is a FRACTION of the frame;
  4. the stale harness says so, in its docstring AND in its output.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. the documented number
# ---------------------------------------------------------------------------
def test_the_production_crop_auc_is_07789_not_07711():
    """The shipped configuration's number, and it must be the one the
    production code documents.

    2026-10-22: the code said 0.7711, which was the number for a REJECTED
    crop (tools/eval_effort_face_crop.py measures its own hand-rolled crop).
    The production crop measures 0.7789 -- verified by running
    tools/score_video_signals_crop.py, which calls the production helper."""
    src = (REPO / "src/vishwas/capabilities/deepfake_video.py").read_text()
    m = re.search(r"face crop\s+AUC (\d\.\d+)", src)
    assert m, "the production crop's AUC is no longer documented"
    assert m.group(1) == "0.7789", (
        f"the production crop documents AUC {m.group(1)}; expected 0.7789 "
        "(0.7711 is the REJECTED crop's number)")


def test_the_production_crop_docstring_quotes_the_measured_numbers():
    src = (REPO / "src/vishwas/capabilities/deepfake_video.py").read_text()
    assert "0.7189" in src, "the whole-frame AUC is no longer documented"
    assert "0.4619" in src and "0.6881" in src, (
        "the crop's medians are no longer documented")


# ---------------------------------------------------------------------------
# 2 & 3. the production crop's actual behaviour
# ---------------------------------------------------------------------------
def test_the_production_crop_selects_the_largest_face_by_area():
    """YuNet's 5th column is a RAW score (~120 here), not a 0..1 score, so
    ordering by it is meaningless. The production helper orders by AREA."""
    src = (REPO / "src/vishwas/capabilities/deepfake_video.py").read_text()
    assert "def _area(f):" in src, (
        "the production crop no longer ranks faces by area")
    assert "max(faces, key=_area)" in src, (
        "the production crop no longer picks the largest-area face")
    assert "YuNet's 5th column" in src, (
        "the raw-score-column trap is no longer documented in the code")


def test_the_production_crop_minimum_face_size_is_a_fraction_of_the_frame():
    """A 16-pixel floor rejected exactly the small distant faces that matter
    on the operator's wide street shots (their face is 3-4% of the width)."""
    src = (REPO / "src/vishwas/capabilities/deepfake_video.py").read_text()
    body = re.search(r"def _effort_face_crop.*?\n    except Exception:\n        return img",
                     src, re.S)
    assert body, "could not isolate the _effort_face_crop body"
    b = body.group(0)
    assert "0.0005 * (w * h)" in b, (
        "the production crop's fractional minimum face size is gone")
    # The 16-pixel FLOOR itself must be gone. The docstring may mention
    # "16" while explaining the trap, so check the CODE, not the prose.
    code_lines = [l for l in b.split("\n")
                  if not l.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert not re.search(r"\bbw\s*<\s*16\b|\bbh\s*<\s*16\b|<\s*16\b", code), (
        "the 16-pixel minimum face floor is back in the production crop")
    assert "0.0005" in code, (
        "the fractional minimum face size is not in the CODE, only in prose")


# ---------------------------------------------------------------------------
# 4. the stale harness warns
# ---------------------------------------------------------------------------
def test_the_stale_harness_warns_that_it_measures_a_different_crop():
    p = REPO / "tools/eval_effort_face_crop.py"
    if not p.exists():
        pytest.skip("harness not present")
    text = p.read_text()
    assert "REJECTED" in text, (
        "the stale harness does not say it measures a REJECTED crop")
    assert "0.7711" in text and "0.7789" in text, (
        "the stale harness does not state both numbers, so its output can be "
        "mistaken for the shipped configuration's")
    assert "score_video_signals_crop.py" in text, (
        "the stale harness does not point at the authoritative harness")


def test_the_authoritative_harness_states_the_shipped_numbers():
    p = REPO / "tools/score_video_signals_crop.py"
    if not p.exists():
        pytest.skip("harness not present")
    text = p.read_text()
    assert "0.7789" in text and "0.7189" in text, (
        "the authoritative harness does not state the expected shipped numbers")
    assert "authoritative" in text.lower(), (
        "the authoritative harness does not say it is authoritative")
