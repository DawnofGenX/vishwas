"""A MEASUREMENT HARNESS MUST NOT HARDCODE THE THRESHOLDS IT IS MEASURING.

2026-10-22. `tools/eval_effort_face_variants.py` was found hardcoding the
EFFORT offset at **-0.30** while the SHIPPED calibration is **-0.40**
(`src/vishwas/fusion.py`: `"effort.prob": (1.0, -0.40)`).

A background run of that harness printed:

    with the -0.30 offset, how many AI videos would the effort term
    alone push into the DO_NOT_USE band (calibrated >= 0.40)?
      full       AI cal: ['0.065', '0.064', '0.016']  -> 0/3 flagged
      face_only  AI cal: ['0.347', '0.243', '0.343']  -> 0/3 flagged

Every number in that block is computed at the WRONG offset, and the block
is presented as if it described the shipped configuration. When PR #12 moved
the offset -0.30 -> -0.40, this harness silently kept reporting the old
one. That is the dangerous class of bug: not a wrong value, but a
measurement tool that has quietly stopped describing the thing it claims to
describe.

THE RULE
--------
A harness that reports a THRESHOLD must READ it from the production
source, or it must loudly say which threshold it is using. A harness whose
number can drift away from production without any signal is a landmine.

These tests pin:
  1. the harness reads the offset from the production calibration table, not
     from a literal;
  2. the harness does not hardcode a stale offset;
  3. the shipped offset and the harness's offset are the SAME value;
  4. the harness's face crop matches the PRODUCTION crop's two rules
     (largest AREA, fractional size floor), not the rejected ones;
  5. the harness carries a warning that it is not the shipped path.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
HARNESS = REPO / "tools/eval_effort_face_variants.py"


def _harness_text() -> str:
    if not HARNESS.exists():
        pytest.skip("harness not present")
    return HARNESS.read_text()


def _shipped_offset() -> float:
    src = (REPO / "src/vishwas/fusion.py").read_text()
    m = re.search(r'"effort\.prob":\s*\(\s*[\d.]+\s*,\s*(-?[\d.]+)\s*\)', src)
    assert m, "the effort.prob calibration entry is gone"
    return float(m.group(1))


# ---------------------------------------------------------------------------
# 1-3. the offset must come from production
# ---------------------------------------------------------------------------
def test_the_harness_does_not_hardcode_the_effort_offset():
    """A literal offset in the harness is how this went stale.

    The harness may still SHOW an offset, but it must not COMPUTE with a
    literal one."""
    text = _harness_text()
    # a subtraction against a bare float literal in the calibration math
    offenders = re.findall(r"-\s*0\.\d+\s*\)\s*$|r\[[\"']\w+[\"']\]\s*-\s*0\.\d+",
                            text, re.M)
    assert not offenders, (
        f"the harness computes with a hardcoded offset literal: {offenders} "
        "- it must read the offset from the production calibration table")


def test_the_harness_reads_the_offset_from_the_production_calibration():
    text = _harness_text()
    assert "_CALIBRATE" in text or "_calibrate" in text, (
        "the harness does not reference the production calibration table "
        "(_CALIBRATE / _calibrate) at all")
    assert "from vishwas.fusion import" in text, (
        "the harness does not import from vishwas.fusion")


def test_the_harness_actually_IMPORTS_the_calibration_table():
    """A TEXT check is not enough.

    The first version of this fix imported `_CALIBRATION` — a name that does
    not exist in vishwas.fusion — and the text-based test PASSED anyway. The
    harness would have died on the first run with an ImportError.

    So this test EXECUTES the harness's import and its offset assignment,
    extracted with AST so that only those statements run (the harness has no
    main() guard, so importing it wholesale would run the whole corpus).
    """
    import ast
    import subprocess
    import sys

    tree = ast.parse(_harness_text())
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "vishwas.fusion":
            wanted.append(node)
        elif isinstance(node, ast.Assign):
            names = set()
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    # _EFFORT_A, _EFFORT_B = _CALIBRATE["effort.prob"]
                    for el in t.elts:
                        if isinstance(el, ast.Name):
                            names.add(el.id)
            if names & {"_EFFORT_A", "_EFFORT_B", "EFFORT_OFFSET", "EFFORT_SLOPE"}:
                wanted.append(node)
    assert wanted, (
        "could not find the harness's calibration import/assignment — the "
        "offset may have been hardcoded again")

    snippet = ast.Module(body=wanted, type_ignores=[])
    code = ("import sys; sys.path.insert(0, 'src')\n"
            + ast.unparse(snippet)
            + "\nprint(EFFORT_OFFSET)\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (
        f"the harness's calibration import does not resolve: {r.stderr[-400:]}")
    assert float(r.stdout.strip().splitlines()[-1]) == _shipped_offset(), (
        f"the harness's offset {r.stdout.strip().splitlines()[-1]} is not the "
        f"shipped {_shipped_offset()}")


def test_the_harness_offset_is_the_shipped_offset():
    """If the harness shows an offset, it must be the shipped one."""
    text = _harness_text()
    m = re.search(r"EFFORT_OFFSET\s*=\s*(-?[\d.]+)", text)
    if m:
        assert float(m.group(1)) == _shipped_offset(), (
            f"the harness's EFFORT_OFFSET is {m.group(1)} but the shipped "
            f"calibration is {_shipped_offset()}")


# ---------------------------------------------------------------------------
# 4. the crop must match production
# ---------------------------------------------------------------------------
def test_the_harness_does_not_order_faces_by_yunets_raw_score_column():
    """YuNet's 5th column is a RAW score (reads ~120 here), not a 0..1
    score, so ordering by it is meaningless. Production orders by AREA."""
    text = _harness_text()
    assert "key=lambda f: f[4]" not in text, (
        "the harness still orders faces by YuNet's raw 5th column")
    assert "key=_area" in text or "key=lambda f: f[2] * f[3]" in text, (
        "the harness does not rank faces by AREA the way production does")


def test_the_harness_does_not_use_the_rejected_16_pixel_face_floor():
    """Production's minimum face size is a FRACTION of the frame area
    (0.05%). A 16-pixel floor rejects exactly the small distant faces that
    matter on the operator's wide street shots."""
    text = _harness_text()
    body = "\n".join(l for l in text.split("\n") if not l.strip().startswith("#"))
    assert not re.search(r">=\s*16\s+and", body), (
        "the harness still uses the rejected 16-pixel minimum face floor")
    assert "0.0005" in body, (
        "the harness does not use the production fractional minimum face size")


# ---------------------------------------------------------------------------
# 5. the warning
# ---------------------------------------------------------------------------
def test_the_harness_carries_a_warning_about_its_provenance():
    """The harness must state where its numbers come from.

    It used to (a) hardcode a stale offset and (b) use a crop that is not
    the shipped one, with no warning about either. The crop is now FIXED, so
    the banner must record that it used to differ AND that the offset is
    READ rather than hardcoded — otherwise a future reader cannot tell
    which parts of the output are trustworthy."""
    text = _harness_text()
    assert "used to" in text.lower(), (
        "the harness does not record that it used to differ from production")
    assert "0.7711" in text and "0.7789" in text, (
        "the harness does not state both AUC numbers, so a reader cannot "
        "tell which is the shipped one")
    assert "READ FROM PRODUCTION" in text or "read from" in text.lower(), (
        "the harness does not say that the offset is read from production "
        "rather than hardcoded")
    assert "hardcoded at -0.30" in text, (
        "the harness does not record the stale offset it used to carry — "
        "that is the whole point of the fix")


def test_the_harness_crop_now_matches_the_production_crop():
    """The whole point of fixing the crop: the two harnesses must AGREE.

    'face_only' reading the same 0.7789 as the production harness is an
    independent confirmation of the shipped number."""
    text = _harness_text()
    # production: largest AREA, fractional floor, whole-frame guard
    assert "key=_area" in text, "the harness no longer ranks faces by AREA"
    assert "MIN_FACE_FRAC" in text, (
        "the harness no longer uses the production fractional minimum face size")
    assert "0.0005" in text, (
        "the harness's fractional minimum face size is not 0.05% of the frame")
    assert "essentially the whole frame" in text, (
        "the harness does not have the production whole-frame guard")
