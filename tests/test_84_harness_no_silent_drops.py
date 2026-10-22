"""NO MEASUREMENT HARNESS MAY SWALLOW AN ITEM IN SILENCE (2026-10-22).

THE PATTERN THAT KEPT RECURRING
-------------------------------
`except Exception: continue` around the object being measured. It showed up
four times in three days, each time as a different defect:

  * `tools/probe_gate_accuracy.py` sliced BEFORE filtering, so the image
    channel reported n=59 on a 60-image corpus (PR #34);
  * the same probe's AUDIO channel kept two bare `except: continue`
    branches (decode + write), so a corpus row that failed to decode
    vanished with no count at all — the docstring claimed "no channel may
    swallow a failure without counting it" while the tests only enforced
    it for the image and video channels;
  * `tools/build_crossmodal_corpus*.py` skipped rows that failed to
    decode while filling a per-class quota, so a bad parquet builds a
    SHORT corpus and every AUC measured on it afterwards is over a
    denominator nobody chose;
  * `tools/score_video_signals_crop.py` — the AUTHORITATIVE harness for
    the shipped face crop (AUC 0.7789) — dropped unreadable frames,
    infer-error frames and missing clips without a word, so a clip scored
    on 1 of 8 frames looked exactly like a clip scored on 8 of 8.

**⇒ A NUMBER DOES NOT ADVERTISE ITS OWN MISSINGNESS.** A count that is
short for a benign reason and a count that is short because the model is
broken look identical in the output. The rule below is the cheap
invariant: if a harness skips an item, it must leave a trace — a counter,
a record, or a print.
"""
from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]

# The harness families that produce numbers a human then trusts.
HARNESS = re.compile(r"(probe|score|measure|eval|grid|sweep|choose|"
                     r"build_.*corpus)")

# A trace is any of: `x += 1` / `x = x + 1` (a counter), or a call that
# records/diagnoses the failure. `print` counts: a printed warning IS the
# trace (see `scripts/audio_score_aasist3.py`, which prints the file and
# the reason and emits NAN).
_TRACE_ATTRS = {"append", "extend", "write", "add"}
_TRACE_NAMES = {"print"}


def _harnesses() -> list[pathlib.Path]:
    files = []
    for sub in ("tools", "scripts"):
        d = REPO / sub
        if not d.is_dir():
            continue
        files.extend(p for p in sorted(d.glob("*.py")) if HARNESS.search(p.name))
    return files


def _has_trace(handler: ast.ExceptHandler) -> bool:
    nodes = list(ast.walk(handler))
    if any(isinstance(n, ast.AugAssign) for n in nodes):
        return True
    for n in nodes:
        if (isinstance(n, ast.Assign) and isinstance(n.value, ast.BinOp)
                and isinstance(n.value.op, ast.Add)):
            return True
        if isinstance(n, ast.Call):
            fn = n.func
            if isinstance(fn, ast.Attribute) and fn.attr in _TRACE_ATTRS:
                return True
            if isinstance(fn, ast.Name) and fn.id in _TRACE_NAMES:
                return True
    return False


def test_the_harness_set_is_not_empty():
    """Guard the guard: an empty file list would make this suite pass."""
    files = _harnesses()
    assert len(files) >= 10, (
        f"only {len(files)} harness files found — the HARNESS pattern or the "
        "directory layout changed and this test is now vacuous")


def test_no_harness_swallows_an_item_without_a_trace():
    """Every `except: continue` in a harness must count, record, or print."""
    offenders = []
    for path in _harnesses():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as e:  # a broken harness is its own finding
            offenders.append(f"{path.name}: SYNTAX ERROR {e}")
            continue
        for handler in [n for n in ast.walk(tree)
                        if isinstance(n, ast.ExceptHandler)]:
            nodes = list(ast.walk(handler))
            if not any(isinstance(n, ast.Continue) for n in nodes):
                continue
            if not _has_trace(handler):
                offending = (handler.body[0].lineno if handler.body
                             else handler.lineno)
                offenders.append(f"{path.name}:{offending}")
    assert not offenders, (
        "these harnesses skip the object they are measuring and leave no "
        f"trace at all: {offenders}")


def test_a_skipped_item_is_visible_in_the_probe_output():
    """The probe must REPORT its drops, not just count them internally."""
    probe = REPO / "tools" / "probe_gate_accuracy.py"
    if not probe.exists():
        import pytest
        pytest.skip("the gate accuracy probe is not present")
    src = probe.read_text()
    assert "skipped" in src and "DROPPED" in src, (
        "the probe counts drops but never prints them; a counter nobody "
        "prints is a counter nobody reads")
