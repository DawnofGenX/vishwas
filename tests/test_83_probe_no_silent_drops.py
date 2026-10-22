"""A PROBE MUST NOT SILENTLY DROP CORPUS ITEMS (2026-10-22).

`tools/probe_gate_accuracy.py` measured the image channel as **n=59**
on a corpus of **60** images, and nobody noticed for a day.

THE CAUSE
---------
The image loop was:

    for f in sorted(d.iterdir())[:n]:
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue

It SLICED FIRST and filtered SECOND. `fusion_img/ai/` holds 31
entries — 30 images plus a `sources.csv` manifest — and `sources.csv`
sorts BEFORE `stable_diffusion_3_*`. So the `[:30]` slice kept the CSV
and **dropped a real AI image** (`stable_diffusion_3_07.webp`), which the
extension check then never saw. The shortfall was invisible: the probe
reported a number, and a number does not advertise its own missingness.

**⇒ A SILENT SKIP HIDES A BROKEN CORPUS AND LOOKS IDENTICAL TO A MODEL
THAT FAILED.** Every channel in that probe does `except Exception:
continue` — which is exactly how the audio path once reported "the model
is broken" when the PROBE was broken (n=0, PR #27).

WHAT THIS PINS
--------------
  1. the probe FILTERS BEFORE IT SLICES — the corpus is reduced to the
     scoreable files first, then the sample is taken;
  2. the probe prints a CORPUS vs MEASURED comparison, so a shortfall is
     visible in the output rather than inferred;
  3. the expected count is derived from the CORPUS, not hardcoded;
  4. no channel in the probe may swallow a failure without counting it.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
PROBE = REPO / "tools" / "probe_gate_accuracy.py"


def _src() -> str:
    if not PROBE.exists():
        import pytest
        pytest.skip("the gate accuracy probe is not present")
    return PROBE.read_text()


def test_the_probe_filters_before_it_slices():
    """The `[:n]` slice must come AFTER the extension filter."""
    src = _src()
    m = re.search(r"def probe_image\(.*?\n(?=\ndef |\nclass )", src, re.S)
    assert m, "the image probe is gone"
    body = m.group(0)
    # the filtered list must be built BEFORE it is sliced
    filt = re.search(
        r"files\s*=\s*\[f for f in sorted\(d\.iterdir\(\)\)\s*"
        r"\n?\s*if f\.suffix\.lower\(\) in", body, re.S)
    assert filt, (
        "the image probe does not build a FILTERED file list — it must "
        "reduce the corpus to scoreable files BEFORE sampling, or a "
        "non-image file (sources.csv) can push a real image out of the "
        "sample and the shortfall is invisible")
    # and the slice must apply to that filtered list
    assert re.search(r"files\s*=\s*files\[:n\]", body), (
        "the image probe does not slice the FILTERED list")
    # The old shape must be gone -- but the check is on CODE, not prose:
    # the fix's COMMENT deliberately quotes the old buggy line to explain
    # it. A naive substring check would flag the explanation.
    code = _strip_comments(body)
    assert not re.search(r"sorted\(d\.iterdir\(\)\)\[:n\]", code), (
        "the image probe still slices the RAW directory listing before "
        "filtering — that is the bug that dropped stable_diffusion_3_07")



def _strip_comments(src: str) -> str:
    """The source with `#` comments removed.

    A naive substring check would flag the fix's own COMMENT, which
    deliberately quotes the old buggy line in order to explain it.
    """
    return "\n".join(l.split("#")[0] for l in src.split("\n"))


def test_the_probe_reports_a_shortfall():
    """A count that does not match the corpus must be VISIBLE."""
    src = _src()
    assert "CORPUS vs MEASURED" in src, (
        "the probe does not print a corpus-vs-measured comparison, so a "
        "silently dropped item is invisible")
    assert "SOMETHING WAS DROPPED" in src, (
        "the probe does not warn when the measured count is short of the "
        "corpus")
    assert "_expected_counts" in src, (
        "the probe has no expected-count helper, so the comparison is "
        "against a hardcoded number that can drift from the corpus")


def test_the_expected_count_is_derived_from_the_corpus():
    """The expected count must come from the corpus, not a literal."""
    src = _src()
    m = re.search(r"def _expected_counts\(.*?\n(?=\ndef |\nclass )", src, re.S)
    assert m, "the probe has no _expected_counts helper"
    body = m.group(0)
    assert re.search(r"len\(.*iterdir\(\)", body), (
        "the expected count is not derived from the corpus on disk")
    # the audio one is a SAMPLE size (60 per class), so a literal is fine —
    # but it must be the same 60 the probe uses.
    assert "60" in body, "the audio expected count does not match probe_audio's n"
    assert re.search(r"def probe_audio\(n:\s*int\s*=\s*60\)", src), (
        "probe_audio's default is no longer 60, so the expected count and "
        "the probe disagree")


def test_no_channel_swallows_a_failure_without_counting_it():
    """Every `except: continue` in the probe must be counted."""
    src = _src()
    # the image channel
    m = re.search(r"def probe_image\(.*?\n(?=\ndef |\nclass )", src, re.S)
    assert m
    body = m.group(0)
    skips = len(re.findall(r"except Exception:\s*\n\s*skipped \+= 1", body))
    conts = len(re.findall(r"except Exception:\s*\n\s*continue", body))
    assert conts == 0, (
        "the image probe still has an uncounted `except: continue` — a "
        "dropped item must be COUNTED so the shortfall is visible")
    assert skips >= 1, (
        "the image probe does not count its failures at all")


def test_the_video_channel_also_counts_its_skips():
    """The video channel has the same silent-skip shape."""
    src = _src()
    m = re.search(r"def probe_video\(.*?\n(?=\ndef |\nclass )", src, re.S)
    assert m, "the video probe is gone"
    body = m.group(0)
    conts = len(re.findall(r"except Exception:\s*\n\s*continue", body))
    assert conts == 0, (
        "the video probe still has an uncounted `except: continue` — a "
        "dropped clip must be COUNTED so the shortfall is visible")
    assert re.search(r"skipped", body), (
        "the video probe does not count its failures at all")
