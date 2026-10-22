"""THE AUDIO LANE MUST NOT RELOAD THE CHECKPOINT ONCE PER CROP.

2026-10-22. The aasist3 lane shells out to
`scripts/audio_score_aasist3.py` — once PER CROP. The helper loads the
Spectra-AASIST3 checkpoint, scores ONE file, and exits. So a 3-crop
verdict pays the load THREE times.

MEASURED on this host:
    aasist3 checkpoint LOAD (cold)   63.1 s
    one aasist3 inference             17.6 s
    one crop, end to end              80.5 s
    the 3-crop audio lane            ~241 s
    the heavy-stage budget            120 s

**=> The audio lane EXCEEDS the budget before it can produce a verdict.**
The E2E result is `UNVERIFIED` with `pending_heavy` — on a cold cache.
(The same host, warm, took 105.95 s: a verdict, but with 14 s of
headroom. The product is non-deterministic.)

THE FIX: score ALL the crops in ONE subprocess, so the checkpoint is
loaded ONCE. 1 x 63.1 + 3 x 17.6 = ~116 s in the worst case, and ~81 s
with a warm cache.

WHAT THESE TESTS PIN
--------------------
  1. the helper accepts MULTIPLE files and scores them all with ONE
     load;
  2. the capability calls the helper ONCE for all crops, not once per
     crop;
  3. the helper's per-file output is still parseable (one posterior per
     file, in order);
  4. a single-file call still works (backwards compatibility).
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "audio_score_aasist3.py"
CAPABILITY = REPO / "src" / "vishwas" / "capabilities" / "deepfake_audio.py"


def _helper_src() -> str:
    if not HELPER.exists():
        import pytest
        pytest.skip("the aasist3 helper is not present")
    return HELPER.read_text()


def _capability_src() -> str:
    if not CAPABILITY.exists():
        import pytest
        pytest.skip("the audio capability is not present")
    return CAPABILITY.read_text()


def test_the_helper_accepts_more_than_one_file():
    """The helper's `audio` argument must accept a LIST."""
    src = _helper_src()
    assert re.search(r'nargs\s*=\s*["\']\+["\']', src), (
        "the aasist3 helper takes exactly ONE file, so scoring 3 crops "
        "loads the checkpoint 3 times (~241 s) against a 120 s budget")
    # and the call site must pass a list
    assert re.search(r"nargs", src), (
        "the aasist3 helper does not declare a multi-file argument")


def test_the_helper_loads_the_model_once_for_all_files():
    """The load must be OUTSIDE the per-file loop."""
    src = _helper_src()
    # find the loop over the input files
    m = re.search(r"for\s+\w+\s+in\s+args\.audio", src)
    assert m, (
        "the aasist3 helper does not loop over the input files, so it "
        "cannot score more than one")
    after = src[m.end():]
    # the load (`spec.build()`) must NOT appear after the loop starts
    assert "spec.build()" not in after, (
        "the aasist3 helper builds the model INSIDE the per-file loop, so "
        "the checkpoint is reloaded for every crop")
    # and it must appear BEFORE the loop
    before = src[:m.start()]
    assert "spec.build()" in before, (
        "the aasist3 helper does not build the model before the loop")


def test_the_helper_prints_one_posterior_per_file():
    """The output contract: one posterior per file, in order."""
    src = _helper_src()
    assert re.search(r"print\([^)]*p[^)]*file=sys\.stdout", src) or \
           re.search(r"print\(\s*f?[\"']\{", src), (
        "the aasist3 helper does not print a per-file posterior")


def test_the_capability_calls_the_helper_once_for_all_crops():
    """The capability must batch the crops into ONE subprocess call."""
    src = _capability_src()
    m = re.search(r"def _multi_crop_subprocess_aasist\b(.*?)\n    def ", src, re.S)
    assert m, "the subprocess aasist3 path is gone from the capability"
    body = m.group(1)
    # the BATCHED call must be the one the capability makes
    spawn = re.search(r"_subprocess_aasist3_scores?\(([^)]*)\)", body)
    assert spawn, (
        "the capability no longer calls the aasist3 helper")
    # the argument must be the LIST of crops, not a single crop
    assert "crops" in spawn.group(1), (
        f"the capability still passes a single crop to the helper "
        f"({spawn.group(1)!r}), so the checkpoint is reloaded per crop")
    # and there must NOT be a comprehension that calls it per crop
    assert not re.search(r"\[\s*_subprocess_aasist3_score\w*\([^)]*\)"
                         r"\s+for\s+[^]]+\]", body), (
        "the capability scores each crop in a SEPARATE subprocess, so the "
        "checkpoint is loaded 3 times for a 3-crop verdict (~241 s vs a "
        "120 s budget)")


def test_a_single_file_call_still_works():
    """Backwards compatibility: one file is still valid."""
    src = _helper_src()
    assert "sys.exit(main())" in src or "raise SystemExit(main())" in src, (
        "the helper lost its entry point")
    # the decode helper takes a single path -> so the per-file work is
    # a function of ONE path
    assert re.search(r"def _decode_f32\(path:\s*str\)", src), (
        "the helper's decoder no longer takes a single path")


def test_one_bad_file_does_not_abort_the_batch():
    """A single undecodable crop must not throw away the OTHER crops'
    scores — the whole point of the batched call is that one load serves
    every crop.

    The helper marks a file it could not score with NAN, and the caller
    DROPS NaN (it is "not scored", NOT 0.0 — a 0.0 would read as
    "bonafide", which is the fail-open direction).
    """
    h = _helper_src()
    assert "NAN" in h, (
        "the helper does not mark an unscoreable file, so one bad crop "
        "either aborts the batch or silently becomes a 0.0 posterior")
    # it must CONTINUE, not return
    assert not re.search(r"if wav\.size < 1000:[^}]*return 2", h, re.S), (
        "the helper still aborts the whole batch on a short file")
    c = _capability_src()
    assert "isnan" in c, (
        "the caller does not drop NaN posteriors — a NaN would poison the "
        "median and a 0.0 would read as bonafide")
    assert "math.isnan" in c, (
        "the caller does not explicitly test for NaN")
