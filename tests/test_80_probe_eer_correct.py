"""THE GATE PROBES MUST REPORT A CORRECT EER — AND MUST NOT BE VACUOUS.

2026-10-22. Two defects in the measurement tooling, both found by
running it:

1. **THE EER WAS COMPUTED WITH THE FAR AND FRR SWAPPED**, which reported
   **50% EER for a perfectly separating detector**. The numbers were the
   giveaway, not the code: a detector that separates its classes
   perfectly (AUC 1.0000, 0 misclassified at 0.5) cannot have a 50% error
   rate. The fix is in `tools/probe_gate_accuracy.py`.

2. **THE FIRST AUDIO PROBE RETURNED n=0 FOR BOTH GATES**, which looks
   exactly like "the model does not work". The truth was that the probe
   called the adapter with the WRONG CONVENTION (`model.predict(data,
   sr)` instead of `preprocess(raw)` where raw is a WAV path or a 1-D
   waveform). It now uses the capability's own `_infer_prob`.

**AN n=0 RESULT MUST ALWAYS BE READ AS "THE PROBE CALLED IT WRONG",
NEVER AS "THE MODEL IS BROKEN".**

WHAT THESE TESTS PIN
--------------------
  1. the EER computation is correct on KNOWN cases (perfectly separated,
     inverted, and random);
  2. a NaN score does not corrupt the EER;
  3. the probe's audio path uses the PRODUCTION inference helper, not a
     reimplementation of it;
  4. the probe reports the number of scores it actually got, so n=0 is
     visible.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PROBE = REPO / "tools" / "probe_gate_accuracy.py"


def _load_probe():
    if not PROBE.exists():
        import pytest
        pytest.skip("the gate accuracy probe is not present")
    spec = importlib.util.spec_from_file_location("probe_gate_accuracy", PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "src"))
    spec.loader.exec_module(mod)
    return mod


def _strip_comments_and_strings(src: str) -> str:
    """The source with comments and string literals removed.

    A structural assertion about CODE must not be fooled by a COMMENT
    that mentions the very pattern it forbids -- which is exactly what
    happened here: the probe's docstring explains the `model.predict(data,
    sr)` bug it fixed, and a naive substring search flagged the
    explanation as if it were a live call.
    """
    import io
    import tokenize
    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src
    for tok in toks:
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


def test_the_eer_is_correct_on_known_cases():
    """A perfectly separated detector has a 0% EER. The first version had
    the FAR and FRR swapped and reported 50%."""
    m = _load_probe()
    # perfectly separated: fake high, real low
    scores = [0.99, 0.98, 0.97, 0.01, 0.02, 0.03]
    labels = [1, 1, 1, 0, 0, 0]
    e = m.eer(scores, labels)
    assert e is not None, "the EER is not computable on a clear case"
    assert e == 0.0, (
        f"a perfectly separated detector must have a 0% EER, got {e*100:.2f}%")

    # INVERTED: fake low, real high -> also perfectly separated
    e2 = m.eer([0.01, 0.02, 0.03, 0.97, 0.98, 0.99], [1, 1, 1, 0, 0, 0])
    assert e2 == 0.0, (
        f"an inverted-but-perfectly-separated detector also has a 0% EER, "
        f"got {e2*100:.2f}%")


def test_the_eer_of_a_random_detector_is_about_half():
    """A detector that cannot separate anything has an EER near 50%."""
    m = _load_probe()
    scores = [0.5, 0.51, 0.49, 0.5, 0.52, 0.48]
    labels = [1, 0, 1, 0, 1, 0]
    e = m.eer(scores, labels)
    assert e is not None
    assert 0.3 <= e <= 0.7, (
        f"an uninformative detector should have an EER near 50%, got "
        f"{e*100:.2f}%")


def test_the_eer_ignores_nan_scores():
    """A NaN must not be compared against a threshold — it would land on
    whichever side and silently corrupt the result."""
    m = _load_probe()
    clean = m.eer([0.99, 0.98, 0.01, 0.02], [1, 1, 0, 0])
    with_nan = m.eer([0.99, 0.98, 0.01, 0.02, float("nan")],
                     [1, 1, 0, 0, 1])
    assert clean == with_nan, (
        "a NaN score changed the EER — NaN must be dropped, not compared")


def test_the_probe_uses_the_production_inference_helper():
    """The audio path must call the capability's own `_infer_prob`, not a
    reimplementation of it. Reimplementing it is exactly how a probe ends
    up reporting "the model is broken" when the PROBE is wrong."""
    src = PROBE.read_text()
    assert "_infer_prob" in src, (
        "the audio probe does not use the production inference helper — it "
        "reimplements the calling convention, which is how it silently "
        "scored 0 clips")
    # The check must look at CODE, not at prose: the probe's own COMMENT
    # mentions `model.predict(data, sr)` to explain the bug it fixed, and a
    # naive substring search flags that comment as if it were a live call.
    # So the source is stripped of comments and strings first.
    code = _strip_comments_and_strings(src)
    assert not re.search(r"model\.predict\(\s*data\s*,", code), (
        "the audio probe calls model.predict(data, sr) — the adapter's "
        "contract is preprocess(raw), where raw is a WAV PATH or a 1-D "
        "waveform")


def test_the_probe_reports_the_number_of_scores_it_got():
    """So an n=0 result is VISIBLE rather than a silently empty row."""
    src = PROBE.read_text()
    assert re.search(r'"n":\s*len\(scores\)', src), (
        "the probe does not report how many scores it actually got, so an "
        "n=0 result is invisible in its output")
    assert re.search(r'"errors":\s*errs', src), (
        "the probe does not report how many clips errored, so a systematic "
        "failure is indistinguishable from a small sample")
