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

import ast
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


# ⚠️ THE PER-CHANNEL TESTS ABOVE ARE WHY THE AUDIO CHANNEL WAS MISSED
# (2026-10-22). PR #34 fixed the image bug and pinned the image and video
# channels BY NAME — and the audio channel was left with TWO bare
# `except Exception: continue` (the sf.read decode and the sf.write to
# disk), so a corpus row that failed to decode still vanished without a
# count. The docstring claimed "no channel in the probe may swallow a
# failure without counting it"; the tests only enforced it for two of the
# three channels.
#
# ⇒ A RULE ENFORCED CHANNEL-BY-CHANNEL IS A RULE THAT DECAYS EVERY TIME A
#   CHANNEL IS ADDED. Enforce it over EVERY `probe_*` function instead.
def test_every_channel_counts_what_it_swallows():
    """EVERY `except: continue` in EVERY channel must increment a counter.

    This is the generic form of the image/video checks above: it catches a
    new channel the day it is added, instead of the day someone remembers
    to write a test for it.

    Parsed with `ast`, not with a regex: a regex that grabs "the indented
    lines after the except" swallows the NEXT handler too, and then finds
    that handler's `+= 1` and clears the guilty one. (That is not a
    hypothetical — the first version of this test did exactly that and
    MISSED a reintroduced audio drop.)
    """
    src = _src()
    tree = ast.parse(src)
    channels = [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name.startswith("probe_")]
    assert len(channels) >= 3, (
        "expected at least the audio, image and video channels, found "
        f"{[c.name for c in channels]}")
    offenders = []
    for fn in channels:
        for h in [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]:
            body = [n for n in ast.walk(h)]
            if not any(isinstance(n, ast.Continue) for n in body):
                continue
            # A handler is allowed to `continue` if it leaves a TRACE:
            #   * a counter  -- `skipped += 1` / `n = n + 1`; or
            #   * a record   -- `out.append({"gate": ..., "error": ...})`,
            #     which is how a failed gate LOAD is reported: it is not a
            #     dropped corpus item, it is a diagnosis the caller prints.
            # What is forbidden is the bare `except: continue` that leaves
            # nothing behind.
            counted = any(isinstance(n, ast.AugAssign) for n in body) or any(
                isinstance(n, ast.Assign)
                and isinstance(n.value, ast.BinOp)
                and isinstance(n.value.op, ast.Add) for n in body)
            recorded = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in {"append", "extend"} for n in body)
            if not (counted or recorded):
                offenders.append(f"{fn.name}():{h.lineno}")
    assert not offenders, (
        "these handlers swallow a failure with `continue` and count nothing, "
        f"so a dropped corpus item is invisible: {offenders}")


def test_the_audio_channel_counts_dropped_corpus_items(monkeypatch, tmp_path):
    """BEHAVIOURAL: a row that fails to decode must be COUNTED, not lost.

    The static test above pins the SHAPE. This one proves the counter moves
    on a payload the probe has to drop — because a counter that is
    incremented in no reachable branch is just as silent as no counter.

    The heavy dependencies (pyarrow, soundfile, the vishwas capability and
    adapter modules) are stubbed: this test is about the probe's
    bookkeeping, not about a model.
    """
    import importlib.util
    import sys
    import types

    probe_path = REPO / "tools" / "probe_gate_accuracy.py"
    if not probe_path.exists():
        import pytest
        pytest.skip("the gate accuracy probe is not present")

    # --- the corpus: 4 rows, row 1 fails to DECODE, row 2 fails to WRITE
    rows = [{"bytes": f"row{i}".encode()} for i in range(4)]

    class _Table:
        def __init__(self, cols):
            self._cols = cols

        def column(self, name):
            class _Col:
                def __init__(self, vals):
                    self._vals = vals

                def to_pylist(self):
                    return self._vals
            return _Col(self._cols[name])

    fake_pq = types.ModuleType("pyarrow.parquet")
    fake_pq.read_table = lambda path, columns=None: _Table(
        {"audio": rows, "key": [0, 0, 1, 1]})
    fake_pyarrow = types.ModuleType("pyarrow")
    fake_pyarrow.parquet = fake_pq

    calls = {"read": 0, "write": 0}

    class _Wave(list):
        """A 1-D waveform stand-in: `ndim` is what the probe reads."""
        ndim = 1

    def _read(buf, dtype=None):
        calls["read"] += 1
        if calls["read"] == 2:
            raise RuntimeError("decode failed")
        return _Wave([0.0] * 16), 16000

    def _write(path, data, sr):
        calls["write"] += 1
        if calls["write"] == 2:
            raise OSError("disk full")

    fake_sf = types.ModuleType("soundfile")
    fake_sf.read = _read
    fake_sf.write = _write

    fake_audio = types.ModuleType("vishwas.capabilities.deepfake_audio")
    fake_audio._infer_prob = lambda gate, model, path: 0.9

    class _Gate:
        def load(self, path):
            return object()

    fake_adapters = types.ModuleType("vishwas.model_adapters")
    fake_adapters.resolve = lambda env_var: _Gate()

    fake_pkg = types.ModuleType("vishwas")
    fake_pkg.__path__ = []
    fake_caps = types.ModuleType("vishwas.capabilities")
    fake_caps.__path__ = []
    fake_pkg.capabilities = fake_caps
    fake_caps.deepfake_audio = fake_audio
    fake_pkg.model_adapters = fake_adapters

    for name, mod in [("pyarrow", fake_pyarrow),
                      ("pyarrow.parquet", fake_pq),
                      ("soundfile", fake_sf),
                      ("vishwas", fake_pkg),
                      ("vishwas.capabilities", fake_caps),
                      ("vishwas.capabilities.deepfake_audio", fake_audio),
                      ("vishwas.model_adapters", fake_adapters)]:
        monkeypatch.setitem(sys.modules, name, mod)

    # the probe makes its own temp tree at import; keep it inside tmp_path
    import tempfile
    monkeypatch.setattr(tempfile, "mkdtemp",
                        lambda **kw: str(tmp_path / "probe-q"))

    spec = importlib.util.spec_from_file_location("_probe_under_test",
                                                  probe_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # point the gate env vars at a real file so the is-file preflight passes,
    # and give the module a parquet path that EXISTS (its contents are stubbed)
    weight = tmp_path / "fake_gate.pth"
    weight.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("VISHWAS_AASIST_WEIGHTS", str(weight))
    monkeypatch.delenv("VISHWAS_XLSRMAMBA_WEIGHTS", raising=False)
    parquet = tmp_path / "fake.parquet"
    parquet.write_bytes(b"")
    mod.PARQUET = parquet
    mod.Q = tmp_path / "probe-q"

    recs = mod.probe_audio(n=60)
    rec = recs[0]
    assert rec["gate"] == "aasist3"
    assert "skipped" in rec, (
        "the audio record has no `skipped` field, so a dropped corpus item "
        "cannot be reported")
    assert rec["skipped"] == 2, (
        f"expected 2 counted drops (1 decode + 1 write), got {rec['skipped']}")
    assert rec["n"] == 2, (
        f"expected the 2 scoreable rows to be scored, got n={rec['n']}")

