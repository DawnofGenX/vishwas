"""Unit tests for the Python-API docling branch (gov_document._try_docling_python).

Hermetic: docling itself is NEVER loaded here. We fake it via sys.modules + a
patched importlib.util.find_spec, which mirrors exactly what the production
code inspects. Covers: gated-off, low-budget, success(str), bytes->str
normalization, ladder ordering vs pypdf, and failure-fallthrough.
"""
from __future__ import annotations

import importlib.util as _iu
import io
import sys
import time

import pytest

from verisafe.capabilities import gov_document as gd
from verisafe.events import Artifact, InputType, JobContext, MediaKind


@pytest.fixture(autouse=True)
def _reset_globals():
    """Each test starts from a clean 'untried' state of the lazy loader."""
    gd._DOCLING_CONV = None
    gd._DOCLING_TRIED = False
    yield
    gd._DOCLING_CONV = None
    gd._DOCLING_TRIED = False


# ---------------------------------------------------------------- fakes ----
class _FakeDoc:
    def __init__(self, out):
        self._out = out

    def export_to_markdown(self):
        return self._out


class _FakeResult:
    def __init__(self, out):
        self.document = _FakeDoc(out)


_FAKE = {"out": None, "raise": None}


class _FakeConverter:
    def __init__(self):
        pass

    def convert(self, source, **kwargs):
        if _FAKE["raise"] is not None:
            raise _FAKE["raise"]
        return _FakeResult(_FAKE["out"])


class _FakeStream:
    def __init__(self, name, stream):
        self.name = name
        self.stream = stream


def _install_fake_docling(monkeypatch, out: object):
    """Register fake docling modules and make find_spec('docling') succeed."""
    import types

    mod_dl = types.ModuleType("docling")
    mod_dd = types.ModuleType("docling.datamodel")
    mod_bm = types.ModuleType("docling.datamodel.base_models")
    mod_cc = types.ModuleType("docling.document_converter")
    mod_bm.DocumentStream = _FakeStream
    mod_cc.DocumentConverter = _FakeConverter
    mod_dl.datamodel = mod_dd
    mod_dd.base_models = mod_bm
    for name, m in (("docling", mod_dl), ("docling.datamodel", mod_dd),
                    ("docling.datamodel.base_models", mod_bm),
                    ("docling.document_converter", mod_cc)):
        monkeypatch.setitem(sys.modules, name, m)

    real_find_spec = _iu.find_spec

    def find_spec_shim(name):
        if name == "docling":
            class _S:  # minimal stand-in; only truthiness is consumed
                pass
            return _S()
        return real_find_spec(name)

    monkeypatch.setattr(_iu, "find_spec", find_spec_shim)
    _FAKE["out"] = out
    _FAKE["raise"] = None


def _mk_ctx(tmp_path, remaining_s: float) -> tuple[Artifact, JobContext]:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")
    art = Artifact(path=p, original_filename="doc.pdf",
                   declared_type=InputType.FILE, verified_kind=MediaKind.PDF)
    ctx = JobContext(job_id="t", artifact=art, quarantine_root=tmp_path,
                     deadline_mono=time.monotonic() + remaining_s)
    return art, ctx


# --------------------------------------------------------------- tests -----
def test_gated_off_returns_none(tmp_path):
    """No docling anywhere -> instant None, no side effects, no crash."""
    art, ctx = _mk_ctx(tmp_path, 300)
    assert gd._try_docling_python(art, ctx) is None
    # second call also fast-path (cache recorded as unavailable)
    assert gd._DOCLING_TRIED is True
    assert gd._try_docling_python(art, ctx) is None


def test_low_budget_skips_conversion(tmp_path, monkeypatch):
    _install_fake_docling(monkeypatch, out="should never be reached")
    art, ctx = _mk_ctx(tmp_path, remaining_s=20)   # < 45s gate
    assert gd._try_docling_python(art, ctx) is None


def test_success_returns_markdown_str(tmp_path, monkeypatch):
    body = "Income Tax Department  PAN: ABCDE1234F"
    _install_fake_docling(monkeypatch, out=body)
    art, ctx = _mk_ctx(tmp_path, 300)
    res = gd._try_docling_python(art, ctx)
    assert res is not None
    text, name = res
    assert name == "docling"
    assert isinstance(text, str)
    assert "ABCDE1234F" in text
    # converter got cached for the process
    assert gd._DOCLING_CONV is not None


def test_bytes_output_is_normalized_to_str(tmp_path, monkeypatch):
    """Defensive path: some builds may hand back bytes; downstream must get str."""
    _install_fake_docling(monkeypatch, out=b"\xff\xfe Income Tax PAN")
    art, ctx = _mk_ctx(tmp_path, 300)
    res = gd._try_docling_python(art, ctx)
    assert res is not None
    text, name = res
    assert isinstance(text, str)          # bytes were decoded, not leaked
    assert text != b""                    # non-empty after decode


def test_empty_output_falls_through(tmp_path, monkeypatch):
    _install_fake_docling(monkeypatch, out="   \n\t ")
    art, ctx = _mk_ctx(tmp_path, 300)
    assert gd._try_docling_python(art, ctx) is None


def test_convert_exception_falls_through(tmp_path, monkeypatch):
    """Conversion failing (bad file etc.) must not break the extraction ladder."""
    _install_fake_docling(monkeypatch, out=None)
    _FAKE["raise"] = RuntimeError("Conversion failed for: .pdf")
    art, ctx = _mk_ctx(tmp_path, 300)
    assert gd._try_docling_python(art, ctx) is None
    # and _extract_text keeps going down its ladder without raising
    try:
        gd._extract_text(art, ctx)
    except Exception as e:  # pragma: no cover - ladder is designed not to raise
        pytest.fail(f"_extract_text raised: {e!r}")


def test_ladder_prefers_docling_over_pypdf(tmp_path, monkeypatch):
    """When both docling (branch 1b) and pypdf could win, docling wins."""
    body = "Voter ID Card Elector"
    _install_fake_docling(monkeypatch, out=body)
    art, ctx = _mk_ctx(tmp_path, 300)   # kind=PDF => pypdf branch eligible
    text, name = gd._extract_text(art, ctx)
    assert name == "docling"
    assert "Voter" in text
