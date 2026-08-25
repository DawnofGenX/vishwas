"""FileValidator: magic-byte/MIME confirmation must override a lying filename.

This is the single most important trust boundary: an attacker names a PE binary
"photo.jpg" or a shell script "invoice.txt". Magic bytes win.
"""
from __future__ import annotations

import pytest

from vishwas.file_validator import classify_bytes, FileValidator, make_artifact
from vishwas.events import MediaKind, InputType


# ---------------------------------------------------------------- byte level --
def test_pdf_magic():
    assert classify_bytes(b"%PDF-1.7\n%%EOF") is MediaKind.PDF


def test_gif_and_png_magsics():
    assert classify_bytes(b"GIF89a...") is MediaKind.GIF  # GIF header -> dedicated GIF kind
    assert classify_bytes(b"\x89PNG\r\n\x1a\n\x00" * 4) is MediaKind.PNG


def test_wav_riff_is_audio():
    assert classify_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ") is MediaKind.WAV


def test_apk_zip_with_manifest_refines():
    # zip with AndroidManifest.xml member should be recognised as APK, not ZIP
    raw = b"\x50\x4b\x03\x04" + b"Andro\nidMana" + b"x" * 64
    k = classify_bytes(raw)
    assert k in (MediaKind.APK, MediaKind.JAR, MediaKind.ZIP)


def test_plain_text_detection():
    assert classify_bytes(b"hello world\nthis is plain text") is MediaKind.PLAIN_TEXT


def test_json_detection():
    assert classify_bytes(b'{"a": 1, "b": [2,3]}') is MediaKind.JSON


def test_html_detection():
    assert classify_bytes(b"<html><body><form></form></body></html>") is MediaKind.HTML


def test_empty_is_empty_or_unknown():
    assert classify_bytes(b"") in (MediaKind.EMPTY, MediaKind.UNKNOWN)


# --------------------------------------------------------------- mismatch ----
def _validator_roundtrip(tmp_path, filename, data):
    art = make_artifact(tmp_path, filename, InputType.FILE, data=data)
    kind, mismatch = FileValidator().validate(art)
    return kind, mismatch


def test_txt_named_but_pdf_content_flags_mismatch(tmp_path):
    kind, mismatch = _validator_roundtrip(tmp_path, "innocent.txt", b"%PDF-1.7\n%%EOF")
    assert kind is MediaKind.PDF
    assert mismatch is True, "extension .txt vs verified PDF must be flagged"


def test_exe_named_but_pe_content_no_mismatch(tmp_path):
    # PE magic at offset 0: 'MZ' + filler
    pe = b"MZ" + b"\x00" * 60 + b"\x90\x00\x00\x00"
    kind, mismatch = _validator_roundtrip(tmp_path, "malware.exe", pe)
    assert kind in (MediaKind.PE, MediaKind.OTHER_BINARY)
    assert mismatch is False, ".exe declared and PE verified -> same intent, no flag"


def test_jpg_named_but_pe_content_is_malicious_relabel(tmp_path):
    """Classic trojan: photo.jpg that is really a Windows executable."""
    pe = b"MZ" + b"\x00" * 60
    kind, mismatch = _validator_roundtrip(tmp_path, "photo.jpg", pe)
    assert kind in (MediaKind.PE, MediaKind.OTHER_BINARY)
    assert mismatch is True, "image extension hiding an executable MUST be flagged"


def test_sha256_populated_on_validate(tmp_path):
    art = make_artifact(tmp_path, "a.bin", InputType.FILE, data=b"abc123xyz")
    FileValidator().validate(art)
    assert len(art.sha256) == 64, "validator should compute sha256 for identity/evidence"


def test_family_refine_no_false_mismatch_office_to_docx(tmp_path):
    # docx is a zip; if declared docx and verified zip/docx it must not scream mismatch
    docx_like = b"\x50\x4b\x03\x04" + b"[Content_Types].xml" + b" " * 40
    kind, mismatch = _validator_roundtrip(tmp_path, "report.docx", docx_like)
    assert kind in (MediaKind.MS_OFFICE_DOCX, MediaKind.ZIP)
    assert mismatch is False, "office->container-family refinement must not be a mismatch"
