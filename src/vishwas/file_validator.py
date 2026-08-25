"""FileValidator: verify what a file *actually* is via magic bytes / container sniffing.

The declared extension is treated as a hypothesis only. A mismatch between the
declared type and the verified kind is itself a security signal (a .jpg that is
really an .apk, or a .txt polyglot) and is surfaced to the caller.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Union
from pathlib import Path

from .events import Artifact, JobContext, MediaKind, InputType, new_job_id, hash_file

MagicPred = Union[bytes, Callable[[bytes], bool]]

MAGIC_TABLE: list[tuple[MagicPred, MediaKind]] = [
    (lambda b: b[:4] == b"%PDF", MediaKind.PDF),
    (lambda b: b[:4] == b"\xd0\xcf\x11\xe0" and b[0x1e:0x26] == b"Macros\0\0".rstrip(b"\0"), MediaKind.MS_OFFICE_DOCX),  # CFB macro-ish; refined below
    (lambda b: b[:4] == b"\x50\x4b\x03\x04" and _zip_member_is_apk(b), MediaKind.APK),
    (lambda b: b[:4] == b"\x50\x4b\x03\x04", MediaKind.ZIP),  # refined inside _refine_zip
    (lambda b: b[:2] == b"\x1f\x8b", MediaKind.GZIP),
    (lambda b: b[:6] == b"7z\xbc\xaf\x27\x1c", MediaKind.SEVEN_ZIP),
    (lambda b: b[:4] == b"Rar!\x1a\x07", MediaKind.RAR),
    (lambda b: b[:2] == b"MZ", MediaKind.PE),
    # P7 red-team: shebang scripts are EXECUTABLE content, never plain text —
    # a '#!/bin/sh' payload named *.txt must be recognised as code, not prose.
    (lambda b: b.startswith(b"#!/") , MediaKind.SOURCE_CODE),
    (lambda b: b[:4] == b"\x7fELF", MediaKind.ELF),
    (lambda b: b[:8] == b"\x89PNG\r\n\x1a\n", MediaKind.PNG),
    (lambda b: b[:3] == b"\xff\xd8\xff", MediaKind.JPEG),
    (lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP", MediaKind.WEBP),
    (lambda b: b[:6] in (b"GIF87a", b"GIF89a"), MediaKind.GIF),
    (lambda b: b[:4] in (b"II*\x00", b"MM\x00*"), MediaKind.TIFF),
    (lambda b: b[4:12] == b"ftypheic" or (b[4:12] == b"ftypmif1"), MediaKind.HEIC),
    (lambda b: b[:3] == b"ID3" or (len(b) > 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0), MediaKind.MP3),
    (lambda b: b[:4] == b"RIFF" and b[8:12] == b"WAVE", MediaKind.WAV),
    (lambda b: b[:4] == b"fLaC", MediaKind.FLAC),
    (lambda b: b[:4] == b"OggS", MediaKind.OGG_OPUS),
    (lambda b: b[4:12] == b"ftypM4A " or b[4:12] == b"ftypM4B ", MediaKind.AAC_M4A),
    (lambda b: b[4:12] == b"ftypmp42" or b[4:12] == b"ftypisom" or b[4:12] == b"ftypmp41" or b[4:12] == b"ftypqt  ", MediaKind.MP4),
    (lambda b: b[4:8] == b"webm" or b[4:8] == b"MKV\x99", MediaKind.MKV),
    (lambda b: b[4:8] == b"\x01\x00\x00\x00" and len(b) > 8 and b[0:4] == b"\x1a\x45\xdf\xa3", MediaKind.MKV),  # EBML header
    (lambda b: b[:4] == b"RIFF" and b[8:12] == b"AVI ", MediaKind.AVI),
    (lambda b: b[0:1] == b"#" and re.search(rb"<\?xml", b[:4096]), MediaKind.XML),
    (lambda b: _looks_like_json(b), MediaKind.JSON),
    (lambda b: _looks_like_csv(b), MediaKind.CSV),
    (lambda b: _looks_like_html(b), MediaKind.HTML),
    (lambda b: _looks_like_source(b), MediaKind.SOURCE_CODE),
]

_TEXTISH_PREFIXES = (b"<?xml", b"{", b"[", b"#include", b"#!/", b"<!DOCTYPE", b"<!--")


def _zip_member_is_apk(head: bytes) -> bool:
    """Quick check for AndroidManifest / classes.dex inside a ZIP container."""
    return b"AndroidManifest" in head[:65536] or b"classes.dex" in head[:65536]


def _refine_zip(raw: bytes) -> MediaKind:
    if b"classes.dex" in raw[:262144] or b"AndroidManifest" in raw[:262144]:
        return MediaKind.APK
    head = raw[:262144].decode("latin-1", errors="ignore")
    for marker in ("[Content_Types].xml",):
        if marker in head:
            return _ms_kind_from_content_types(raw)
    return MediaKind.ZIP


def _ms_kind_from_content_types(raw: bytes) -> MediaKind:
    head = raw[:262144].decode("latin-1", errors="ignore")
    if "word/" in head:
        return MediaKind.MS_OFFICE_DOCX
    if "xl/" in head:
        return MediaKind.MS_OFFICE_XLSX
    if "ppt/" in head:
        return MediaKind.MS_OFFICE_PPTX
    return MediaKind.ZIP


def _looks_like_json(b: bytes) -> bool:
    s = b.strip()
    if not s:
        return False
    if s[:1] in (b"{", b"["):
        try:
            json.loads(s[:65536].decode("utf-8"))
            return True
        except Exception:
            pass
    return False


def _looks_like_csv(b: bytes) -> bool:
    lines = b[:16384].splitlines()
    if not lines:
        return False
    first = lines[0].decode("utf-8", errors="ignore")
    cols = first.count(",")
    return 1 <= cols <= 30 and sum(1 for ln in lines[:21] if ln.decode("utf-8", "ignore").count(",") == cols) >= max(2, len(lines) // 2)


def _looks_like_html(b: bytes) -> bool:
    s = b[:16384].lower().decode("utf-8", errors="ignore")
    return ("<html" in s) or ("</html>" in s) or ("<!doctype html" in s)


def _looks_like_source(b: bytes) -> bool:
    s = b[:8192]
    if b"\x00" in s:
        return False
    text = s.decode("utf-8", errors="ignore").lower()
    hits = 0
    for kw in ("def ", "import ", "function ", "const ", "#include", "public class", "package ", "fn main", "void ", "int main"):
        if kw in text:
            hits += 1
    return hits >= 2


def classify_bytes(raw: bytes) -> MediaKind:
    if len(raw) < 2:
        return MediaKind.EMPTY if not raw else MediaKind.UNKNOWN
    if raw.startswith(b"\x50\x4b\x03\x04"):
        return _refine_zip(raw)
    for pred, kind in MAGIC_TABLE:
        try:
            if isinstance(pred, bytes):
                if raw.startswith(pred):
                    return kind
            else:
                if pred(raw):
                    return kind
        except Exception:
            continue
    # text heuristics last (JSON/HTML/XML/source/plain)
    if b"\x00" not in raw[:4096]:
        s = raw[:4096]
        prefixes = _TEXTISH_PREFIXES if isinstance(_TEXTISH_PREFIXES, tuple) else (_TEXTISH_PREFIXES,)
        if s.lstrip().startswith(prefixes) or re.match(rb"^[\x20-\x7e\n\r\t\f]+", s):
            if _looks_like_json(raw):
                return MediaKind.JSON
            if _looks_like_html(raw):
                return MediaKind.HTML
            if _looks_like_csv(raw):
                return MediaKind.CSV
            if _looks_like_source(raw):
                return MediaKind.SOURCE_CODE
            return MediaKind.PLAIN_TEXT
    return MediaKind.OTHER_BINARY


def _encrypted_archive_check(kind: MediaKind, raw: bytes) -> bool:
    if kind in (MediaKind.ZIP, MediaKind.RAR, MediaKind.SEVEN_ZIP):
        try:
            if kind is MediaKind.ZIP:
                # zip encryption flag per local header
                return bool(raw[:262144] & 0x0000 if False else (len(raw) > 5 and (raw[6] << 8 | raw[5]) & 0x0001 != 0))
        except Exception:
            pass
    return False


_EXT_TO_KIND: dict[str, MediaKind] = {
    "pdf": MediaKind.PDF,
    "docx": MediaKind.MS_OFFICE_DOCX,
    "doc": MediaKind.MS_OFFICE_DOCX,
    "xlsx": MediaKind.MS_OFFICE_XLSX,
    "xls": MediaKind.MS_OFFICE_XLSX,
    "pptx": MediaKind.MS_OFFICE_PPTX,
    "ppt": MediaKind.MS_OFFICE_PPTX,
    "apk": MediaKind.APK,
    "jar": MediaKind.JAR,
    "zip": MediaKind.ZIP,
    "gz": MediaKind.GZIP,
    "7z": MediaKind.SEVEN_ZIP,
    "rar": MediaKind.RAR,
    "exe": MediaKind.PE,
    "dll": MediaKind.PE,
    "scr": MediaKind.PE,
    "elf": MediaKind.ELF,
    "so": MediaKind.ELF,        # linux shared object — executable, treat as malicious candidate
    "dylib": MediaKind.ELF,    # macos shared library
    "bin": MediaKind.OTHER_BINARY,   # generic binary payload
    "o": MediaKind.OTHER_BINARY,     # object file
    "png": MediaKind.PNG,
    "jpg": MediaKind.JPEG,
    "jpeg": MediaKind.JPEG,
    "webp": MediaKind.WEBP,
    "gif": MediaKind.GIF,
    "tif": MediaKind.TIFF,
    "tiff": MediaKind.TIFF,
    "heic": MediaKind.HEIC,
    "heif": MediaKind.HEIC,
    "mp3": MediaKind.MP3,
    "wav": MediaKind.WAV,
    "flac": MediaKind.FLAC,
    "ogg": MediaKind.OGG_OPUS,
    "opus": MediaKind.OGG_OPUS,
    "m4a": MediaKind.AAC_M4A,
    "aac": MediaKind.AAC_M4A,
    "mp4": MediaKind.MP4,
    "mov": MediaKind.QUICKTIME,
    "mkv": MediaKind.MKV,
    "webm": MediaKind.WEBM,
    "avi": MediaKind.AVI,
    "txt": MediaKind.PLAIN_TEXT,
    "md": MediaKind.PLAIN_TEXT,
    "log": MediaKind.PLAIN_TEXT,
    "csv": MediaKind.CSV,
    "json": MediaKind.JSON,
    "xml": MediaKind.XML,
    "html": MediaKind.HTML,
    "htm": MediaKind.HTML,
    "py": MediaKind.SOURCE_CODE,
    "js": MediaKind.SOURCE_CODE,
    "sh": MediaKind.SOURCE_CODE,
}


class FileValidator:
    """Verify actual content vs declared type; set sha256/md5 on the artifact."""

    SNIFF_BYTES = 262144

    def validate(self, art: Artifact, ctx: JobContext | None = None) -> tuple[MediaKind, bool]:
        """Return (verified_kind, ext_mismatch). Mutates art in place."""
        path: Path = art.path
        raw = path.read_bytes()[: self.SNIFF_BYTES] if path.exists() else b""
        if path.exists():
            art.sha256, art.md5, art.size_bytes = hash_file(path)
        kind = classify_bytes(raw)
        art.verified_kind = kind
        art.meta["sniff_head"] = raw[:64].hex()

        declared_kind = _EXT_TO_KIND.get(art.original_filename.rsplit(".", 1)[-1].lower() if "." in art.original_filename else "", MediaKind.UNKNOWN)
        mismatch = False
        if declared_kind is not MediaKind.UNKNOWN and kind is not declared_kind:
            # allow container families that refine from each other
            same_family = (
                (kind in (MediaKind.APK, MediaKind.JAR, MediaKind.ZIP) and declared_kind in (MediaKind.APK, MediaKind.JAR, MediaKind.ZIP))
                or (kind in (MediaKind.MS_OFFICE_DOCX, MediaKind.MS_OFFICE_XLSX, MediaKind.MS_OFFICE_PPTX, MediaKind.ZIP))
                or (kind in (MediaKind.WAV, MediaKind.MP4, MediaKind.MKV, MediaKind.MP3, MediaKind.AAC_M4A, MediaKind.QUICKTIME, MediaKind.AVI) and declared_kind in (MediaKind.WAV, MediaKind.MP4, MediaKind.MKV, MediaKind.MP3, MediaKind.AAC_M4A, MediaKind.QUICKTIME, MediaKind.AVI))
            )
            mismatch = not same_family
        art.ext_mismatch = mismatch
        if ctx:
            ctx.note(f"file-validated: declared={art.declared_type.value} verified={kind.value} mismatch={mismatch}")
        return kind, mismatch


def make_artifact(job_dir: Path, filename: str, declared: InputType, data: bytes | None = None) -> Artifact:
    """Materialize user data into the job quarantine (always under job_dir)."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)[:120] or "upload.bin"
    p = job_dir / "original" / safe
    p.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        p.write_bytes(data)
    art = Artifact(path=p, original_filename=safe, declared_type=declared)
    return art
