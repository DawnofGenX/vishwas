"""Deterministic input routing: classify incoming items, pick capability targets.

Rules (deterministic, no LLM involvement):
  url text or media_path ending .html -> phishing/url flow (downloaded files
      re-enter file validation inside that capability)
  video extensions/kind -> deepfake_video
  audio kind -> deepfake_audio (+ cross-modal if video carries audio — handled
      by orchestrator target 'media_av' when both streams present)
  image -> deepfake_video frame-analysis path (static face check)
  doc kinds (pdf/office/source/text) -> govdoc when issuer signals hit, else generic
  other-file (apk/pe/archive/binary) -> malware
Priority on ties: security-relevant targets (malware/phishing) win because
they're higher-harm; detection targets only run on genuine media/documents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .events import InputType, MediaKind

_MEDIA_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".3gp", ".wmv"}
_MEDIA_AUDIO_EXT = {".mp3", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".flac", ".wma"}
_IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".bmp", ".tiff"}
_DOC_KINDS = {MediaKind.PDF, MediaKind.MS_OFFICE_DOCX, MediaKind.MS_OFFICE_XLSX,
              MediaKind.MS_OFFICE_PPTX}
_EXEC_KINDS = {MediaKind.PE, MediaKind.ELF}
_PKG_KINDS = {MediaKind.APK, MediaKind.JAR, MediaKind.ZIP, MediaKind.RAR,
              MediaKind.SEVEN_ZIP, MediaKind.GZIP}

_URL_RE = re.compile(r"^(?:https?://|www\.)[^\s]+(?:/[^\s]*)?$/i")
_URL_IN_TEXT_RE = re.compile(
    r"(?:https?://|www\.)[a-z0-9\-._~%]+(?::\d+)?(?:/[^\s<>\"']+)?", re.I)

_GOV_HINTS = re.compile(
    r"(?i)\b(aadhaar|pan\b|voter\s*(id|card)|pass(port|book)|driv?er'?s?\s*licen[sc]e"
    r"|election\s*commiss|epfo|esic|pension|ayushman|rashan|ration\s*card"
    r"|land\s*record[s]?|khata\s*copy|encumbrance\s*certif"
    r"|income\s*certificate|caste\s*certificate|birth\s*certificate"
    r"|marriage\s*certificate|degree\s*verific|salary\s*slip|pay\s*slip"
    r"|bank\s*statement|nsdl|nsdl\.in|passport\.gov\.in|indiabillboard)"
)


@dataclass(slots=True)
class RouteDecision:
    text: str | None
    is_url: bool
    url: str | None          # normalized URL string if is_url (or first URL found in text)
    urls_in_text: list[str]
    input_type: InputType   # what we believe the *message* contains
    media_kind: MediaKind | None
    target_hint: str        # suggested capability target name (router may override)
    text_payload: str       # non-URL part of a text message (may be empty)

    @property
    def has_media(self) -> bool:
        return self.input_type in (InputType.IMAGE, InputType.AUDIO, InputType.VIDEO, InputType.FILE)


class Router:
    """Pure functions over (text, media_path, media_type). No I/O."""

    def classify(self, msg: dict) -> RouteDecision:
        text = (msg.get("text") or "").strip()
        media_path = msg.get("media_path")
        media_type = (msg.get("media_type") or "").lower()

        raw_urls = _URL_IN_TEXT_RE.findall(text)
        bare_is_url = bool(text) and (_URL_RE.match(text.replace(" ", "")) or raw_urls)
        url = raw_urls[0] if raw_urls else (text if (bare_is_url and not media_path) else None)

        # decide input type
        itype = msg.get("input_type")  # channel may hint
        if itype:
            try:
                input_type = InputType(itype)
            except ValueError:
                input_type = InputType.UNKNOWN
        elif media_path:
            ext = "." + media_path.rsplit(".", 1)[-1].lower() if "." in media_path else ""
            if ext in _MEDIA_VIDEO_EXT or media_type == "video":
                input_type = InputType.VIDEO
            elif ext in _MEDIA_AUDIO_EXT or media_type == "audio":
                input_type = InputType.AUDIO
            elif ext in _IMG_EXT or media_type == "image":
                input_type = InputType.IMAGE
            else:
                input_type = InputType.FILE
        elif url:
            input_type = InputType.URL
        elif text:
            input_type = InputType.TEXT
        else:
            input_type = InputType.UNKNOWN

        media_kind = None
        if media_path:
            from .file_validator import _EXT_TO_KIND
            ext = ("." + media_path.rsplit(".", 1)[-1]) if "." in media_path else ""
            media_kind = _EXT_TO_KIND.get(ext.lower().lstrip("."), None)

        decision = RouteDecision(text=text or None,
                                 is_url=bool(url) and not media_path,
                                 url=url,
                                 urls_in_text=[u for u in raw_urls],
                                 input_type=input_type,
                                 media_kind=media_kind,
                                 target_hint="",
                                 text_payload=(text or "")[:2000])
        decision.target_hint = self._hint(decision)
        return decision

    def target_for(self, d: RouteDecision, art=None) -> str:
        """Final target after magic-byte validation (orchestrator calls this)."""
        kind = getattr(art, "verified_kind", None)
        h = self._hint(d, kind, original_filename=getattr(art, "original_filename", ""))
        return h or "unclassified"

    def _hint(self, d: RouteDecision, kind: MediaKind | None = None,
              original_filename: str = "") -> str:
        k = kind or d.media_kind
        if d.is_url or (d.input_type is InputType.URL and d.url):
            return "url_phishing"
        if d.input_type in (InputType.VIDEO, InputType.AUDIO, InputType.IMAGE):
            if k in _EXEC_KINDS or k in _PKG_KINDS:
                return "malicious_file"          # executable disguised as media
            if d.input_type is InputType.VIDEO or k in ({MediaKind.MP4, MediaKind.MKV,
                                                         MediaKind.WEBM, MediaKind.QUICKTIME, MediaKind.AVI}):
                return "deepfake_video"
            if d.input_type is InputType.AUDIO or k in (MediaKind.MP3, MediaKind.WAV,
                                                        MediaKind.FLAC, MediaKind.OGG_OPUS, MediaKind.AAC_M4A):
                return "deepfake_audio"
            if d.input_type is InputType.IMAGE:
                return "image_facecheck"
        if k in _DOC_KINDS or k in (MediaKind.PLAIN_TEXT, MediaKind.HTML,
                                    MediaKind.SOURCE_CODE, MediaKind.JSON, MediaKind.CSV):
            blob = ((d.text_payload or "") + " ").lower()
            # Finding D fix (2026-08-21): the real filename was never checked —
            # only the target_hint string ("document_generic"), so a PDF named
            # aadhaar.pdf always routed document_generic via CLI.
            if (_GOV_HINTS.search(blob)
                    or _looks_gov_artifact(art_filename=getattr(d, "target_hint", ""))
                    or _looks_gov_artifact(art_filename=original_filename)):
                return "gov_document"
            return "document_generic"
        if k in _EXEC_KINDS or k in _PKG_KINDS:
            return "malicious_file"
        if k in (MediaKind.EMPTY, MediaKind.UNKNOWN) or k is None:
            return "unclassified"
        return "malicious_file"


def _looks_gov_artifact(art_filename: str = "") -> bool:
    if not art_filename:
        return False
    return bool(_GOV_HINTS.search(art_filename.lower()))


def extract_urls(text: str) -> list[str]:
    return _URL_IN_TEXT_RE.findall(text or "")
