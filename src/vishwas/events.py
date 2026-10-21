"""Core event/evidence data structures shared across all layers."""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class InputType(str, Enum):
    """Top-level classification of an incoming item."""
    TEXT = "text"
    URL = "url"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"        # other-file (docs, executables, archives...)
    UNKNOWN = "unknown"


class MediaKind(str, Enum):
    """Verified (magic-byte) content family. May differ from extension."""
    PDF = "pdf"
    MS_OFFICE_DOCX = "ms_office_docx"
    MS_OFFICE_XLSX = "ms_office_xlsx"
    MS_OFFICE_PPTX = "ms_office_pptx"
    APK = "apk"
    JAR = "jar"
    ZIP = "zip"
    GZIP = "gzip"
    SEVEN_ZIP = "7zip"
    RAR = "rar"
    PE = "pe"           # .exe/.dll/.scr
    ELF = "elf"
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"
    TIFF = "tiff"
    HEIC = "heic"
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    OGG_OPUS = "ogg_opus"
    AAC_M4A = "aac_m4a"
    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"
    QUICKTIME = "quicktime"
    AVI = "avi"
    PLAIN_TEXT = "plain_text"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    CSV = "csv"
    SOURCE_CODE = "source_code"
    SCRYPTED_ARCHIVE = "encrypted_archive"  # AEAD-protected zip/rar
    OTHER_BINARY = "other_binary"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class Verdict(str, Enum):
    TRUST = "trust"
    CAUTION = "caution"
    DO_NOT_USE = "do_not_use"
    UNABLE_TO_VERIFY = "unable_to_verify"


class CheckStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"          # ran but with reduced coverage (stale db, missing weights)
    UNAVAILABLE = "unavailable"    # dependency missing -> skipped by design
    FAILED = "failed"              # crashed mid-run; caught, recorded
    SKIPPED = "skipped"            # not run because earlier evidence was decisive / budget


@dataclass(slots=True)
class Artifact:
    """An isolated copy of one user-provided item inside its job quarantine."""
    path: Path
    original_filename: str
    declared_type: InputType                 # what the channel said (hypothesis)
    verified_kind: MediaKind | None = None   # magic-byte truth (set by FileValidator)
    ext_mismatch: bool = False               # declared vs verified disagree
    sha256: str = ""
    size_bytes: int = 0
    md5: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobContext:
    """Everything a capability may need; also carries budgets/gates."""
    job_id: str
    artifact: Artifact
    quarantine_root: Path                    # all derived files go under here
    log_lines: list[str] = field(default_factory=list)
    # gates (env-driven)
    model_weights_available: bool = False
    dynamic_sandbox_available: bool = False
    browser_available: bool = False
    vt_api_key: str | None = None
    llm_available: bool = False
    pades_available: bool = False          # asn1crypto + cryptography importable
    rag_cache_available: bool = False      # RAG template-cache index present+parseable
    # budgets
    # `deadline_mono` defaults to `time.monotonic` — i.e. the budget is ALREADY
    # exhausted the moment the context is created. That default exists so that
    # a context built WITHOUT an explicit deadline fails closed rather than
    # running forever. Since 2026-10-21 it is also a trap: `cap_timeout()`
    # (F6) reads `remaining_s()`, so a context with the default deadline hands
    # every subprocess a 0-second timeout and a healthy scan silently degrades
    # to status="degraded". The orchestrator always passes an explicit
    # deadline; any other caller that does not must pass one too.
    deadline_mono: float = field(default_factory=time.monotonic)
    wall_budget_s: float = 300.0
    extra: dict[str, Any] = field(default_factory=dict)
    # 2.1: persisted at job start so a background heavy-stage follow-up can
    # reach the user after the fast verdict already shipped.
    chat_id: str | None = None

    def remaining_s(self) -> float:
        return max(0.0, self.deadline_mono - time.monotonic())

    def expired(self) -> bool:
        return self.remaining_s() <= 0

    def cap_timeout(self, requested_s: float, floor_s: float = 5.0) -> float:
        """F6 fix (2026-10-21): cap a subprocess/operation timeout at this job's
        remaining wall budget so a single stage can never overrun the job
        envelope. The old static timeouts (mobsf 420s, CAPE/strace 480s, quark
        240s) all exceeded the 300s per-job budget and were never interrupted
        mid-stage, so one stage could run past the whole job budget.

        `floor_s` keeps a degenerate sub-floor timeout from being handed out when
        the remaining budget is generous, but the cap is NEVER lifted above the
        remaining budget — that would re-open the overrun.
        """
        rem = self.remaining_s()
        if rem <= 0:
            return 0.0
        capped = min(float(requested_s), rem)
        if capped < floor_s:
            capped = min(floor_s, rem)   # lift to the floor, but never above budget
        return capped

    def note(self, msg: str) -> None:
        self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def new_job_id(prefix: str = "job") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def hash_file(path: Path, chunk: int = 1 << 20) -> tuple[str, str, int]:
    h_sha, h_md5, n = hashlib.sha256(), hashlib.md5(), 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h_sha.update(b)
            h_md5.update(b)
            n += len(b)
    return h_sha.hexdigest(), h_md5.hexdigest(), n
