"""Offline QR verification for Indian identity documents.

Public surface::

    verify_payload(payload, *, extra_trust_paths=None) -> QrVerifyResult
    decode_image(path_or_pil_or_ndarray) -> list[str]

Formats were reverse-engineered per the repo's docs/research QR scouting
report; method provenance: StarkAg/aadhaar-secure-qr-verifier (MIT) for the
Aadhaar Secure-QR chain and captn3m0/epicqr (MIT) for the EPIC envelope —
ports of method, not copies of code. Fully offline: no network calls
anywhere in this package.

Tamper discipline: no code path in this package raises out of
``verify_payload`` — every failure becomes a ``failed``/``unavailable``
result carrying an ``error_class`` signal. Additionally, any 12+ digit run
in string signals is redacted (Aadhaar numbers never travel in signals;
the payload itself only ever carries the last 4 digits).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import aadhaar_secure, decoder, epic, pan_legacy
from .classifier import classify_payload
from .decoder import decode_image  # re-export

__all__ = ["QrVerifyResult", "verify_payload", "decode_image"]

_UID_RUN_RE = re.compile(r"\d{12,}")


@dataclass(slots=True)
class QrVerifyResult:
    """Structured outcome of one offline QR verification attempt.

    status: 'ok' | 'degraded' | 'unavailable' | 'failed' (same vocabulary
    as capabilities.base.CheckResult). signals stay machine-readable.
    """
    kind: str
    status: str
    signals: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


def _scrub_uid_runs(obj: Any) -> tuple[Any, bool]:
    """Recursively redact 12+ digit runs inside strings. Returns (clean, scrubbed)."""
    if isinstance(obj, str):
        if _UID_RUN_RE.search(obj):
            return _UID_RUN_RE.sub("[REDACTED]", obj), True
        return obj, False
    if isinstance(obj, dict):
        out, scrubbed = {}, False
        for k, v in obj.items():
            out[k], s = _scrub_uid_runs(v)
            scrubbed = scrubbed or s
        return out, scrubbed
    if isinstance(obj, list):
        out, scrubbed = [], False
        for v in obj:
            nv, s = _scrub_uid_runs(v)
            out.append(nv)
            scrubbed = scrubbed or s
        return out, scrubbed
    return obj, False


def verify_payload(payload: str, *, extra_trust_paths: Any = None) -> QrVerifyResult:
    """Classify + verify one decoded QR payload. Never raises.

    extra_trust_paths: optional path (or iterable of paths) to additional
    X.509 certificates or bare public keys (PEM/DER) — used for hermetic
    tests pinning a fixture key; production verification relies solely on
    the bundled UIDAI certificates.
    """
    if extra_trust_paths is None:
        extras: list = []
    elif isinstance(extra_trust_paths, (str, Path)):
        extras = [extra_trust_paths]
    else:
        extras = list(extra_trust_paths)

    kind = classify_payload(payload)
    try:
        if kind == "aadhaar_secure":
            status, signals, detail = aadhaar_secure.verify(payload, extras)
        elif kind == "epic_b64":
            status, signals, detail = epic.verify(payload, extras)
        elif kind == "pan_text":
            status, signals, detail = pan_legacy.verify(payload, extras)
        else:
            return QrVerifyResult(
                kind="unknown", status="unavailable",
                signals={"classified_as": "unknown"},
                detail="payload type not recognised; no offline verification "
                       "strategy available")
    except Exception as exc:  # belt & braces: handlers trap their own errors
        status, signals, detail = ("failed",
                                   {"error_class": type(exc).__name__},
                                   f"unexpected {type(exc).__name__} during verification")
    signals, scrubbed = _scrub_uid_runs(signals)
    if scrubbed:
        signals["uid_leak_scrubbed"] = True
    return QrVerifyResult(kind=kind, status=status, signals=signals, detail=detail)
