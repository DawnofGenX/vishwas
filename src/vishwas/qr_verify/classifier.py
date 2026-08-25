"""Payload classification for Indian-identity QR codes (fully offline).

Categories and thresholds come from the docs/research QR scouting report:

- ``aadhaar_secure``  all-digit numeric envelope, len > 500 (UIDAI Secure QR
  encodes gzip+RSA bytes as one huge decimal integer)
- ``epic_b64``        base64 charset, length 60..120 (EPIC voter-ID AES blob)
- ``digilocker_url``  http(s) URL whose host mentions digilocker|udyam
- ``pan_text``        multi-line plain text containing a PAN-pattern token
- ``unknown``         anything else — callers must answer honestly that no
  offline verification strategy exists
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

__all__ = ["classify_payload", "PAN_RE"]

PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_HOST_HINT_RE = re.compile(r"digilocker|udyam", re.IGNORECASE)


def classify_payload(payload: str) -> str:
    """Return one of: aadhaar_secure | epic_b64 | digilocker_url | pan_text | unknown."""
    p = (payload or "").strip()
    if not p:
        return "unknown"
    if p.isdigit() and len(p) > 500:
        return "aadhaar_secure"
    if 60 <= len(p) <= 120 and _B64_RE.match(p):
        return "epic_b64"
    if p.lower().startswith("http"):
        try:
            host = urlparse(p).netloc
        except ValueError:
            host = ""
        if _HOST_HINT_RE.search(host or p):
            return "digilocker_url"
    if "\n" in p and PAN_RE.search(p):
        return "pan_text"
    return "unknown"
