"""Legacy (pre-EQR) PAN-card QR: plain multi-line text, no crypto at all.

Old PAN cards printed a text-block QR (NAME / FATHER / DOB / PAN lines).
Nothing here is authenticated — extraction only, honestly labelled as such
in the detail string. PANs are public-format tax identifiers (not secret
like Aadhaar), so the PAN itself may appear in signals; the format-valid
flag is reported alongside it.
"""
from __future__ import annotations

import re

from .classifier import PAN_RE

__all__ = ["extract_pan", "verify"]

_NAME_LINE_RE = re.compile(r"(?im)^\s*(?:full\s+)?name\b\s*[:\-]\s*(.+?)\s*$")
_DOB_RE = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")


def extract_pan(text: str) -> str:
    """First PAN-pattern token in the text, or ''."""
    m = PAN_RE.search(text or "")
    return m.group(0) if m else ""


def verify(payload: str, extra_trust_paths=None) -> tuple[str, dict, str]:
    """Extract PAN + name-line heuristics. Returns (status, signals, detail)."""
    text = payload or ""
    pan = extract_pan(text)
    if not pan:
        return ("failed",
                {"pan": "", "pan_format_valid": False, "name_line": "",
                 "dob_found": False, "error_class": "pan_not_found"},
                "no PAN-pattern token found in text payload")
    nm = _NAME_LINE_RE.search(text)
    name_line = nm.group(1) if nm else ""
    signals = {
        "pan": pan,
        "pan_format_valid": bool(PAN_RE.fullmatch(pan)),
        "name_line": name_line,
        "dob_found": bool(_DOB_RE.search(text)),
    }
    if name_line:
        return ("ok", signals,
                "PAN extracted from legacy plain-text card QR "
                "(unauthenticated: legacy QRs carry no signature)")
    return ("degraded", signals,
            "PAN extracted, but no NAME: line matched the heuristic")
