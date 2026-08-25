"""EPIC (Voter ID) Secure QR: static-key AES-256-CBC blob, base64-wrapped.

Spec per github.com/capt n3m0/epicqr (MIT) — port of method, not copy of
code; also documented in the repo's docs/research QR scouting report.
Envelope::

    51-char compact JSON {"epic_no": ..., "unique_generated_id": <int>}
    -> PKCS7 pad -> AES-256-CBC(KEY=b'X_4k$uq23FSwI.qT', IV=b'H76$suq23_po(8sD')
    -> base64

The KEY/IV are hard-coded in the Election Commission's official Android app;
they are obfuscation, not a secret, and confer NO authenticity — anyone can
encrypt arbitrary EPIC-looking JSON. Signals therefore describe structure
only; callers must not treat a successful decrypt as issuer verification.
EPIC numbers are public-record electoral identifiers, safe to surface.
"""
from __future__ import annotations

import base64
import json
import re

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

__all__ = ["decrypt_payload", "verify"]

_KEY = b"X_4k$uq23FSwI.qT"
_IV = b"H76$suq23_po(8sD"
_EPIC_NO_RE = re.compile(r"^[A-Z]{1,3}[0-9]{6,8}$")


def decrypt_payload(b64: str) -> dict:
    """base64 blob -> decrypted JSON dict. Raises on any structural fault."""
    ct = base64.b64decode((b64 or "").strip(), validate=True)
    if not ct or len(ct) % 16:
        raise ValueError("ciphertext length is not a positive AES block multiple")
    dec = Cipher(algorithms.AES(_KEY), modes.CBC(_IV)).decryptor()
    padded = dec.update(ct) + dec.finalize()
    pad = padded[-1]
    if not 1 <= pad <= 16 or padded[-pad:] != bytes([pad]) * pad:
        raise ValueError("invalid PKCS7 padding")
    return json.loads(padded[:-pad].decode("utf-8"))


def verify(payload: str, extra_trust_paths=None) -> tuple[str, dict, str]:
    """Decrypt + structure-check one EPIC blob. Returns (status, signals, detail)."""
    try:
        obj = decrypt_payload(payload)
    except Exception as exc:
        return ("failed",
                {"epic_no": "", "structure_ok": False,
                 "error_class": type(exc).__name__},
                "EPIC blob did not decrypt/parse to the expected JSON structure")
    if not isinstance(obj, dict):
        return ("failed",
                {"epic_no": "", "structure_ok": False, "error_class": "structure_mismatch"},
                "decrypted payload is not a JSON object")
    epic_no = str(obj.get("epic_no", ""))
    has_ugid = "unique_generated_id" in obj
    structure_ok = bool(_EPIC_NO_RE.fullmatch(epic_no)) and has_ugid
    signals = {"epic_no": epic_no, "structure_ok": structure_ok}
    if structure_ok:
        return ("ok", signals,
                "EPIC decrypted; static-key AES envelope intact "
                "(confidentiality only — this scheme does not authenticate the issuer)")
    signals["error_class"] = "structure_mismatch"
    return ("degraded", signals,
            "decrypted but fields do not match the epicqr schema")
