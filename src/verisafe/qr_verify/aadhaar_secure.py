"""Aadhaar Secure QR (offline paper QR): parse + RSA-SHA256 signature verify.

Envelope format reverse-engineered per the repo's docs/research QR scouting
report; the verification method is ported from
github.com/StarkAg/aadhaar-secure-qr-verifier (MIT) — port of method, not
copy of code. Structure::

    decimal string -> int -> big-endian bytes -> gzip.decompress
        -> [body][last 256 bytes = RSA-2048 PKCS1v15 SHA-256 signature over body]
    body = 0xFF-delimited latin-1/utf-8 segments:
        segs[0] = version (e.g. '20260216')
        segs[1] = hashed email/mobile presence flag (bitmask: bit0 email,
                  bit1 mobile; '0' = neither present)
        segs[2:] = demographic fields in FIELD_NAMES order

Signatures are checked against every certificate in
``src/verisafe/assets/uidai_certs/`` (loaded once per process, cached);
``extra_trust_paths`` accepts additional X.509 certs or bare public keys
(PEM or DER) so hermetic tests can pin a fixture key without touching the
production trust store.

PRIVACY INVARIANT: the Secure-QR payload never carries the full Aadhaar
number — at most the last 4 digits. This module enforces that structurally:
``aadhaar_last4`` is truncated to 4 characters and the package-level scrubber
redacts any 12+ digit run before signals can leave ``verify_payload``.
"""
from __future__ import annotations

import gzip
from pathlib import Path

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

__all__ = ["FIELD_NAMES", "parse_payload", "verify", "CERT_DIR"]

CERT_DIR = Path(__file__).resolve().parents[1] / "assets" / "uidai_certs"

# Exact demographic field order of the Secure-QR body (scouting report).
FIELD_NAMES = (
    "reference_id", "name", "dob", "gender", "care_of", "district",
    "landmark", "house", "location", "pincode", "post_office", "state",
    "street", "sub_district", "vtc", "aadhaar_last4",
)

_ANCHOR_CACHE: dict[tuple, object] = {}


def _load_anchor(path: Path):
    """Load one trust anchor: X.509 cert (PEM or DER) or bare public key."""
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        return cached
    data = path.read_bytes()
    pub = None
    for loader in (x509.load_pem_x509_certificate, x509.load_der_x509_certificate):
        try:
            pub = loader(data).public_key()
            break
        except Exception:
            continue
    if pub is None:
        for loader in (serialization.load_der_public_key,
                       serialization.load_pem_public_key):
            try:
                pub = loader(data)
                break
            except Exception:
                continue
    if pub is None:
        raise ValueError(f"unusable trust anchor: {path.name}")
    _ANCHOR_CACHE[key] = pub
    return pub


def collect_anchors(extra_trust_paths=None) -> list[tuple[str, object]]:
    """Bundled UIDAI certs first (sorted), then caller-supplied extras."""
    anchors: list[tuple[str, object]] = []
    seen: set[bytes] = set()
    bundled: list[Path] = []
    for pattern in ("*.cer", "*.pem"):
        bundled.extend(sorted(CERT_DIR.glob(pattern)))
    for path in bundled:
        try:
            pub = _load_anchor(path)
        except Exception:
            continue
        spki = pub.public_bytes(serialization.Encoding.DER,
                                serialization.PublicFormat.SubjectPublicKeyInfo)
        if spki in seen:
            continue
        seen.add(spki)
        anchors.append((path.name, pub))
    for raw in extra_trust_paths or []:
        try:
            pub = _load_anchor(Path(raw))
        except Exception:
            continue
        spki = pub.public_bytes(serialization.Encoding.DER,
                                serialization.PublicFormat.SubjectPublicKeyInfo)
        if spki in seen:
            continue
        seen.add(spki)
        anchors.append((Path(raw).name, pub))
    return anchors


def parse_payload(numeric: str):
    """numeric string -> (signed_body, signature, fields dict, presence flag).

    Raises ValueError/gzip.BadGzipFile on any structural mismatch — callers
    translate exceptions into failed results, never propagate them.
    """
    s = (numeric or "").strip()
    if not s.isdigit():
        raise ValueError("payload is not an all-digit Secure-QR numeric string")
    if len(s) <= 500:
        raise ValueError("payload too short for a Secure-QR envelope")
    n = int(s)
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    data = gzip.decompress(raw)  # raises on non-gzip
    if len(data) <= 256:
        raise ValueError("decompressed envelope too small to carry body + signature")
    signed, signature = data[:-256], data[-256:]
    segs = signed.split(b"\xff")
    if len(segs) < 3:
        raise ValueError("body has too few 0xFF-delimited segments")
    version = segs[0].decode("latin-1", "replace")
    presence = segs[1].decode("latin-1", "replace")
    vals = []
    for seg in segs[2:2 + len(FIELD_NAMES)]:
        try:
            vals.append(seg.decode("utf-8"))
        except UnicodeDecodeError:
            vals.append(seg.decode("latin-1", "replace"))
    fields = {"version": version}
    fields.update(dict(zip(FIELD_NAMES, vals)))  # absent trailing fields stay absent
    return signed, signature, fields, presence


def verify(payload: str, extra_trust_paths=None) -> tuple[str, dict, str]:
    """Parse + verify one Secure-QR payload. Returns (status, signals, detail).

    status: 'ok' (signature valid) | 'failed' (parse or signature mismatch)
    | 'unavailable' (no usable trust anchor). Never raises.
    """
    signals: dict = {
        "signature_valid": False, "signer_cert": "", "version": "",
        "name_present": False, "dob": "", "gender": "",
        "aadhaar_last4": "", "pincode": "", "email_mobile_hashed": False,
    }
    try:
        signed, signature, fields, presence = parse_payload(payload)
    except Exception as exc:
        signals["error_class"] = type(exc).__name__
        return ("failed", signals,
                f"Secure-QR payload unparseable ({type(exc).__name__}); "
                "no demographic signals extracted")

    signals["version"] = fields.get("version", "")
    signals["name_present"] = bool(fields.get("name", "").strip())
    signals["dob"] = fields.get("dob", "")
    signals["gender"] = fields.get("gender", "")
    # Privacy invariant: keep at most the trailing 4 digits.
    signals["aadhaar_last4"] = fields.get("aadhaar_last4", "")[-4:]
    signals["pincode"] = fields.get("pincode", "")
    try:
        signals["email_mobile_hashed"] = bool(int(presence.strip() or "0") & 0b11)
    except ValueError:
        signals["email_mobile_hashed"] = presence.strip() not in ("", "0")

    try:
        anchors = collect_anchors(extra_trust_paths)
    except Exception:
        anchors = []
    if not anchors:
        signals["error_class"] = "no_trust_anchors"
        return ("unavailable", signals,
                "no usable trust anchor found; signature not evaluated")

    for label, pub in anchors:
        try:
            pub.verify(signature, signed, padding.PKCS1v15(), hashes.SHA256())
        except InvalidSignature:
            continue
        except Exception as exc:
            signals.setdefault("anchor_error_class", type(exc).__name__)
            continue
        signals["signature_valid"] = True
        signals["signer_cert"] = label
        return ("ok", signals,
                f"Secure-QR signature verified against trust anchor '{label}'")

    signals["error_class"] = "signature_mismatch"
    return ("failed", signals,
            f"signature over body did not verify against any of "
            f"{len(anchors)} trust anchor(s)")
