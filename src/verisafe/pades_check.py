"""PAdES / CMS-PKCS#7 signature verification (non-detached & detached).

Pure-function verifier with two gated third-party parts (both guarded by the
``pades`` dep id in ``app.detect_available_deps``):

* ``asn1crypto.cms.ContentInfo`` — structural parse of the CMS envelope
  (SignedData -> SignerInfos -> EncapsulatedContentInfo / certificates).
* ``cryptography`` — RSA(PKCS#1 v1.5) / ECDSA verification of the SignerInfo
  signature over the DER-encoded SignedAttributes.

Ground truth (verified against ``openssl cms -verify`` and an independent
manual RSA decode of the SignerInfo signature, fixtures under
``.delegation/fixtures``): the ToBeSigned is the SignedAttributes *as a SET*
— take the wire ``[0] IMPLICIT`` node (tag 0xa0) and flip only the leading
tag byte to 0x31, keeping the length field and all member encodings
verbatim (RFC 5652 §9.1.2.1.2). A good blob verifies; a flipped payload
byte flips ``digest_ok`` to False; a flipped signature byte flips
``rsa_ok`` to False. Matches the openssl oracle on all three fixtures.

Contract: nothing here raises. Malformed input yields ``error`` strings in
the returned dict; callers (``gov_document``) fold that into tri-state
``valid`` (True / False / None) evidence consumed by fusion.
"""
from __future__ import annotations

import base64
import hashlib
import logging

log = logging.getLogger("verisafe.pades")

_OID_TO_NAME = {
    "1.3.14.3": "md5",
    "1.3.14.3.2.26": "sha1",
    "2.16.840.1.101.3.4.2.1": "sha256",
    "2.16.840.1.101.3.4.2.2": "sha384",
}


# ---------------------------------------------------------------- helpers --
def _as_map(v) -> dict:
    """Normalize a parsed ASN.1 value (object with .native, or dict) to dict."""
    if v is None:
        return {}
    n = getattr(v, "native", v)
    return n if isinstance(n, dict) else {}


def _octets(v) -> bytes | None:
    """Bytes out of an OctetString-like node across upstream/fork builds."""
    if v is None:
        return None
    for attr in ("content", "contents"):
        b = getattr(v, attr, None)
        if b is not None:
            return bytes(b)
    try:
        return bytes(v)
    except Exception:
        return None


def _unwrap_constructed_wrappers(d: bytes) -> bytes:
    """Strip [0]-style context wrappers to the first definite-length child.

    The local trimmed asn1crypto build wraps CMSAttributes in ``[0]`` (0xa0)
    and even renders the inner SET as SEQUENCE (0x30); upstream yields plain
    0x31. Unwrap whatever decoration precedes the real attributes body so the
    retag below is deterministic either way.
    """
    while len(d) > 2 and (d[0] & 0xC0) == 0x80:      # context-class, any tag
        l = d[1]
        hdr = 2 if l < 0x80 else 2 + (l & 0x7F)
        if hdr >= len(d):
            break
        d = d[hdr:]
    return d


def _seq_form_of_signed_attrs(sa) -> bytes | None:
    """SignedAttributes re-typed to SEQUENCE, ready to hash (PKCS#7 rule).

    Minimal, empirically-proven recipe against the local builds:
    * CMS wire form: ``[0] IMPLICIT SignedAttributes`` (tag 0xa0) — retag the
      OUTER node's tag byte to 0x30, keeping its own length field and all
      member encodings verbatim (fork dumps reproduce original wire bytes).
    * Upstream asn1crypto: a genuine ``SET OF`` (0x31) — same single-byte
      retag applies.
    Anything else is unreadable -> None (unverifiable, never fatal).
    """
    d = bytes(sa.dump())
    if not d:
        return None
    t = d[0]
    if (t & 0xC0) == 0x80:          # [0] implicit — canonical CMS shape
        return bytes([0x31]) + d[1:]          # retag [0] -> SET (the signed form)
    if t == 0x31:                   # already a SET (upstream shape)
        return d
    if t == 0x30:                   # some forks normalize to SEQUENCE
        return bytes([0x31]) + d[1:]          # retag to SET
    return None


def _pem_to_der(pem: bytes) -> bytes:
    b64 = "".join(
        ln.decode("ascii", "ignore") for ln in pem.splitlines()
        if not ln.strip().startswith(b"-----")
    ).strip()
    return base64.b64decode(b64)


def _normalize_cert(raw: bytes) -> bytes | None:
    """PEM or DER certificate bytes -> DER bytes, or None when unparsable."""
    if raw is None:
        return None
    if b"BEGIN CERTIFICATE" in raw:
        try:
            return _pem_to_der(raw)
        except Exception:
            return None
    if len(raw) >= 2 and raw[0] == 0x30:
        return bytes(raw)
    return None


def _cert_key(cert_der: bytes) -> tuple[bytes, int] | None:
    """(subject_name_der, serial) for anchoring identity, or None."""
    c = _load_cert_any(_normalize_cert(cert_der) if cert_der is not None else None)
    if c is None:
        return None
    return (c.subject.public_bytes(), c.serial_number)


def _load_cert_any(raw):
    """Load a PEM-or-DER certificate via cryptography; None on failure."""
    der = _normalize_cert(raw)
    if der is None:
        return None
    try:
        from cryptography import x509
        return x509.load_der_x509_certificate(der)
    except Exception:
        return None


# ------------------------------------------------------------- trust store --
def load_trust_store(path) -> list[bytes]:
    """Load CA trust anchors (PEM or DER, one or many per file) from *path*.

    Tolerant by contract: unreadable or malformed entries are skipped with a
    log line, never fatal. Non-certificate files (.gitkeep, README…) are
    ignored. Returns normalized DER certificate bytes in stable order.
    """
    from pathlib import Path
    p = Path(path)
    roots: list[bytes] = []
    if not p.exists():
        return roots
    files = sorted(p.iterdir()) if p.is_dir() else ([p] if p.is_file() else [])
    for f in files:
        if not f.is_file() or f.suffix.lower() not in (".cer", ".crt", ".pem", ".der"):
            continue
        try:
            raw = f.read_bytes()
        except OSError as e:
            log.warning("truststore: cannot read %s: %s", f.name, e)
            continue
        chunks = []
        if b"BEGIN CERTIFICATE" in raw:
            for part in raw.split(b"END CERTIFICATE"):
                if b"BEGIN CERTIFICATE" in part:
                    chunks.append(part.strip() + b" END CERTIFICATE\n")
        else:
            chunks.append(raw)
        for ch in chunks:
            der = _normalize_cert(ch)
            if der is None or _cert_key(der) is None:
                log.warning("truststore: skipping malformed entry in %s", f.name)
                continue
            roots.append(der)
    return roots


# ----------------------------------------------------------- envelope bits --
def _extract_embedded_content(sd) -> bytes | None:
    """EncapsulatedContentInfo.content bytes; None for detached CMS."""
    nat = sd.native
    if "encap_content_info" not in nat or nat["encap_content_info"] is None:
        return None
    eoc_nat = _as_map(nat["encap_content_info"])
    cont = eoc_nat.get("content")
    if cont is None:
        return None
    if isinstance(cont, (bytes, bytearray, memoryview)):
        return bytes(cont)
    d = cont.dump() if hasattr(cont, "dump") else bytes(cont)
    try:
        from asn1crypto.core import Any
        return bytes(Any.load(d)["content"])
    except Exception:
        return d


def _message_digest_from_attrs(sa_native) -> bytes | None:
    """MessageDigest attribute value from the parsed signed-attributes."""
    if sa_native is None:
        return None
    items = list(sa_native) if isinstance(sa_native, (list, tuple)) \
        else (list(sa_native.values()) if isinstance(sa_native, dict) else [sa_native])
    for item in items:
        if not isinstance(item, dict):
            continue
        t = str(item.get("type", "")).lower()
        if t not in ("message_digest", "messagedigest"):
            continue
        vals = item.get("values")
        if not vals:
            return None
        v = vals[-1] if isinstance(vals, list) else vals
        if isinstance(v, (bytes, bytearray, memoryview)):
            return bytes(v)
        if isinstance(v, str):
            try:
                return bytes.fromhex(v)
            except ValueError:
                return None
    return None


def _digest_algo_name(da_native: dict, default: str = "sha256") -> str:
    alg = str(da_native.get("algorithm") or default).lower()
    return _OID_TO_NAME.get(alg, alg)


def _sid_parts(sid_native) -> tuple[dict | None, int | None]:
    """Best-effort (issuer_dict, serial) from either fork naming scheme."""
    if not isinstance(sid_native, dict):
        return None, None
    ias = sid_native.get("issuer_and_serial_number")
    if isinstance(ias, dict):
        return ias.get("issuer"), ias.get("serial_number")
    return sid_native.get("issuer"), sid_native.get("serial_number")


def _issuer_rdn_match(iss_dict: dict, cert) -> bool | None:
    """Compare sid.issuer RDNs (fork snake_case keys) vs cert.issuer DN."""
    from cryptography import x509
    want = {str(k): v for k, v in iss_dict.items() if v is not None}
    if not want:
        return True
    oid_by_key = {
        "country_name": x509.NameOID.COUNTRY_NAME,
        "organization_name": x509.NameOID.ORGANIZATION_NAME,
        "organizational_unit_name": x509.NameOID.ORGANIZATIONAL_UNIT_NAME,
        "common_name": x509.NameOID.COMMON_NAME,
    }
    have: dict = {}
    for rd in cert.issuer.rdns:
        attrs = list(rd) if type(rd).__name__ == "RelativeDistinguishedName" else [rd]
        for a in attrs:
            oid = getattr(a, "oid", None)
            oid_str = getattr(oid, "dotted_string", None) or \
                str(getattr(oid, "_id", "")) if oid is not None else None
            if oid_str is None or not str(oid_str):
                continue                 # exotic attr shape without a resolvable OID
            have.setdefault(str(oid_str), set()).add(getattr(a, "value", a))
    for k, v in want.items():
        target = oid_by_key.get(k)
        if target is None:
            return None                     # unknown attribute — indeterminate
        bucket = have.get(target.dotted_string)
        if bucket is None or v not in bucket:
            return False
    return True


def _pick_signer_cert(certs_box, sid_native) -> bytes | None:
    """Certificate referenced by the SignerInfo SID, else the first embedded."""
    if certs_box is None:
        return None
    clist = list(certs_box) if type(certs_box) not in (list, tuple) else certs_box
    if not clist:
        return None
    der_list = [ch.dump() for ch in clist]
    iss, ser = _sid_parts(sid_native)
    if iss is not None or ser is not None:
        for der in der_list:
            key = _cert_key(der)
            if key is None:
                continue
            _, cert_serial = key
            if ser is not None and cert_serial != ser:
                continue
            if iss is not None:
                cert = _load_cert_any(der)
                if cert is None:
                    continue
                m = _issuer_rdn_match(iss, cert)
                if m is False:
                    continue               # issuer mismatch — wrong cert
            return der
    return der_list[0]


_CHAIN_RANK = {"untrusted": 3, "incomplete": 2, "trusted": 1}


def _assess_chain(signer_cert, certs_box, trusted_roots: list[bytes]) -> str | None:
    """Classify the chain for one signer cert (tri-state, spec-mapped).

    ``trusted``    — signer (subject + serial) matches a trust-store anchor.
    ``incomplete`` — chain looks intact (self-signed, or its issuer DN is
                     present in the provided material) but no store anchor
                     matched: the *designed* partial-store outcome, never an
                     accusation.
    ``untrusted``  — the issuer cannot be found anywhere in the available
                     material (wrong-anchor case only).
    ``None``       — certificate unreadable; unverifiable.
    """
    signer = _load_cert_any(signer_cert)
    if signer is None:
        return None
    anchor = (signer.subject.public_bytes(), signer.serial_number)
    for root in trusted_roots:
        rk = _cert_key(root)
        if rk is not None and rk == anchor:
            return "trusted"
    issuer_dn = signer.issuer.public_bytes()
    if issuer_dn == anchor[0]:                 # self-signed (incl. test CAs)
        return "incomplete"
    candidates: list[bytes] = []
    box = list(certs_box) if certs_box is not None and \
        type(certs_box) not in (list, tuple) else (certs_box or [])
    for ch in box:
        c2 = _load_cert_any(ch.dump())
        if c2 is not None:
            candidates.append(c2.subject.public_bytes())
    for root in trusted_roots:
        rc = _load_cert_any(root)
        if rc is not None:
            candidates.append(rc.subject.public_bytes())
    return "incomplete" if issuer_dn in candidates else "untrusted"


# ------------------------------------------------------------------ verify --
def verify_cms(
    cms_bytes: bytes,
    trusted_roots: list[bytes],
    content_bytes: bytes | None = None,
) -> dict:
    """Verify a CMS/PKCS#7 SignedData blob. Never raises.

    Returns ``{"digest_ok", "rsa_ok", "chain", "signer_cn", "error"}`` where
    ``digest_ok`` / ``rsa_ok`` / ``chain`` are tri-state (bool | None, str |
    None) and ``error`` carries a short parse/structure-failure reason when
    the blob could not be understood at all.
    """
    out: dict = {"digest_ok": None, "rsa_ok": None, "chain": None,
                 "signer_cn": None, "error": None}
    notes: list[str] = []
    try:
        from asn1crypto.cms import ContentInfo
        ci = ContentInfo.load(bytes(cms_bytes))
    except Exception as e:  # noqa: BLE001 — graceful-degradation contract
        out["error"] = f"cms-parse: {e.__class__.__name__}"
        return out

    try:
        sd = ci["content"]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"cms-content: {e.__class__.__name__}"
        return out
    if type(sd).__name__ != "SignedData":
        out["error"] = f"unexpected content type: {type(sd).__name__}"
        return out

    try:
        sig_infos = list(sd["signer_infos"])
        certs_box = sd["certificates"]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"cms-structure: {e.__class__.__name__}"
        return out

    for si in sig_infos:
        si_nat = _as_map(si)
        sid_native = si_nat.get("sid")
        sa = si["signed_attrs"]
        # Fork build exposes .native as a LIST of attribute OrderedDicts;
        # upstream builds give a dict keyed by attr name. Handle both.
        sa_native_raw = getattr(sa, "native", None) if sa is not None else None
        seq_form = _seq_form_of_signed_attrs(sa)
        hname = _digest_algo_name(_as_map(si_nat.get("digest_algorithm")))

        # ---- digest over payload -----------------------------------------
        embedded = _extract_embedded_content(sd)
        payload = embedded if embedded is not None else content_bytes
        md_stored = _message_digest_from_attrs(sa_native_raw)
        h_fn = getattr(hashlib, hname, None)
        if payload is not None and md_stored is not None and h_fn is not None:
            if h_fn(payload).digest() == bytes(md_stored):
                out["digest_ok"] = (out["digest_ok"] if out["digest_ok"] is not None
                                    else True)
            else:
                out["digest_ok"] = False
        elif payload is None:
            out["digest_ok"] = None            # detached without provided content
        elif md_stored is None or h_fn is None:
            notes.append("digest-unverifiable")

        # ---- signature over SignedAttributes ------------------------------
        cert_der = _pick_signer_cert(certs_box, sid_native)
        rsa_res: bool | None = None
        if seq_form is not None and cert_der is not None:
            cert = _load_cert_any(cert_der)
            pub = cert.public_key() if cert is not None else None
            if pub is not None:
                from cryptography.hazmat.primitives.asymmetric import ec as _ec
                from cryptography.hazmat.primitives.asymmetric import padding as _pad
                from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
                from cryptography.hazmat.primitives import hashes as _ch
                from cryptography.exceptions import InvalidSignature
                hc = {"sha256": _ch.SHA256, "sha384": _ch.SHA384}.get(hname, _ch.SHA256)()
                sig = _octets(si["signature"]) or b""
                try:
                    if isinstance(pub, _ec.EllipticCurvePublicKey):
                        pub.verify(sig, seq_form, _ec.ECDSA(hc))
                    elif isinstance(pub, _rsa.RSAPublicKey):
                        pub.verify(sig, seq_form, _pad.PKCS1v15(), hc)
                    else:
                        raise TypeError(
                            f"unsupported CMS signature key type: {type(pub).__name__}")
                    rsa_res = True
                except InvalidSignature:
                    rsa_res = False
                except Exception:
                    rsa_res = None            # unsupported key etc. — unverifiable
                if cert is not None:
                    cn_attr = None
                    try:
                        from cryptography.x509 import NameOID
                        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
                        cn_attr = cn[0].value if cn else None
                    except Exception:
                        pass
                    out["signer_cn"] = cn_attr
        if rsa_res is False:
            out["rsa_ok"] = False
        elif rsa_res is True:
            out["rsa_ok"] = (out["rsa_ok"] if out["rsa_ok"] is not None else True)
        else:
            out["rsa_ok"] = (out["rsa_ok"] if out["rsa_ok"] is not None else None)
            if seq_form is None:
                notes.append("attrs-encoding-unreadable")
            else:
                notes.append("rsa-unverifiable-no-certificate")

        # ---- chain anchoring ----------------------------------------------
        cur = _assess_chain(cert_der, certs_box, trusted_roots)
        if cur is not None:
            prev = out["chain"]
            out["chain"] = cur if (prev is None or _CHAIN_RANK[cur] > _CHAIN_RANK.get(prev, 0)) \
                else prev

    if out["chain"] is None and trusted_roots and sig_infos:
        # store exists but no anchor matched anything verifiable -> incomplete
        out["chain"] = "incomplete"
    if notes:
        out["error"] = "; ".join(dict.fromkeys(notes))
    return out


def pdf_signature_contents(pdf_bytes: bytes) -> list[bytes]:
    """Extract every /Sig object's decoded /Contents value from raw PDF bytes.

    Byte-pattern based (no PDF-library dependency; mirrors the indicator-style
    scanning already used in ``gov_document``): locates each ``/Type /Sig``
    dict, then within a bounded window decodes the first successfully-parsed
    ``/Contents <hex>`` or ``/Contents (...)`` literal string. Compressed
    object streams (FlateDecode) are invisible to this scanner — callers must
    treat an empty result as *not-yet-verifiable*, never as *unsigned*.
    """
    out: list[bytes] = []
    pos = 0
    needle = b"/Type /Sig"
    while True:
        i = pdf_bytes.find(needle, pos)
        if i < 0:
            break
        window = pdf_bytes[i:i + 8192]
        extracted: bytes | None = None
        cp = window.find(b"/Contents")
        while cp != -1:
            tail = window[cp + len(b"/Contents"):].lstrip()
            if tail.startswith(b"<"):
                chunk = tail[1:]
                end = chunk.find(b">")
                if end > 0:
                    hx = chunk[:end].replace(b" ", b"").replace(
                        b"\n", b"").replace(b"\r", b"")
                    if len(hx) % 2 == 0 and all(
                            c in b"0123456789abcdefABCDEF" for c in hx):
                        try:
                            extracted = bytes.fromhex(hx.decode("ascii"))
                        except ValueError:
                            extracted = None
            elif tail.startswith(b"("):
                depth, j, raw_tok = 0, 0, bytearray()
                lit = tail
                while j < len(lit) and len(raw_tok) < 65536:
                    ch = lit[j:j + 1]
                    if ch == b"\\":
                        raw_tok += lit[j:j + 2]
                        j += 2
                        continue
                    if ch == b"(":
                        depth += 1
                    elif ch == b")":
                        depth -= 1
                    raw_tok += ch
                    j += 1
                    if depth == 0 and j > 1:
                        break
                tok = bytes(raw_tok).lstrip(b"(").rstrip()
                if tok.endswith(b")"):
                    tok = tok[:-1]
                try:
                    extracted = base64.b64decode(tok, validate=False)
                except Exception:
                    extracted = None
            if extracted is not None:
                break
            cp = window.find(b"/Contents", cp + 1)
        if extracted is not None:
            out.append(extracted)
        pos = i + 1
    return out
