"""PAdES/CMS integration tests (brief .delegation/brief_pades.md item 6).

Hermetic, NO network. The four fixtures under .delegation/fixtures/ ARE the
oracle: their pass/fail behavior was independently established with
`openssl cms -verify`, so asserting against them is verification, not
tautology.

Coverage:
  * good blob          -> digest_ok=True,  rsa_ok=True   (+ tri-state valid)
  * tampered EOC       -> digest_ok=False, rsa_ok=True
  * tampered signature -> digest_ok=True,  rsa_ok=False
  * empty trust store  -> chain "incomplete" (designed partial-store outcome)
  * garbage CMS        -> graceful: error string, no exception escapes
  * anchored-chain     -> runtime-generated CA in tmp dir; signer cert pinned
                          in the store yields chain=="trusted" (and the
                          designed partial-store neighbors stay "incomplete");
                          capability-level end-to-end gives valid==True;
                          committed production anchor smoke (README-pinned)
  * ToBeSigned retag   -> [0] implicit node retagged to SET (0x31), length +
                          content preserved byte-for-byte (RFC 5652 §9.1.2.1.2)
  * gate-OFF           -> capability emits indicator-only CheckResult, pipeline
                          still returns (mirror test_11_docling_branch idiom)
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.util as _iu
import time
from pathlib import Path

import pytest

from vishwas.capabilities import gov_document as gd
from vishwas.events import Artifact, InputType, JobContext, MediaKind

FIX = Path(__file__).resolve().parent.parent / ".delegation" / "fixtures"


def _fix(name: str) -> bytes:
    return (FIX / name).read_bytes()


# ------------------------------------------------------- core verifier -----
def test_good_blob_matches_oracle():
    r = gd._get_pades().verify_cms(_fix("nod.der"), [])
    assert r["digest_ok"] is True
    assert r["rsa_ok"] is True
    assert r["signer_cn"] == "VeriSafe Test Signing Authority"  # historical cert CN inside the fixture blob (binary, not swept by the rename)


def test_tampered_embedded_content_flips_digest_only():
    r = gd._get_pades().verify_cms(_fix("tampered_eoc.der"), [])
    assert r["digest_ok"] is False
    assert r["rsa_ok"] is True          # sig over attrs still intact


def test_tampered_signature_flips_rsa_only():
    r = gd._get_pades().verify_cms(_fix("tampered_sig.der"), [])
    assert r["digest_ok"] is True
    assert r["rsa_ok"] is False         # invalid padding per openssl oracle


def test_truststore_anchor_gives_trusted_chain():
    roots = gd._get_pades().load_trust_store(FIX / "cert.pem")
    assert len(roots) == 1
    r = gd._get_pades().verify_cms(_fix("nod.der"), roots)
    assert r["chain"] == "trusted"


def test_empty_truststore_yields_incomplete_not_untrusted():
    """Partial/empty store is the DESIGNED outcome — never 'untrusted'."""
    r = gd._get_pades().verify_cms(_fix("nod.der"), [])
    assert r["chain"] == "incomplete"


def test_garbage_input_is_graceful():
    mod = gd._get_pades()
    for blob in (b"", b"\x01\x02\x03", b"not a cms envelope at all"):
        r = mod.verify_cms(blob, [])
        assert r["error"] is not None
        assert r["digest_ok"] is None and r["rsa_ok"] is None


# ------------------------------------------- anchored-chain (runtime CA) ----
# Semantics under test (pades_check._assess_chain): chain=="trusted" requires
# the SIGNER certificate itself (subject + serial) to be present as a
# trust-store anchor — the operator vouched for the exact signing identity.
# A store holding only the issuing CA (or nothing) yields the designed
# "incomplete" partial-store outcome, never an accusation. The CA here is
# generated at RUNTIME in a temp dir: no committed private keys, fully
# offline, deterministic serials/dates.

def _mk_cert(subject, issuer_name, pub, sign_key, is_ca, serial, now):
    from cryptography import x509 as _cx
    from cryptography.hazmat.primitives import hashes as _h
    return (_cx.CertificateBuilder()
            .subject_name(subject).issuer_name(issuer_name)
            .public_key(pub).serial_number(serial)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(_cx.BasicConstraints(ca=is_ca, path_length=None),
                           critical=True)
            .sign(sign_key, _h.SHA256()))


def _build_runtime_cms(ca_cert, signer_key, signer_cert) -> bytes:
    """Minimal RFC 5652 SignedData (SHA-256/RSA) signed by *signer_key*.

    Embeds BOTH the signer and issuing-CA certs so chain material is
    complete and the outcome is purely a function of the trust store handed
    to verify_cms. The RSA signature covers the signed-attrs SET (same retag
    rule proven by test_seq_form_retags_context_tag_to_set_preserving_body).
    """
    import asn1crypto.cms as _cms
    import asn1crypto.core as _core
    import asn1crypto.x509 as _ax
    from asn1crypto.algos import DigestAlgorithm as _DA
    from cryptography.hazmat.primitives import hashes as _h
    from cryptography.hazmat.primitives import serialization as _ser
    from cryptography.hazmat.primitives.asymmetric import padding as _pad

    payload = b"%PDF-1.4 anchored-chain runtime payload\n"
    acert = _ax.Certificate.load(
        signer_cert.public_bytes(_ser.Encoding.DER))
    aca = _ax.Certificate.load(ca_cert.public_bytes(_ser.Encoding.DER))
    attrs = _cms.CMSAttributes([
        _cms.CMSAttribute({"type": "content_type",
                           "values": [_cms.ContentType("data")]}),
        _cms.CMSAttribute({"type": "message_digest",
                           "values": [hashlib.sha256(payload).digest()]}),
    ])
    tbs = bytes([0x31]) + attrs.dump()[1:]     # wire form -> SET (RFC 5652)
    sig = signer_key.sign(tbs, _pad.PKCS1v15(), _h.SHA256())
    sd = _cms.SignedData({
        "version": "v1",
        "digest_algorithms": [_DA({"algorithm": "sha256"})],
        "encap_content_info": {"content_type": "data",
                               "content": _core.OctetString(payload)},
        "certificates": [acert, aca],
        "signer_infos": [_cms.SignerInfo({
            "version": "v1",
            "sid": _cms.SignerIdentifier({
                "issuer_and_serial_number": _cms.IssuerAndSerialNumber({
                    "issuer": acert.issuer,
                    "serial_number": acert.serial_number,
                }),
            }),
            "digest_algorithm": _DA({"algorithm": "sha256"}),
            "signed_attrs": attrs,
            "signature_algorithm": {"algorithm": "sha256_rsa"},
            "signature": sig,
        })],
    })
    return _cms.ContentInfo({"content_type": "signed_data",
                             "content": sd}).dump()


@pytest.fixture(scope="module")
def runtime_chain(tmp_path_factory):
    """(store_dir, ca_der, signer_der, cms_blob) from a runtime-generated CA."""
    from cryptography import x509 as _x
    from cryptography.hazmat.primitives import serialization as _ser
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
    from cryptography.x509.oid import NameOID as _NO

    now = datetime.datetime(2026, 8, 21, 12, 0, 0,
                            tzinfo=datetime.timezone.utc)   # fixed/deterministic

    def _name(cn):
        return _x.Name([
            _x.NameAttribute(_NO.COUNTRY_NAME, "IN"),
            _x.NameAttribute(_NO.ORGANIZATION_NAME, "Vishwas Runtime Test CA"),
            _x.NameAttribute(_NO.COMMON_NAME, cn),
        ])

    ca_key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signer_key = _rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = _name("Vishwas Runtime Anchor Root")
    ca_cert = _mk_cert(ca_name, ca_name, ca_key.public_key(), ca_key,
                       True, 0x1001, now)
    signer_cert = _mk_cert(_name("Vishwas Runtime Signer"), ca_name,
                           signer_key.public_key(), ca_key, False, 0x2001, now)
    ca_der = ca_cert.public_bytes(_ser.Encoding.DER)
    signer_der = signer_cert.public_bytes(_ser.Encoding.DER)
    store_dir = tmp_path_factory.mktemp("truststore")
    (store_dir / "ca.der").write_bytes(ca_der)
    (store_dir / "signer.der").write_bytes(signer_der)
    return store_dir, ca_der, signer_der, _build_runtime_cms(ca_cert, signer_key, signer_cert)


def test_runtime_anchored_chain_yields_trusted(runtime_chain):
    """Signer cert pinned in the store -> digest+RSA OK AND chain=='trusted'."""
    mod = gd._get_pades()
    store_dir, ca_der, signer_der, blob = runtime_chain
    roots = mod.load_trust_store(store_dir)      # dir-scan load path
    assert signer_der in roots
    r = mod.verify_cms(blob, roots)
    assert r["digest_ok"] is True
    assert r["rsa_ok"] is True
    assert r["chain"] == "trusted"


def test_runtime_issuing_ca_anchor_alone_does_not_pin_signer(runtime_chain):
    """Only the issuing CA in the store -> still 'incomplete' by design."""
    _, ca_der, _, blob = runtime_chain
    r = gd._get_pades().verify_cms(blob, [ca_der])
    assert r["digest_ok"] is True and r["rsa_ok"] is True
    assert r["chain"] == "incomplete"


def test_runtime_chain_without_store_stays_incomplete(runtime_chain):
    """No store at all -> chain material embedded but nothing vouched."""
    _, _, _, blob = runtime_chain
    r = gd._get_pades().verify_cms(blob, [])
    assert r["chain"] == "incomplete"


def test_capability_end_to_end_anchored_chain_valid_true(runtime_chain,
                                                         tmp_path, monkeypatch):
    """Production entry point: anchored chain -> status ok, valid==True."""
    store_dir, _, _, blob = runtime_chain
    monkeypatch.setattr(gd, "_DEFAULT_TRUSTSTORE", str(store_dir))
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\n/Type /Sig\n/Contents <"
                  + blob.hex().upper().encode() + b">\n%%EOF\n")
    art = Artifact(path=p, original_filename="doc.pdf",
                   declared_type=InputType.FILE, verified_kind=MediaKind.PDF)
    ctx = JobContext(job_id="t", artifact=art, quarantine_root=tmp_path,
                     deadline_mono=time.monotonic() + 300,
                     pades_available=True)
    res = gd.GovDocumentCapability()._digital_signature(art, ctx)
    cr = res[0]
    assert cr.name == "digital_signature"
    assert cr.status == "ok"
    assert cr.signals["chain"] == "trusted"
    assert cr.signals["valid"] is True
    assert cr.signals["digest_ok"] is True and cr.signals["rsa_ok"] is True


def test_committed_anchor_smoke():
    """The seeded production store (assets/ca_truststore) loads and parses.

    Provenance contract with the truststore README: the committed public
    root must be present, be a CA, and carry the documented subject CN.
    """
    from cryptography import x509 as _x
    from cryptography.x509.oid import NameOID as _NO

    roots = gd._get_pades().load_trust_store(gd._DEFAULT_TRUSTSTORE)
    assert len(roots) >= 1
    subjects = []
    for der in roots:
        c = _x.load_der_x509_certificate(der)
        bc = c.extensions.get_extension_for_class(_x.BasicConstraints).value
        assert bc.ca is True
        cn = c.subject.get_attributes_for_oid(_NO.COMMON_NAME)
        subjects.append(cn[0].value if cn else None)
    assert "ISRG Root X1" in subjects


# ------------------------------------------------- ToBeSigned construction --
def test_seq_form_retags_context_tag_to_set_preserving_body():
    """The canonical CMS shape is `[0] IMPLICIT SignedAttributes` (0xa0);
    the signed form flips only that leading byte to 0x31 (SET)."""
    from asn1crypto.cms import ContentInfo

    raw = _fix("nod.der")
    sd = ContentInfo.load(raw)["content"]
    sa = sd["signer_infos"][0]["signed_attrs"]
    wire = sa.dump()
    tbs = gd._get_pades()._seq_form_of_signed_attrs(sa)
    assert tbs is not None
    assert tbs[:1] == b"\x31"               # SET, not SEQUENCE
    assert tbs[1:] == wire[1:]              # length + body verbatim

    # ...and that exact TBS must be the byte string the signer actually signed.
    # Proof: recover the RSA pre-image with the cert's public key, pull the
    # DigestInfo(SHA-256) field out of the PKCS#1 v1.5 block, and compare to
    # sha256(tbs). (Distinct from the messageDigest attribute, which holds
    # sha256(embedded-content); the RSA sig covers the signed attributes.)
    from cryptography import x509
    from cryptography.hazmat.primitives.asymmetric import rsa as _rsa
    cert = x509.load_pem_x509_certificate((FIX / "cert.pem").read_bytes())
    pub = cert.public_key()
    assert isinstance(pub, _rsa.RSAPublicKey)   # fixture is RSA-2048
    pn = pub.public_numbers()
    sig = sd["signer_infos"][0]["signature"]
    sig_bytes = bytes(sig.content) if hasattr(sig, "content") else bytes(sig)
    m = pow(int.from_bytes(sig_bytes, "big"), pn.e, pn.n).to_bytes(256, "big")
    DI = bytes.fromhex("3031300d060960864801650304020105000420")
    di = m.rfind(DI)
    assert di >= 0, "no SHA-256 DigestInfo in RSA pre-image"
    embedded = m[di + len(DI): di + len(DI) + 32]
    assert hashlib.sha256(tbs).digest() == embedded


# ------------------------------------------------------------ capability ----
@pytest.fixture(autouse=True)
def _reset_pades_globals():
    gd._PADES_MOD = None
    gd._PADES_TRIED = False
    yield
    gd._PADES_MOD = None
    gd._PADES_TRIED = False


def _mk_ctx(tmp_path: Path, pades_available: bool):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")
    art = Artifact(path=p, original_filename="doc.pdf",
                   declared_type=InputType.FILE, verified_kind=MediaKind.PDF)
    ctx = JobContext(job_id="t", artifact=art, quarantine_root=tmp_path,
                     deadline_mono=time.monotonic() + 300,
                     pades_available=pades_available)
    return art, ctx


def test_gate_off_emits_indicator_only_and_pipeline_continues(tmp_path, monkeypatch):
    """asn1crypto/cryptography absent -> cheap indicator-only CheckResult,
    no crash, analyze() still completes (mirrors test_11 gate-off idiom)."""
    real_find_spec = _iu.find_spec

    def find_spec_shim(name):
        if name in ("asn1crypto", "cryptography"):
            return None
        return real_find_spec(name)

    monkeypatch.setattr(_iu, "find_spec", find_spec_shim)
    # PDF with a visible signature marker, but the dep gate is off
    p = tmp_path / "s.pdf"
    p.write_bytes(b"%PDF-1.4\n/Type /Sig\n<<>>\n%%EOF\n")
    art = Artifact(path=p, original_filename="s.pdf",
                   declared_type=InputType.FILE, verified_kind=MediaKind.PDF)
    ctx = JobContext(job_id="t", artifact=art, quarantine_root=tmp_path,
                     deadline_mono=time.monotonic() + 300,
                     pades_available=False)

    res = gd.GovDocumentCapability()._digital_signature(art, ctx)
    assert len(res) >= 1
    cr = res[0]
    assert cr.name == "digital_signature"
    assert cr.signals.get("format") == "pdf"
    assert cr.signals.get("has_sig_object") is True
    # gate-off must NOT claim cryptographic validity or failure
    assert "valid" not in cr.signals
    assert cr.status in ("ok", "degraded")
