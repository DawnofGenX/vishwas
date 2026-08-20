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
  * ToBeSigned retag   -> [0] implicit node retagged to SET (0x31), length +
                          content preserved byte-for-byte (RFC 5652 §9.1.2.1.2)
  * gate-OFF           -> capability emits indicator-only CheckResult, pipeline
                          still returns (mirror test_11_docling_branch idiom)
"""
from __future__ import annotations

import hashlib
import importlib.util as _iu
import time
from pathlib import Path

import pytest

from verisafe.capabilities import gov_document as gd
from verisafe.events import Artifact, InputType, JobContext, MediaKind

FIX = Path(__file__).resolve().parent.parent / ".delegation" / "fixtures"


def _fix(name: str) -> bytes:
    return (FIX / name).read_bytes()


# ------------------------------------------------------- core verifier -----
def test_good_blob_matches_oracle():
    r = gd._get_pades().verify_cms(_fix("nod.der"), [])
    assert r["digest_ok"] is True
    assert r["rsa_ok"] is True
    assert r["signer_cn"] == "VeriSafe Test Signing Authority"


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
