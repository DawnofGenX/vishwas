"""Hermetic tests for the GPG keyring-backed digital-signature check (D2).

All fixtures are generated locally with the REAL system gpg binary inside
throwaway GNUPGHOME dirs under tmp_path — zero network. Skips cleanly if the
gpg binary is missing (it is present on this box, so most cases run).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from vishwas import gpg_check
from vishwas.capabilities import gov_document as gd
from vishwas.events import Artifact, InputType, JobContext, MediaKind

pytestmark = pytest.mark.skipif(
    not gpg_check.available(), reason="gpg binary not available")


# ------------------------------------------------------------------ helpers
def _gpg_bin() -> str:
    return subprocess.run(["which", "gpg"], capture_output=True, text=True).stdout.strip() \
        or "gpg"


def _run(home: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["gpg", "--homedir", str(home), "--batch", "--no-tty",
                           "--pinentry-mode", "loopback", *args],
                          capture_output=True, text=True, timeout=timeout)


def _gen_key(home: Path, name: str, email: str) -> str:
    """Generate a throwaway RSA key; return its primary fingerprint."""
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)  # gpg warns on unsafe perms but still works; avoid noise
    params = home / "_params"
    params.write_text(
        "Key-Type: RSA\nKey-Length: 1024\nSubkey-Type: RSA\nSubkey-Length: 1024\n"
        f"Name-Real: {name}\nName-Email: {email}\nExpire-Date: 0\n"
        "Passphrase: vishwastest\n%commit\n")
    p = subprocess.run(["gpg", "--homedir", str(home), "--batch", "--no-tty",
                        "--pinentry-mode", "loopback", "--gen-key", str(params)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0 or "created" in p.stderr.lower(), p.stderr
    out = subprocess.run(["gpg", "--homedir", str(home), "--with-colons", "--list-keys"],
                        capture_output=True, text=True, timeout=60)
    for ln in out.stdout.splitlines():
        cols = ln.split(":")
        if len(cols) >= 10 and cols[0] == "fpr":
            return cols[9]
    raise AssertionError("no fingerprint after keygen")


def _export_pub(home: Path, fpr: str) -> bytes:
    p = _run(home, "--armor", "--export", fpr)
    assert p.returncode == 0, p.stderr
    return p.stdout.encode()


def _sign(home: Path, data: bytes, out_sig: Path) -> None:
    datafile = home / "_data.bin"
    datafile.write_bytes(data)
    p = subprocess.run(["gpg", "--homedir", str(home), "--batch", "--yes",
                        "--no-tty", "--pinentry-mode", "loopback", "--passphrase",
                        "vishwastest", "--detach-sign", "--output", str(out_sig),
                        str(datafile)],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    datafile.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def keys():
    """Two throwaway keys in separate homes; returns dict with fprs + pub blobs."""
    import tempfile
    base = Path(tempfile.mkdtemp(prefix="vishwas-gpg-test-"))
    h1 = base / "home1"
    h2 = base / "home2"
    fpr1 = _gen_key(h1, "Alice Signer", "alice@example.org")
    fpr2 = _gen_key(h2, "Bob Signer", "bob@example.org")
    yield {
        "home1": h1, "home2": h2,
        "fpr1": fpr1, "fpr2": fpr2,
        "pub1": _export_pub(h1, fpr1),
        "pub2": _export_pub(h2, fpr2),
    }


def _ctx(tmp_path: Path, art: Artifact) -> JobContext:
    return JobContext(job_id="t", artifact=art, quarantine_root=tmp_path,
                      deadline_mono=time.monotonic() + 300)


def _art(tmp_path: Path, data: bytes, name="doc.txt") -> Artifact:
    p = tmp_path / name
    p.write_bytes(data)
    return Artifact(path=p, original_filename=name, declared_type=InputType.FILE,
                    verified_kind=MediaKind.UNKNOWN)


# ------------------------------------------------------------------- a ----
def test_a_valid_trusted_signature(keys, tmp_path):
    home = keys["home1"]
    data = b"the quick brown fox jumps over the lazy dog"
    sig = tmp_path / "doc.sig"
    _sign(home, data, sig)

    # truststore contains Alice's public key
    ts = tmp_path / "truststore"
    ts.mkdir()
    (ts / "alice.asc").write_bytes(keys["pub1"])

    v = gpg_check.verify_with_truststore(data, sig, truststore_dir=ts,
                                         workdir=tmp_path / "work")
    assert v.valid is True
    assert v.fingerprint == keys["fpr1"]
    assert v.trusted is True


# ------------------------------------------------------------------- b ----
def _pubdir(tmp_path: Path, name: str, pubbytes: bytes) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "key.asc").write_bytes(pubbytes)
    return d


def test_b_valid_but_untrusted(keys, tmp_path):
    home = keys["home1"]
    content = (b"-----BEGIN PGP SIGNED MESSAGE-----\nhash: SHA512\n\n"
               b"some document payload\n-----BEGIN PGP SIGNATURE-----\nabc\n"
               b"-----END PGP SIGNATURE-----\n")
    art = _art(tmp_path, content)                      # writes doc.txt
    _sign(home, content, art.path.with_suffix(""))      # detached sig -> "doc"

    # Signer's public key lives ONLY in the known/untrusted ring, NOT the
    # (empty) truststore. Offline it can still verify because the key blob is
    # present locally, but the signer is not operator-vouched => not trusted.
    ts = tmp_path / "empty-ts"; ts.mkdir()
    ut = _pubdir(tmp_path, "known", keys["pub1"])

    v = gpg_check.verify_with_truststore(
        content, art.path.with_suffix(""),
        truststore_dir=ts, untrusted_dir=ut, workdir=tmp_path / "work")
    assert v.valid is True
    assert v.trusted is False

    # capability folds this into 'degraded'
    ctx = _ctx(tmp_path, art)
    os.environ["VISHWAS_GPG_TRUSTSTORE"] = str(ts)
    os.environ["VISHWAS_GPG_KNOWN_DIR"] = str(ut)
    try:
        res = gd.GovDocumentCapability()._digital_signature(art, ctx)
    finally:
        del os.environ["VISHWAS_GPG_TRUSTSTORE"]
        del os.environ["VISHWAS_GPG_KNOWN_DIR"]
    cr = res[0]
    assert cr.name == "digital_signature"
    assert cr.status == "degraded"
    assert cr.signals["valid"] is True
    assert cr.signals["signer_trusted"] is False


# ------------------------------------------------------------------- c ----
def test_c_tampered_payload_failed(keys, tmp_path):
    home = keys["home1"]
    data = b"original content"
    sig = tmp_path / "doc.sig"
    _sign(home, data, sig)

    # Tamper one byte; the signer's key IS present locally (untrusted ring) so
    # gpg can actually evaluate the digest and report BADSIG => valid False.
    tampered = b"original conteXt"  # flip one byte
    ut = _pubdir(tmp_path, "known", keys["pub1"])
    v = gpg_check.verify_with_truststore(
        tampered, sig,
        truststore_dir=tmp_path / "nope",  # nonexistent -> no vouched keys
        untrusted_dir=ut,
        workdir=tmp_path / "work")
    assert v.valid is False


# ------------------------------------------------------------------- d ----
def test_d_no_signature_skipped(tmp_path):
    # A plain text doc with no PGP armor and no PDF sig object -> 'skipped'
    art = _art(tmp_path, b"just some plain text, no signature at all")
    ctx = _ctx(tmp_path, art)
    res = gd.GovDocumentCapability()._digital_signature(art, ctx)
    cr = res[0]
    assert cr.name == "digital_signature"
    assert cr.status == "skipped"


# ------------------------------------------------------------------- e ----
def test_e_truststore_two_keys(keys, tmp_path):
    ts = tmp_path / "ts2"
    ts.mkdir()
    (ts / "a.asc").write_bytes(keys["pub1"])
    (ts / "b.asc").write_bytes(keys["pub2"])
    fps = gpg_check.load_truststore(ts)
    assert len(fps) == 2
    assert set(fps) == {keys["fpr1"], keys["fpr2"]}


# ------------------------------------------------------------------- f ----
def test_f_garbage_sig_failed_no_raise(keys, tmp_path):
    data = b"payload bytes"
    v = gpg_check.verify_with_truststore(data, b"\x00\x01\x02garbage-not-a-sig",
                                         truststore_dir=tmp_path / "nope",
                                         workdir=tmp_path / "work")
    # must not raise; either invalid or unverifiable, never valid
    assert v.valid in (False, None)


# ------------------------------------------------------------------- g ----
def test_g_corrupt_homedir_graceful(keys, tmp_path):
    # chmod 000 a homedir to simulate corruption; verify must degrade, not crash
    bad_home = tmp_path / "locked-home"
    bad_home.mkdir()
    kr = gpg_check.GpgKeyring(bad_home)
    try:
        os.chmod(bad_home, 0o000)
        v = kr.verify(b"data", b"sig-bytes")
        # graceful: a GpgVerdict comes back, no exception escapes
        assert isinstance(v, gpg_check.GpgVerdict)
    finally:
        os.chmod(bad_home, 0o700)


# ------------------------------------------------------- capability wiring
def test_capability_trusted_ok(keys, tmp_path):
    home = keys["home1"]
    content = (b"-----BEGIN PGP SIGNED MESSAGE-----\nhash: SHA512\n\n"
               b"content\n-----BEGIN PGP SIGNATURE-----\nx\n"
               b"-----END PGP SIGNATURE-----\n")
    art = _art(tmp_path, content)                      # writes doc.txt
    _sign(home, content, art.path.with_suffix(""))      # detached sig -> "doc"
    ts = tmp_path / "ts"
    ts.mkdir()
    (ts / "alice.asc").write_bytes(keys["pub1"])

    ctx = _ctx(tmp_path, art)
    os.environ["VISHWAS_GPG_TRUSTSTORE"] = str(ts)
    try:
        res = gd.GovDocumentCapability()._digital_signature(art, ctx)
    finally:
        del os.environ["VISHWAS_GPG_TRUSTSTORE"]
    cr = res[0]
    assert cr.name == "digital_signature"
    assert cr.status == "ok"
    assert cr.signals["valid"] is True
    assert cr.signals["signer_trusted"] is True
    assert cr.signals["signer_fingerprint"] == keys["fpr1"]
