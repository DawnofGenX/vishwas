"""Task: offline QR verification package (vishwas.qr_verify).

Hermetic by construction: every QR image is generated inline with the
qrcode lib into tmp_path, the Aadhaar signature test uses a PINNED numeric
payload + test public key (embedded below as constants — deterministic
across runs, private key exists nowhere), and no test touches the network.
Degradation mimics photo-like capture per the scouting bake-off: 3x upscale,
seeded Gaussian noise, then GaussianBlur.
"""
import base64
import gzip
import json
import re

import cv2
import numpy as np
import pytest
import qrcode

from vishwas.qr_verify import QrVerifyResult, decode_image, verify_payload
from vishwas.qr_verify import aadhaar_secure
from vishwas.qr_verify.classifier import classify_payload


# ------------------------------------------------------- pinned fixtures ----

# Deterministic Secure-QR payload signed by a TEST RSA-2048 key whose public
# SPKI is _TEST_PUBKEY_B64. The private key exists nowhere; real UIDAI
# signatures cannot be synthesized, hence this pinned pair.
_AADHAAR_NUMERIC = (
    "715831756024166361525606427419363453301870491946012191757430111214933911894496302490382875483252784901542557944980783194198506792732181324021421602815735571206787462554460976815042123963980386828542644127106070518301524220866232555802801875143415200558316039215624140892421980698222436475889489682760114258403712037835021436095446981315585318633197152415714128720205472070382449919712868201403634253863206184909291548449137274924284555823918994981175053031795270297194787712555758099815681874870671574748010214681587667155580314937990214632236749188241642823864863151308432776906555671038434405482623690112520953617967254367790968234157725275624014442427074305651277340230320021226836625795705447501176257585329314426204270260356639732961611553911621024557028493547279132680231944022800105590481297790762035386677872132210734337703789977086398632681316642073084920459966260772069993090299566230035623937035163876588821479424"
)
_TEST_PUBKEY_B64 = (
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAxPtJCn7epHM4rx5b6Q7lBCkuDOo3W2h"
    "BUc8057CeKX8YLM26Ye8ImXE66xLUGqsbeAU8m/qAsq1SZ9/zAaJR+ZwS4im7m3o0OdhNMsJqlN"
    "iVFUvxTCDYkkZEOMwOMMIisqTDKsjx5T5qEe8glf+PnilT7L3c1XtYtS1JeGkZhlhYzqnLl2Djo"
    "gP4Qd7HDeRQCRml6zdaLMMDBz4YxblqU5ciNVyaEjUK3yRqwVQFFlt8V+5Kt6J83bUpgO02BsV6"
    "XZPBrXxsIQRlEHOCT3K4KiLSayA1/tg0vSgRdGA24SjrukaOYyPtR9nVOlf06la18j+AwgUh7Ws"
    "0SSOjDjnF/QIDAQAB"
)
_EPIC_GOLDEN_B64 = ("dbhvecY6Roa4NF3gAzEbkTibZZzXAEYpMg8197BQWMS2+ID24FGDKWB5IEcuxjsA"
                    "gLYJlzD8OcfimODRDN7mZA==")
_EPIC_KEY = b"X_4k$uq23FSwI.qT"
_EPIC_IV = b"H76$suq23_po(8sD"


# ------------------------------------------------------------- helpers ------

def _write_pubkey(tmp_path):
    p = tmp_path / "fixture_test_pubkey.der"
    p.write_bytes(base64.b64decode(_TEST_PUBKEY_B64))
    return p


def _qr_png(path, data):
    """Render `data` to a QR PNG at `path` (numeric data -> numeric mode)."""
    qrcode.make(data).save(str(path))
    return path


def _degrade_photo_like(png_path, out_path):
    """3x upscale + seeded Gaussian noise + blur, like the scouting bake-off."""
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    big = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    noise = np.random.default_rng(20260822).normal(0, 20, big.shape)
    noisy = np.clip(big.astype("int16") + noise, 0, 255).astype("uint8")
    blurred = cv2.GaussianBlur(noisy, (5, 5), 1.3)
    cv2.imwrite(str(out_path), blurred)
    return out_path


def _epic_encrypt(obj) -> str:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    raw = json.dumps(obj, separators=(",", ":")).encode()
    pad = 16 - len(raw) % 16
    raw += bytes([pad]) * pad
    enc = Cipher(algorithms.AES(_EPIC_KEY), modes.CBC(_EPIC_IV)).encryptor()
    return base64.b64encode(enc.update(raw) + enc.finalize()).decode()


# ---------------------------------------------------------------- tests -----

def test_aadhaar_end_to_end_decode_classify_verify(tmp_path):
    """Fixture PNG (photo-degraded) -> decode -> classify -> RSA verify OK."""
    png = _qr_png(tmp_path / "aadhaar_clean.png", _AADHAAR_NUMERIC)
    degraded = _degrade_photo_like(png, tmp_path / "aadhaar_noisy.png")
    texts = decode_image(str(degraded))
    assert texts == [_AADHAAR_NUMERIC]

    assert classify_payload(_AADHAAR_NUMERIC) == "aadhaar_secure"

    pubkey = _write_pubkey(tmp_path)
    res = verify_payload(texts[0], extra_trust_paths=[pubkey])
    assert isinstance(res, QrVerifyResult)
    assert res.kind == "aadhaar_secure"
    assert res.status == "ok", res.detail
    assert res.signals["signature_valid"] is True
    assert res.signals["signer_cert"] == "fixture_test_pubkey.der"
    assert res.signals["version"] == "20260216"
    assert res.signals["name_present"] is True
    assert res.signals["pincode"] == "800001"
    # clean capture decodes identically (sanity: degradation didn't mask content)
    assert decode_image(str(png)) == [_AADHAAR_NUMERIC]


def test_aadhaar_tampered_body_fails_signature(tmp_path):
    """Flip one body byte before verify -> honest signature_valid=False."""
    signed, sig, _fields, _presence = aadhaar_secure.parse_payload(_AADHAAR_NUMERIC)
    tampered = bytearray(signed)
    tampered[10] ^= 0x01  # inside a demographic field segment
    blob = gzip.compress(bytes(tampered) + sig)
    tampered_numeric = str(int.from_bytes(blob, "big"))

    assert classify_payload(tampered_numeric) == "aadhaar_secure"

    pubkey = _write_pubkey(tmp_path)
    res = verify_payload(tampered_numeric, extra_trust_paths=[pubkey])
    assert res.kind == "aadhaar_secure"
    assert res.status == "failed"
    assert res.signals["signature_valid"] is False
    assert res.signals["error_class"] == "signature_mismatch"
    assert res.detail  # human-readable honesty


def test_epic_round_trip_golden_vector_with_qr_image(tmp_path):
    """epicqr envelope: encrypt -> golden match -> QR PNG -> decode -> verify."""
    obj = {"epic_no": "ABC1234566", "unique_generated_id": 12345}
    assert _epic_encrypt(obj) == _EPIC_GOLDEN_B64  # deterministic cipher chain

    png = _qr_png(tmp_path / "epic.png", _EPIC_GOLDEN_B64)
    texts = decode_image(str(png))
    assert texts == [_EPIC_GOLDEN_B64]
    assert classify_payload(_EPIC_GOLDEN_B64) == "epic_b64"

    res = verify_payload(texts[0])
    assert res.kind == "epic_b64"
    assert res.status == "ok", res.detail
    assert res.signals["epic_no"] == "ABC1234566"
    assert res.signals["structure_ok"] is True


def test_pan_legacy_text_classification_and_extraction():
    text = ("NAME:RAVI KUMAR\nFATHER:SURESH KUMAR\n"
            "DOB:01/01/1990\nPAN:ABCDE1234F")
    assert classify_payload(text) == "pan_text"
    res = verify_payload(text)
    assert res.kind == "pan_text"
    assert res.status == "ok", res.detail
    assert res.signals["pan"] == "ABCDE1234F"
    assert res.signals["pan_format_valid"] is True
    assert res.signals["name_line"] == "RAVI KUMAR"
    assert res.signals["dob_found"] is True


def test_unknown_payload_honest_unavailable():
    payload = "random whatsapp note about tomorrow's weather, 42C likely"
    assert classify_payload(payload) == "unknown"
    res = verify_payload(payload)
    assert isinstance(res, QrVerifyResult)
    assert res.kind == "unknown"
    assert res.status in ("unavailable", "degraded")
    assert res.detail


def test_decoder_cv2_fallback_when_zxing_blind(monkeypatch, tmp_path):
    """zxing returning nothing must fall through to cv2.QRCodeDetector."""
    import vishwas.qr_verify.decoder as decoder_mod

    monkeypatch.setattr(decoder_mod.zxingcpp, "read_barcodes",
                        lambda *a, **k: [])
    png = _qr_png(tmp_path / "fallback.png", "vishwas-fallback-probe-123")
    got = decoder_mod.decode_image(str(png))
    assert got == ["vishwas-fallback-probe-123"]

    # and garbage input degrades to an empty list instead of raising
    assert decoder_mod.decode_image(tmp_path / "does_not_exist.png") == []


def test_no_full_uid_leaks_through_signals(tmp_path):
    """json.dumps(signals) must never contain a 12-digit run — ever."""
    uid_run_re = re.compile(r"\d{12}")
    pubkey = _write_pubkey(tmp_path)

    # (a) legitimate fixture verification
    res = verify_payload(_AADHAAR_NUMERIC, extra_trust_paths=[pubkey])
    assert not uid_run_re.search(json.dumps(res.signals))
    assert len(res.signals.get("aadhaar_last4", "")) <= 4

    # (b) hostile body carrying a full 12-digit UID in reference_id:
    #     unsigned garbage sig forces failure, but signals must still be scrubbed
    import hashlib
    filler = "".join(hashlib.sha256(bytes([i])).hexdigest() for i in range(24))
    # segs[4] maps to the emitted 'dob' field -> scrubber must fire on it
    fields = ["20260216", "3", "Sita Devi", "01-01-1965", "123456789012", filler]
    body = b"\xff".join(f.encode() for f in fields)
    blob = gzip.compress(body + bytes(256))
    hostile_numeric = str(int.from_bytes(blob, "big"))
    res2 = verify_payload(hostile_numeric, extra_trust_paths=[pubkey])
    assert res2.status == "failed"  # bad signature reported honestly
    assert res2.signals["dob"] == "[REDACTED]"
    dumped = json.dumps(res2.signals)
    assert not uid_run_re.search(dumped), dumped
    assert res2.signals.get("uid_leak_scrubbed") is True
