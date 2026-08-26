"""Task: QR verification wired into GovDocumentCapability.analyze() (pipeline).

Hermetic by construction: every QR image is generated inline with the
qrcode lib into tmp_path; OCR/docling/RAG-cache/network gates are all
monkeypatched or env-isolated so analyze() stays fast and offline. The
Aadhaar fixture reuses the PINNED numeric payload + test public key from
test_24 (deterministic; private key exists nowhere). The capability's own
scrub guard is proven by feeding a hostile QrVerifyResult through the real
boundary.
"""
import base64
import gzip
import json
import re

import cv2
import numpy as np
import pytest
import qrcode

from vishwas.capabilities import gov_document as gd
from vishwas.capabilities.base import CheckResult
from vishwas.capabilities.gov_document import GovDocumentCapability
from vishwas.events import Artifact, InputType, JobContext, MediaKind
from vishwas.qr_verify import QrVerifyResult
from vishwas.qr_verify import aadhaar_secure

# ------------------------------------------------- pinned fixtures (test_24) --

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
_UID_RUN_RE = re.compile(r"\d{12}")


# ------------------------------------------------------------------ helpers --

def _write_pubkey(tmp_path):
    p = tmp_path / "fixture_test_pubkey.der"
    p.write_bytes(base64.b64decode(_TEST_PUBKEY_B64))
    return p


def _qr_png(path, data):
    qrcode.make(data).save(str(path))
    return path


def _image_artifact(png_path) -> Artifact:
    return Artifact(path=png_path, original_filename=png_path.name,
                    declared_type=InputType.IMAGE, verified_kind=MediaKind.PNG)


def _ctx(art: Artifact, tmp_path) -> JobContext:
    q = tmp_path / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    return JobContext(job_id="job_qr_pipeline", artifact=art, quarantine_root=q,
                      browser_available=False)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    """Text-less fast path + empty RAG cache + no network-capable gates.

    analyze() still runs its REAL digital-signature / qnative / official-web /
    rag stages — all of which degrade to skipped or indicator-only records for
    a text-less PNG, keeping this end-to-end honest.
    """
    monkeypatch.setattr(gd, "_extract_text", lambda art, ctx: ("", "none"))
    monkeypatch.setenv("VISHWAS_RAG_CACHE", str(tmp_path / "rag-cache"))
    for var in ("VISHWAS_DOCLING", "VISHWAS_QR_EXTRA_TRUST_PATHS"):
        monkeypatch.delenv(var, raising=False)


def _qr_checks(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.name == "qr_payload_check"]


# ------------------------------------------------------------------- tests ---

def test_aadhaar_qr_flows_end_to_end_through_analyze(monkeypatch, tmp_path):
    """Photo'd-card PNG carrying a signed Aadhaar QR -> one ok qr_payload_check."""
    monkeypatch.setenv("VISHWAS_QR_EXTRA_TRUST_PATHS", str(_write_pubkey(tmp_path)))
    png = _qr_png(tmp_path / "card.png", _AADHAAR_NUMERIC)
    art = _image_artifact(png)

    results = GovDocumentCapability().analyze(art, _ctx(art, tmp_path))

    qr = _qr_checks(results)
    assert len(qr) == 1, [r.name for r in results]
    chk = qr[0]
    assert chk.cost == "mid"
    assert chk.status == "ok", chk.notes
    assert chk.signals["qr_kind"] == "aadhaar_secure"
    assert chk.signals["signature_valid"] is True
    assert chk.signals["prob_forged"] == 0.05          # strong authenticity evidence
    assert len(chk.signals.get("aadhaar_last4", "")) <= 4
    assert "VALID" in chk.notes
    # scrub discipline holds at the capability boundary too
    dumped = json.dumps(chk.signals)
    assert not _UID_RUN_RE.search(dumped), dumped


def test_forged_aadhaar_qr_yields_failed_check_and_high_prob_forged(monkeypatch, tmp_path):
    """Tampered payload inside a QR image -> failed check, prob_forged 0.9."""
    monkeypatch.setenv("VISHWAS_QR_EXTRA_TRUST_PATHS", str(_write_pubkey(tmp_path)))
    signed, sig, _fields, _presence = aadhaar_secure.parse_payload(_AADHAAR_NUMERIC)
    tampered = bytearray(signed)
    tampered[10] ^= 0x01                                # flip one body byte
    blob = gzip.compress(bytes(tampered) + sig)
    forged_numeric = str(int.from_bytes(blob, "big"))
    png = _qr_png(tmp_path / "forged.png", forged_numeric)
    art = _image_artifact(png)

    results = GovDocumentCapability().analyze(art, _ctx(art, tmp_path))

    qr = _qr_checks(results)
    assert len(qr) == 1
    chk = qr[0]
    assert chk.status == "failed"
    assert chk.signals["signature_valid"] is False
    assert chk.signals["prob_forged"] == 0.9            # forged signature = strong evidence
    assert "forgery" in chk.notes.lower()
    assert not _UID_RUN_RE.search(json.dumps(chk.signals))


def test_epic_qr_structural_ok_maps_gentle_prob_forged(tmp_path):
    """EPIC envelope decrypts to spec JSON -> ok, structure_ok, prob_forged 0.2."""
    png = _qr_png(tmp_path / "epic_card.png", _EPIC_GOLDEN_B64)
    art = _image_artifact(png)

    results = GovDocumentCapability().analyze(art, _ctx(art, tmp_path))

    qr = _qr_checks(results)
    assert len(qr) == 1
    chk = qr[0]
    assert chk.status == "ok", chk.notes
    assert chk.signals["qr_kind"] == "epic_b64"
    assert chk.signals["structure_ok"] is True
    assert chk.signals["prob_forged"] == 0.2            # structural proof only
    assert chk.notes == "EPIC structure valid"


def test_image_without_qr_emits_no_new_checks(tmp_path):
    """Blank photo -> zero qr_payload_check records; absence adds no noise."""
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((240, 320, 3), 255, dtype=np.uint8))
    art = _image_artifact(blank)

    results = GovDocumentCapability().analyze(art, _ctx(art, tmp_path))

    assert _qr_checks(results) == []
    # sanity: the standard pipeline still ran around it
    names = [r.name for r in results]
    assert "document_extraction" in names and "doc_type_identify" in names


def test_non_image_artifact_never_enters_qr_pipeline(tmp_path):
    """Scope fence: only image media kinds get QR extraction."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 not-really-a-pdf-but-magic-bytes-match")
    art = Artifact(path=p, original_filename="doc.pdf",
                   declared_type=InputType.FILE, verified_kind=MediaKind.PDF)

    results = GovDocumentCapability().analyze(art, _ctx(art, tmp_path))

    assert _qr_checks(results) == []


def test_capability_layer_scrubs_uid_runs_defensively(monkeypatch, tmp_path):
    """If a hostile verifier result ever carries a full UID, the boundary
    collapses signals to {'scrubbed': True} — package guarantee asserted here."""
    png = _qr_png(tmp_path / "hostile.png", _AADHAAR_NUMERIC)

    def hostile_verify(payload, *, extra_trust_paths=None):
        return QrVerifyResult(kind="aadhaar_secure", status="ok",
                              signals={"dob": "123456789012",   # raw UID!
                                       "aadhaar_last4": "4321"},
                              detail="hostile fixture bypassing package scrub")

    monkeypatch.setattr("vishwas.qr_verify.verify_payload", hostile_verify)
    art = _image_artifact(png)

    results = GovDocumentCapability().analyze(art, _ctx(art, tmp_path))

    qr = _qr_checks(results)
    assert len(qr) == 1
    chk = qr[0]
    assert chk.signals == {"scrubbed": True}
    assert "prob_forged" not in chk.signals            # leaked evidence never scores
    assert "blocked" in chk.notes.lower()


def test_verify_crash_becomes_failed_evidence_not_exception(monkeypatch, tmp_path):
    """verify_payload blowing up must never raise out of analyze()."""
    png = _qr_png(tmp_path / "crashy.png", _AADHAAR_NUMERIC)

    def bomb(payload, *, extra_trust_paths=None):
        raise RuntimeError("simulated verifier crash")

    monkeypatch.setattr("vishwas.qr_verify.verify_payload", bomb)
    art = _image_artifact(png)

    results = GovDocumentCapability().analyze(art, _ctx(art, tmp_path))

    qr = _qr_checks(results)
    assert len(qr) == 1
    chk = qr[0]
    assert chk.status == "failed"
    assert chk.signals["error_class"] == "RuntimeError"
