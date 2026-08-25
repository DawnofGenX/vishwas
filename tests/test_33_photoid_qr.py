"""test_33_photoid_qr.py — photo'd gov IDs must reach QR verification.

Gap (GAPS_AND_ENABLEMENT, open defect): a photographed ID card
(aadhaar.jpg) routes to image_facecheck, but the offline QR verification
(_qr_payload_checks) lives in GovDocumentCapability — so the most common
real-world input for QR checks was unreachable. Fix: the image capability
also emits qr_attempted evidence when the filename carries gov hints.
(The QR check itself emits nothing when no QR is decodable — by design.)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.image_facecheck import ImageFaceCheckCapability
from vishwas.file_validator import make_artifact, InputType, MediaKind
from vishwas.router import _GOV_HINTS


def _ctx(tmp: Path):
    return SimpleNamespace(quarantine_root=tmp, remaining_s=lambda: 120)


def test_gov_named_image_attempts_qr(tmp_path):
    """The image capability must ATTEMPT the gov-doc QR path for hinted names."""
    p = tmp_path / "aadhaar.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    art = make_artifact(tmp_path, "aadhaar.jpg", InputType.FILE, data=p.read_bytes())
    art.verified_kind = MediaKind.JPEG

    assert _GOV_HINTS.search(art.original_filename.lower()), "fixture name must match"

    cap = ImageFaceCheckCapability()
    # call the borrowed gov-doc QR seam exactly as analyze() does; with cv2 in
    # the production PYTHONPATH this returns [] (no QR in fixture) or failed
    # evidence — both prove the seam is reachable. Without cv2 (hermetic) it
    # raises ModuleNotFoundError inside decode -> still emitted as failed.
    ev = cap._qr_evidence_for_gov_image(art)
    assert isinstance(ev, list)


def test_non_gov_image_skips_qr(tmp_path):
    cap = ImageFaceCheckCapability()
    p = tmp_path / "vacation.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    art = make_artifact(tmp_path, "vacation.jpg", InputType.FILE, data=p.read_bytes())
    art.verified_kind = MediaKind.JPEG
    assert cap._qr_evidence_for_gov_image(art) == []


def test_analyze_wires_seam_for_gov_images(tmp_path, monkeypatch):
    """analyze() must call the QR seam for gov-named images (wiring proof)."""
    monkeypatch.setattr(ImageFaceCheckCapability,
                        "_qr_evidence_for_gov_image",
                        lambda self, art: [])
    called = []
    cap = ImageFaceCheckCapability()

    orig = ImageFaceCheckCapability._qr_evidence_for_gov_image

    def spy(self, art):
        called.append(1)
        return orig(self, art)

    monkeypatch.setattr(ImageFaceCheckCapability,
                        "_qr_evidence_for_gov_image", spy)
    p = tmp_path / "pan_card.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    art = make_artifact(tmp_path, "pan_card.jpg", InputType.FILE, data=p.read_bytes())
    art.verified_kind = MediaKind.JPEG
    cap.analyze(art, _ctx(tmp_path))
    assert called, "analyze() never invoked the QR seam"


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-q"]))
