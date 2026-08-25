"""Deterministic routing: input-type classification -> capability target.

Pins the router's decision tree (WORKFLOW.md section 2) to exact behaviour so a
regression can't silently re-route a dangerous item onto a benign pipeline.
"""
from __future__ import annotations

import pytest

from vishwas.router import Router
from vishwas.events import InputType


R = Router()


def _classify(msg):
    return R.classify(msg)


# ---------------------------------------------------------------- URLs -----
def test_bare_url_text_routes_to_phishing():
    d = _classify({"text": "https://evil.example.com/pay"})
    assert d.input_type is InputType.URL
    assert d.is_url is True
    assert d.url == "https://evil.example.com/pay"
    assert R.target_for(d) == "url_phishing"


def test_url_embedded_in_text_routes_to_phishing():
    d = _classify({"text": "plz click http://a.b/c ok?"})
    assert d.urls_in_text, "should have extracted the embedded URL"
    assert R.target_for(d) == "url_phishing"


def test_https_url_with_path_and_query():
    d = _classify({"text": "https://bank-secure-login.example-verify.com/pay/login?ref=wa"})
    assert d.is_url
    assert "example-verify.com" in d.url


# ------------------------------------------------------------- media ------
@pytest.mark.parametrize("ext,target", [
    (".mp4", "deepfake_video"),
    (".mkv", "deepfake_video"),
    (".mov", "deepfake_video"),
    (".mp3", "deepfake_audio"),
    (".wav", "deepfake_audio"),
    (".opus", "deepfake_audio"),
    (".m4a", "deepfake_audio"),
])
def test_media_ext_routes_to_correct_target(ext, target):
    d = _classify({"media_path": f"x{ext}"})
    assert R.target_for(d) == target, f"{ext} should route to {target}, got {R.target_for(d)}"


@pytest.mark.parametrize("ext", ["png", "jpg", "jpeg", "webp"])
def test_image_routes_to_facecheck(ext):
    d = _classify({"media_path": f"f.{ext}", "input_type": "image"})
    assert R.target_for(d) == "image_facecheck"


# ----------------------------------------------- exec / package -> malware --
@pytest.mark.parametrize("ext", ["exe", "dll", "so", "apk", "jar", "zip"])
def test_executable_and_package_route_to_malware(ext):
    d = _classify({"media_path": f"file.{ext}"})
    assert R.target_for(d) == "malicious_file", \
        f".{ext} must be treated as potentially malicious, not benign media"


def test_disguised_media_is_re_routed_by_magic_bytes():
    """A .mp4 whose magic bytes say PE binary must go to malware, not deepfake."""
    # classify with an explicit input_type hint of video, but the *final* target
    # comes from target_for(art) after validation; here we simulate the art.
    class Art:
        verified_kind = None
    from vishwas.events import MediaKind
    d = _classify({"media_path": "video.mp4", "input_type": "video"})
    art = Art(); art.verified_kind = MediaKind.PE
    assert R.target_for(d, art) == "malicious_file"


# ------------------------------------------------------------ documents ----
def test_gov_hint_in_text_routes_to_gov_document():
    d = _classify({"media_path": "cert.pdf",
                   "text": "please verify my PAN card number ABC123"})
    assert R.target_for(d) == "gov_document"


def test_plain_pdf_without_gov_hint_is_generic_document():
    d = _classify({"media_path": "notes.pdf", "text": "meeting minutes"})
    assert R.target_for(d) == "document_generic"


# ------------------------------------------------------------- plain text --
def test_plain_text_no_url_routes_to_unclassified_or_text():
    d = _classify({"text": "hello how are you"})
    t = R.target_for(d)
    assert t in ("unclassified", "text") or d.input_type is InputType.TEXT


def test_empty_message_is_unknown():
    d = _classify({})
    assert d.input_type is InputType.UNKNOWN
    assert R.target_for(d) in ("unclassified", "")
