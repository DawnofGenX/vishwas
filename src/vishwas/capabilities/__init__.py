"""Capability modules."""
from .url_phish_scanner import UrlPhishScannerCapability
from .malware_file import MaliciousFileCapability
from .gov_document import GovDocumentCapability
from .deepfake_video import DeepfakeVideoCapability
from .deepfake_audio import DeepfakeAudioCapability
from .image_facecheck import ImageFaceCheckCapability
from .cross_modal import CrossModalCapability

__all__ = [
    "UrlPhishScannerCapability", "MaliciousFileCapability", "GovDocumentCapability",
    "DeepfakeVideoCapability", "DeepfakeAudioCapability", "ImageFaceCheckCapability",
    "CrossModalCapability",
]


def default_capabilities(available_deps: set[str]) -> dict[str, list]:
    """Map target name -> ordered capability instances honoring dependency gates.

    Gating happens twice: here (skip instantiating heavy deps that can't load)
    and in the orchestrator (record 'unavailable' evidence instead of a silent
    skip so gaps stay visible in every report).
    """
    caps = {
        "url_phishing": [], "malicious_file": [], "gov_document": [],
        "deepfake_video": [], "deepfake_audio": [], "image_facecheck": [],
        "cross_modal": [], "document_generic": [], "unclassified": [],
    }
    try:
        # 2026-08-26 operator decision: PhishingScanner ALONE decides URL
        # phishing. VT/heuristics/url-phishml were dropped from this target.
        caps["url_phishing"].append(UrlPhishScannerCapability())
    except Exception:
        pass
    try:
        caps["malicious_file"].append(MaliciousFileCapability())
    except Exception:
        pass
    try:
        caps["gov_document"].append(GovDocumentCapability())
    except Exception:
        pass
    try:
        caps["deepfake_video"].append(DeepfakeVideoCapability())
        caps["deepfake_video"].append(CrossModalCapability())  # runs only when video carries audio
    except Exception:
        pass
    try:
        caps["deepfake_audio"].append(DeepfakeAudioCapability())
    except Exception:
        pass
    try:
        caps["image_facecheck"].append(ImageFaceCheckCapability())
    except Exception:
        pass
    try:
        caps["cross_modal"].append(CrossModalCapability())
    except Exception:
        pass
    return caps
