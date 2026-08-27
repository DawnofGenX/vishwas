"""Honesty / coverage regression tests for the fusion verdict layer.

These pin the 2026-08-27 fixes:
  A. Zero real detector evidence -> UNABLE_TO_VERIFY (never a 0.5 -> "MEDIUM"
     pin when nothing actually ran).
  B. document_generic / unclassified carry weight maps + a scanner, so a
     scanned file gets a real verdict instead of dead-ending at UNVERIFIED.
  C. ext_mismatch (a const_true presence flag) only counts as risk when a real
     mismatch is present — a matching extension is N/A, not a phantom +1.0.
  D. A file a decisive scanner actually cleared reads LOW/TRUST, not MEDIUM.
"""
from vishwas.fusion import FusionEngine, _extract, _SIGNAL_SOURCES, WEIGHTS
from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict

fe = FusionEngine()


# ------------------------------------------------------- A. zero evidence ----
def test_video_all_detectors_gated_is_unverified_not_medium():
    checks = [
        C("effort_face_forensics", "heavy", "unavailable", {}, "weights missing"),
        C("frame_heuristics", "mid", "unavailable", {}, "weights missing"),
        C("cross_modal_av", "heavy", "unavailable", {}, "no audio"),
        C("havic_crossmodal_model", "heavy", "unavailable", {}, "weights missing"),
        C("media_probe", "cheap", "ok", {"duration_s": 5}, "ffprobe ok"),
    ]
    d = fe.decide("deepfake_video", checks)
    assert d.verdict is Verdict.UNABLE_TO_VERIFY, d.verdict
    assert d.score == 0.0


def test_url_scanner_gated_is_unverified_not_medium():
    checks = [
        C("ext_mismatch_flag", "cheap", "ok", {}, "extension matches"),
        C("url_phish_scanner", "mid", "unavailable", {}, "scanner not provisioned"),
    ]
    d = fe.decide("url_phishing", checks)
    assert d.verdict is Verdict.UNABLE_TO_VERIFY, d.verdict


def test_document_generic_no_scanner_is_unverified():
    # document_generic now HAS a weight map, but with no scanner run there is
    # still no real value -> honest UNVERIFIED (not the old empty-map path, and
    # not a fake medium).
    checks = [C("ext_mismatch_flag", "cheap", "ok", {}, "match")]
    d = fe.decide("document_generic", checks)
    assert d.verdict is Verdict.UNABLE_TO_VERIFY, d.verdict


# --------------------------------------------------- B. real doc verdicts ----
def test_document_generic_has_weight_map():
    assert WEIGHTS["document_generic"], "document_generic must not be an empty map"
    assert WEIGHTS["unclassified"], "unclassified must not be an empty map"


def test_clean_pdf_scanned_is_trust():
    checks = [
        C("ext_mismatch_flag", "cheap", "ok", {}, "match"),
        C("clamscan", "cheap", "ok", {"detected": False}, "clean"),
        C("yara_x", "mid", "ok", {"hits_norm": 0.0}, "no hits"),
    ]
    d = fe.decide("document_generic", checks)
    assert d.verdict is Verdict.TRUST, d.verdict


def test_malicious_pdf_is_do_not_use():
    checks = [
        C("ext_mismatch_flag", "cheap", "ok", {}, "match"),
        C("clamscan", "cheap", "ok", {"detected": False}, "clean"),
        C("yara_x", "mid", "ok", {"hits_norm": 0.9}, "pdf exploit family"),
    ]
    d = fe.decide("document_generic", checks)
    assert d.verdict is Verdict.DO_NOT_USE, d.verdict


# ----------------------------------------------- C. ext_mismatch const_true --
def test_ext_mismatch_match_case_is_not_risky():
    spec = _SIGNAL_SOURCES["ext_mismatch.present"]
    match = C("ext_mismatch_flag", "cheap", "ok", {}, "extensions MATCH")
    state, _ = _extract(spec, match)
    assert state == "known_gap", state  # N/A, not a risk value


def test_ext_mismatch_real_mismatch_is_risky():
    spec = _SIGNAL_SOURCES["ext_mismatch.present"]
    mism = C("ext_mismatch_flag", "cheap", "ok",
             {"declared": "pdf", "verified": "pe"}, "MISMATCH")
    state, val = _extract(spec, mism)
    assert state == "value" and val == 1.0


def test_disguised_executable_not_masked_as_clean():
    # PE disguised as PDF, but AV-clean: the mismatch must block the clean bonus
    # and keep it out of TRUST/UNVERIFIED.
    checks = [
        C("ext_mismatch_flag", "cheap", "ok",
          {"declared": "pdf", "verified": "pe"}, "MISMATCH"),
        C("clamscan", "cheap", "ok", {"detected": False}, "clean"),
        C("yara_x", "mid", "ok", {"hits_norm": 0.0}, "no hits"),
        C("file_entropy", "cheap", "ok", {"entropy": 6.0, "anomaly": False}, "normal"),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE), d.verdict


# ------------------------------------------------------- D. clean-side file --
def test_clean_scanned_file_is_trust():
    checks = [
        C("ext_mismatch_flag", "cheap", "ok", {}, "match"),
        C("clamscan", "cheap", "ok", {"detected": False}, "clean"),
        C("yara_x", "mid", "ok", {"hits_norm": 0.0}, "no hits"),
        C("file_entropy", "cheap", "ok", {"entropy": 5.0, "anomaly": False}, "normal"),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict is Verdict.TRUST, d.verdict


def test_detected_malware_is_do_not_use():
    checks = [
        C("ext_mismatch_flag", "cheap", "ok", {}, "match"),
        C("clamscan", "cheap", "ok", {"detected": True, "sig": "Win.Trojan"}, "HIT"),
        C("yara_x", "mid", "ok", {"hits_norm": 0.5}, "hits"),
        C("file_entropy", "cheap", "ok", {"entropy": 7.9, "anomaly": True}, "packed"),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict is Verdict.DO_NOT_USE, d.verdict


def test_unscanned_file_only_entropy_is_not_falsely_trusted():
    # entropy alone is not a decisive scan -> must NOT be TRUST.
    checks = [
        C("ext_mismatch_flag", "cheap", "ok", {}, "match"),
        C("clamscan", "cheap", "unavailable", {}, "no clamav"),
        C("yara_x", "mid", "unavailable", {}, "no yara"),
        C("file_entropy", "cheap", "ok", {"entropy": 5.0, "anomaly": False}, "normal"),
    ]
    d = fe.decide("malicious_file", checks)
    assert d.verdict is not Verdict.TRUST, d.verdict
