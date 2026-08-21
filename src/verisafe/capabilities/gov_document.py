"""Government-document verification capability.

Progressive & spec-compliant:
  1. Docling extraction (pip-gated) OR tesseract OCR (bin-gated) — always fall
     back so the pipeline degrades, never dies.
  2. Document-type + issuing-authority identification (lexicon based, Indian
     documents prioritized: Aadhaar/e-KYC, PAN, Voter ID, Passport, DL, RL,
     EPF, Bank Passbook, Income Tax notices, Ration Card, NREGA/PM-KISAN letters).
  3. AUTHORITATIVE checks, in priority order:
       a. Digital signature / QR-native verification  (PGP/gpg, PDF signer)
       b. DigiLocker API           (key-gated; official UIDAI ecosystem)
       c. QR payload verification  (parse & validate structure per doc type)
       d. API Setu                 (key-gated; govt-API aggregator)
       e. Official website lookups via controlled Playwright (gated) / urllib
          SSRF-guarded fallback, only against allowlisted official domains.
  4. Versioned verified-RAG database = RETRIEVAL CACHE ONLY (never source of
     truth): stores previously-verified template shapes / number formats so
     repeated verifications are fast; every match is still backed by live
     authoritative evidence or flagged as cache-consistent-only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from ..events import Artifact, JobContext, MediaKind
from .base import CheckResult

# Allowlisted official domains — controlled access list (SSRF-safe by construction)
OFFICIAL_DOMAINS = [
    "digilocker.gov.in", "uidai.gov.in", "incometax.gov.in", "passportindia.gov.in",
    "epfindia.gov.in", "nrega.nic.in", "pmkisan.gov.in", "nsdl.com", "voter.gov.in",
    "indiagov.org", ".nic.in", "rajbundeshawar.go.in", "sbi.co.in",
]

DOC_TYPE_LEXICON: dict[str, list[str]] = {
    "aadhaar_ecyc": ["ekyc", "aadharnumber", "aadhaar number", "uidai", "unique identification"],
    "pan_card": ["permanent account number", "pan card", "income-tax department"],
    "voter_id": ["electoral roll", "voter id", "ephw", "electoral registration office", "voter card"],
    "passport": ["passport", "issue post", "expiry date", "mrz", "<ind"],
    "driving_license": ["driving licence", "driving license", "vehicle class"],
    "ration_card": ["ration card", "food supply department", "card holder"],
    "bank_passbook": ["pass book", "account number", "branch code", "ifsc"],
    "epf_statement": ["employee provident fund", "epf", "uepf member id", "challan"],
    "income_tax_notice": ["department of income tax", "tax year", "assessment year", "intimation u/s"],
    "pm_kisan_letter": ["pm-kisan", "direct benefit transfer", "samrat nidhi"],
}

NUM_PATTERNS = {
    "aadhaar": r"\b(\d{4})\s?(?:\d{4}\s?)?\d{4}\b",
    "pan": r"\b[A-Z]{5}[0-9][A-Z][0-9][A-Z][0-9]\b",
    "upi_vpa": r"\b[\w.\-]{2,}@[a-z]{2,15}\b",
    "ifsc": r"\b([A-Z]{4}0[A-Z0-9]{6})\b",
}


def _extract_text(art: Artifact, ctx: JobContext) -> tuple[str, str]:
    """Return (text, extractor_name). Try best available; degrade gracefully."""
    kind = art.verified_kind or MediaKind.UNKNOWN
    # 1) Docling (structured extraction incl. layout) — pip-gated
    dl = os.environ.get("VERISAFE_DOCLING")
    if dl and _which(dl):
        outdir = ctx.quarantine_root / "docling"
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            p = subprocess.run([dl, str(art.path), "-o", str(outdir)],
                               capture_output=True, text=True, timeout=180)
            md_files = list(outdir.rglob("*.md"))
            if p.returncode == 0 and md_files:
                return _join_text(md_files), "docling"
        except Exception:
            pass
    # 1b) Docling via its Python API — matches the `docling` dep gate in
    #     app.py (find_spec), covering pip installs where no CLI is wired.
    #     Lazy + cached: first call pays the model-load cost once per process.
    dl_py = _try_docling_python(art, ctx)
    if dl_py is not None:
        return dl_py
    # 2) PDF text via pypdf if available
    if kind == MediaKind.PDF:
        try:
            import pypdf  # type: ignore
            r = pypdf.PdfReader(str(art.path))
            txt = "\n".join((pg.extract_text() or "") for pg in r.pages[:6])
            if txt.strip():
                return txt, "pypdf"
        except Exception:
            pass
    # 3) MS Office via zipfile xml strip
    if kind in (MediaKind.MS_OFFICE_DOCX, MediaKind.MS_OFFICE_XLSX,
                MediaKind.MS_OFFICE_PPTX, MediaKind.ZIP):
        try:
            import zipfile
            zf = zipfile.ZipFile(art.path)
            parts = [n for n in zf.namelist() if n.endswith(".xml")][:60]
            import xml.etree.ElementTree as ET
            blob = []
            for n in parts:
                data = zf.read(n)[:500_000]
                m = re.findall(r"<[^>]*>([^<>]+)</", data.decode("utf-8", "ignore"))
                blob.extend(m[:400])
            txt = " ".join(blob)
            if txt.strip():
                return txt, "office_xml"
        except Exception:
            pass
    # 4) Images / scanned: tesseract
    if kind in (MediaKind.PNG, MediaKind.JPEG, MediaKind.WEBP, MediaKind.GIF,
                MediaKind.TIFF, MediaKind.HEIC, MediaKind.EMPTY):
        exe = _which(os.environ.get("VERISAFE_TESSERACT_BIN", "tesseract"))
        if exe:
            img = art.path
            if kind == MediaKind.HEIC:
                img = _heic_to_png(art, ctx)  # may fail -> falls through
            ocr_txt = ctx.quarantine_root / "ocr.txt"
            try:
                p = subprocess.run([exe, str(img), "stdout", "-l", "eng+hin"],
                                   capture_output=True, text=True, timeout=120)
                txt = p.stdout or ""
                if txt.strip():
                    ocr_txt.write_text(txt)
                    return txt, "tesseract"
            except Exception:
                pass
    # 5) plain text-ish kinds
    if kind in (MediaKind.PLAIN_TEXT, MediaKind.HTML, MediaKind.XML,
                MediaKind.JSON, MediaKind.CSV, MediaKind.SOURCE_CODE):
        try:
            return art.path.read_bytes().decode("utf-8", errors="replace"), "raw"
        except OSError:
            pass
    return "", "none"


def _heic_to_png(art: Artifact, ctx: JobContext) -> Path:
    out = ctx.quarantine_root / "converted.png"
    ffmpeg = _which(os.environ.get("VERISAFE_FFMPEG_BIN", "ffmpeg"))
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-i", str(art.path), str(out)],
                       capture_output=True, timeout=60)
    return out


def _which(binp: str | None) -> str | None:
    import shutil
    if not binp:
        return None
    w = shutil.which(binp)
    return w


# --- Docling Python-API bridge (cached; heavy model load happens once) -------
_DOCLING_CONV: Any = None
_DOCLING_TRIED: bool = False


def _try_docling_python(art: Artifact, ctx: JobContext) -> tuple[str, str] | None:
    """Run docling via its Python API when it's importable but no CLI is wired.

    Mirrors the `docling` dep detected by app.py (`importlib.util.find_spec`).
    Returns (markdown_text, "docling") on success, else None → caller falls
    through to the lighter extractors. Lazy import + process-level cache: the
    first call pays for the ONNX/layout model load once per process; subsequent
    jobs reuse it. Soft budgets keep it thermal-safe on the i5-8250U target:
    skipped when <45s remain, capped at 6 pages (parity with the pypdf branch),
    output truncated like _join_text. Never raises.
    """
    global _DOCLING_CONV, _DOCLING_TRIED
    if _DOCLING_TRIED and _DOCLING_CONV is None:
        return None                      # already known-unavailable this process
    try:
        if ctx.remaining_s() < 45:
            return None                  # too little budget to start a heavy run
        import importlib.util
        if importlib.util.find_spec("docling") is None:
            _DOCLING_TRIED = True
            return None                  # gated off — same as the dep matrix says
        from docling.datamodel.base_models import DocumentStream
        from docling.document_converter import DocumentConverter
        import io as _io
        if _DOCLING_CONV is None:
            _DOCLING_CONV = DocumentConverter()
        _DOCLING_TRIED = True
        with art.path.open("rb") as fh:
            blob = fh.read(30 * 1024 * 1024)   # 30MB cap mirrors quarantine scan size
        res = _DOCLING_CONV.convert(
            DocumentStream(name=Path(art.original_filename).suffix or ".pdf",
                           stream=_io.BytesIO(blob)),
            max_num_pages=6,
        )
        md = getattr(res, "document", None)
        text = (md.export_to_markdown() if md is not None else "") or ""
        # Defensive: never hand a non-str downstream (identify_document does
        # `"kw" in text`). Some docling builds/branches can emit bytes.
        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode("utf-8", "ignore")
        if isinstance(text, str) and text.strip():
            return text[:200_000], "docling"
        return None
    except Exception:
        # Import/convert failure (missing models, bad file) → fall through to
        # the lighter extractors; never let a heavy optional break the ladder.
        if not _DOCLING_TRIED:
            _DOCLING_TRIED = True
            _DOCLING_CONV = None
        return None


def _join_text(paths: list[Path]) -> str:
    return "\n\n".join(p.read_text(errors="ignore")[:200_000] for p in paths[:3])


# --- PAdES / CMS-PKCS#7 signature verification (lazy; mirrors docling idiom) --
_PADES_MOD: Any = None          # cached verisafe.pades_check module (or None)
_PADES_TRIED: bool = False     # process-level "already known-unavailable" latch


def _get_pades():
    """Lazy-import the PAdES/CMS verifier (pades dep gate). Never raises.

    Mirrors app.py's ``pades`` dep detection (asn1crypto + cryptography both
    importable) so a missing/trimmed library yields the cheap indicator-only
    branch rather than a crash. Cache semantics match _DOCLING_*: first call
    pays the import cost once per process, later calls are O(1).
    """
    global _PADES_MOD, _PADES_TRIED
    if _PADES_TRIED:
        return _PADES_MOD
    try:
        import importlib.util
        if (importlib.util.find_spec("asn1crypto") is None
                or importlib.util.find_spec("cryptography") is None):
            raise ImportError("pades deps not importable")
        from verisafe import pades_check
        _PADES_MOD = pades_check
    except Exception:  # noqa: BLE001 — graceful-degradation contract
        _PADES_MOD = None
    _PADES_TRIED = True
    return _PADES_MOD


_DEFAULT_TRUSTSTORE = os.environ.get(
    "VERISAFE_CA_TRUSTSTORE",
    str(Path(__file__).resolve().parent.parent / "assets" / "ca_truststore"))


def identify_document(text: str) -> tuple[str | None, float, str | None, float]:
    """Lexicon scoring: doc type + likely issuing authority. Deterministic."""
    low = text.lower()
    best_t, best_ts, best_a, best_as = None, 0.0, None, 0.0
    for dt, kws in DOC_TYPE_LEXICON.items():
        score = sum(1.0 for kw in kws if kw in low) / max(1, len(kws))
        if score > best_ts:
            best_t, best_ts = dt, score
    auths = [
        ("UIDAI", ["uidai", "unique identification authority of india"]),
        ("Income Tax Department", ["department of income tax", "director general of income tax"]),
        ("RBI", ["reserve bank of india"]),
        ("EPFO", ["employee provident fund organization"]),
        ("Ministry of Road Transport", ["licensing authority", "rto"]),
        ("Food Supply Dept", ["food and civil supplies", "food supply department"]),
        ("Election Commission", ["electoral commission", "returning officer"]),
    ]
    for name, kws in auths:
        s = sum(1.0 for kw in kws if kw in low) / max(1, len(kws))
        if s > best_as:
            best_a, best_as = name, s
    return (best_t if best_ts >= 0.5 else None), round(best_ts, 3), \
           (best_a if best_as >= 0.5 else None), round(best_as, 3)


class GovDocumentCapability:
    requires: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.rag_cache_path = Path(os.environ.get(
            "VERISAFE_RAG_CACHE", "/home/hermes/rag-cache")) / "cache-index.json"
        self.rag_version = int(os.environ.get("VERISAFE_RAG_VERSION", "1"))

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        text, extractor = _extract_text(art, ctx)
        if not text.strip():
            out.append(CheckResult("document_extraction", "cheap", "degraded",
                                   {"extractor": extractor},
                                   "could not extract readable text; layout-only clues may remain"))
        else:
            out.append(CheckResult("document_extraction", "cheap", "ok",
                                   {"extractor": extractor, "chars": len(text)},
                                   f"extracted with {extractor}"))
        doc_type, ts, auth, ascore = identify_document(text)
        out.append(CheckResult("doc_type_identify", "cheap", "ok",
                               {"doc_type": doc_type, "type_conf": ts,
                                "issuer": auth, "issuer_conf": ascore},
                               f"likely {doc_type or 'unrecognized'} ({ts:.0%})" if doc_type
                               else "no known government-doc fingerprint in text"))
        # ---- authoritative verification, priority order -----------------
        out.extend(self._digital_signature(art, ctx))
        out.extend(self._qnative_checks(art, text, ctx, doc_type))
        out.extend(self._digilocker(art, text, ctx))
        out.extend(self._api_setu(text, ctx))
        out.extend(self._official_web_check(ctx, doc_type, auth))
        out.extend(self._rag_cache(text, ctx, doc_type, auth))
        return out

    def _digital_signature(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        kind = art.verified_kind or MediaKind.UNKNOWN
        # PDF: inspect for signature objects (cert / PKCS7)
        if kind == MediaKind.PDF:
            raw = art.path.read_bytes()
            has_sig = bool(re.search(rb"/Type\s*/Sig", raw)) or (
                b"PKCS" in raw or b"CMS_SIGNED_DATA" in raw or b"CRLink" in raw)
            pades_mod = _get_pades() if has_sig and ctx.pades_available else None
            if pades_mod is not None:
                return self._run_pades(pades_mod, raw, has_sig)
            # Gate off -> EXACTLY today's indicator-only behavior (brief rule:
            # never claim 'unavailable' for an expected-absent optional dep;
            # orchestrator .gated evidence covers provisioning gaps at the
            # capability level, not inside sub-checks).
            status, notes = ("ok", "signed-object indicator present"
                             ) if has_sig else ("degraded", "no digital-signature object found")
            return [CheckResult("digital_signature", "mid", status,
                                {"has_sig_object": bool(has_sig), "format": "pdf"},
                                notes)]
        # PGP-signed content (.asc/.sig siblings or armor inside)
        if b"-----BEGIN PGP" in art.path.read_bytes()[:200_000]:
            from .. import gpg_check
            gpg = _which(os.environ.get("VERISAFE_GPG_BIN", "gpg"))
            if not gpg or not gpg_check.available():
                return [CheckResult("digital_signature", "mid", "degraded",
                                    {"format": "pgp", "gpg_available": False},
                                    "PGP armor present but gpg binary missing; cannot verify")]
            sig_file = art.path.with_suffix("")
            if sig_file.exists():
                ts_dir = os.environ.get("VERISAFE_GPG_TRUSTSTORE") or None
                ut_dir = os.environ.get("VERISAFE_GPG_KNOWN_DIR") or None
                v = gpg_check.verify_with_truststore(
                    art.path.read_bytes(), sig_file,
                    truststore_dir=ts_dir,
                    untrusted_dir=ut_dir,
                    workdir=ctx.quarantine_root / "gpg")
                signals = {
                    "format": "pgp",
                    "valid": v.valid,
                    "good_key": v.valid is True,
                    "key_fingerprint": v.fingerprint,
                    "signer_fingerprint": v.fingerprint,
                    "signer_trusted": v.trusted,
                }
                if v.valid is True and v.trusted is True:
                    return [CheckResult("digital_signature", "mid", "ok", signals,
                                        "PGP signature verified against configured truststore")]
                if v.valid is True:
                    return [CheckResult("digital_signature", "mid", "degraded", signals,
                                        "signature cryptographically valid but signer not in configured truststore")]
                if v.valid is False:
                    return [CheckResult("digital_signature", "mid", "failed", signals,
                                        "PGP signature INVALID — treat as untrusted")]
                return [CheckResult("digital_signature", "mid", "degraded", signals,
                                    f"PGP signature unverifiable ({v.err})")]
        return [CheckResult("digital_signature", "cheap", "skipped", {},
                            "no signature object detected for this format")]

    def _run_pades(self, pades_mod, raw: bytes, has_sig: bool) -> list[CheckResult]:
        """Cryptographic verification of embedded PDF /Sig CMS objects.

        Returns one CheckResult per parsed signature plus, optionally, a
        single 'unavailable'-style record when NO parseable CMS payload was
        found in the raw bytes (compressed object streams etc.). The
        ``valid`` key is tri-state by contract: True = digest+RSA+trust-
        anchored; False = any cryptographic failure; None = unverifiable.
        Fusion maps digital_signature.valid through KIND_NEG_BOOL, so a
        None (absent-from-dict semantics: only set when known) never fires
        the -4.0 negative weight. Never raises.
        """
        try:
            sigs = pades_mod.pdf_signature_contents(raw)
        except Exception:  # noqa: BLE001
            return [CheckResult("digital_signature", "mid", "degraded",
                                {"has_sig_object": bool(has_sig), "format": "pdf"},
                                "signature object present but CMS extraction failed")]
        if not sigs:
            return [CheckResult("digital_signature", "mid", "degraded",
                                {"has_sig_object": bool(has_sig), "format": "pdf",
                                 "cms_extracted": 0},
                                "signing structure present but no readable CMS "
                                "payload (possibly compressed object stream); "
                                "not yet machine-verifiable")]
        roots = pades_mod.load_trust_store(_DEFAULT_TRUSTSTORE)
        results: list[CheckResult] = []
        for i, blob in enumerate(sigs[:5]):        # cap work per job
            r = pades_mod.verify_cms(blob, roots)
            d_ok, s_ok = r.get("digest_ok"), r.get("rsa_ok")
            chain = r.get("chain")
            if d_ok is False or s_ok is False:
                valid: bool | None = False         # cryptographic failure
            elif d_ok and s_ok and chain == "trusted":
                valid = True
            else:
                valid = None                       # unverifiable (chain/digest)
            detail = {
                "has_sig_object": True, "format": "pdf",
                "sig_index": i, "cms_count": len(sigs),
                "digest_ok": d_ok, "rsa_ok": s_ok,
                "chain": chain, "signer_cn": r.get("signer_cn"),
                "valid": valid,
                **({"error": r["error"]} if r.get("error") else {}),
            }
            if valid is True:
                note = (f"PAdES signature verified (digest+RSA OK, trust store "
                        f"anchor '{r.get('signer_cn')}')")
            elif valid is False:
                fails = [k for k, v in (("digest", d_ok), ("signature", s_ok))
                         if v is False]
                note = ("PAdES signature INVALID — " +
                        "/".join(fails) + " check failed; treat as untrusted")
            else:
                note = ("PAdES signature unverifiable against current trust "
                        "store (no matching CA anchor)")
            results.append(CheckResult("digital_signature", "mid",
                                       "ok" if valid is True else
                                       ("failed" if valid is False else "degraded"),
                                       detail, note))
        return results

    def _qnative_checks(self, art: Artifact, text: str, ctx: JobContext, doc_type: str | None) -> list[CheckResult]:
        """QR-native + structural native validations (offline, no network)."""
        out: list[CheckResult] = []
        # e-KYC XML docs: canonical self-attestation + hash check
        if doc_type == "aadhaar_ecyc" or (art.original_filename or "").lower().endswith(".xml"):
            ekyc = _parse_ekyc(text)
            if ekyc:
                ok_hash = ekyc["sha1"] is None or (ekyc["declared_sha1"] == ekyc["sha1"])
                out.append(CheckResult("qr_native_check", "mid", "ok",
                                       {"ekyc_fields_present": sorted(ekyc.keys()),
                                        "sha1_matches_declaration": bool(ok_hash),
                                        "aadhaar_masked": ekyc["masked_number"]},
                                       "e-KYC structure validated; SHA1 "
                                       + ("matches embedded declaration" if ok_hash else "DOES NOT MATCH — possible tampering")))
                return out
        # UPI VPA + IFSC sanity (common scam vector)
        vpas = re.findall(NUM_PATTERNS["upi_vpa"], text)
        ifs = re.findall(NUM_PATTERNS["ifsc"], text)
        vpa_ok = [v for v in set(vpas) if _vpa_sane(v)]
        bad_vpa = set(vpas) - set(vpa_ok)
        ifcs_ok = [i for i in set(ifs) if _ifsc_sane(i)]
        if vpas or ifs:
            out.append(CheckResult("financial_field_validation", "mid", "ok",
                                   {"upis_found": sorted(set(vpas)), "valid_upis": vpa_ok,
                                    "invalid_upis": sorted(bad_vpa),
                                    "ifsc_codes": sorted(set(ifcs_ok))},
                                   "found UPI/IFSC references; invalid ones often indicate tampered payment instructions"))
        return out

    def _digilocker(self, art: Artifact, text: str, ctx: JobContext) -> list[CheckResult]:
        key = os.environ.get("VERISAFE_DIGILOCKER_KEY")
        if not key:
            return [CheckResult("digilocker_verify", "mid", "unavailable", {},
                                "DigiLocker credential not provisioned; skip authoritative DigiLookup")]
        url = os.environ.get("VERISAFE_DIGILOCKER_URL", "https://apis.digilocker.gov.in/dl/v1/verDoc")
        payload = {"dl_docid": _find_docid(text)}
        if not payload["dl_docid"]:
            return [CheckResult("digilocker_verify", "mid", "skipped", {},
                                "no DigiLocker document id present in the document text")]
        import urllib.request
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "x-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read().decode())
            valid = bool(data.get("success") or data.get("status") in ("SUCCESS", "VERIFIED"))
            notes = ("DigiLookup verified the document id against UIDAI records"
                     if valid else "DigiLookup returned a NON-verification response")
            return [CheckResult("digilocker_verify", "mid", "ok",
                                {"dl_verified": bool(valid),
                                 "dl_response_code": str(data.get("statusCode") or data.get("status"))},
                                notes)]
        except Exception as e:  # noqa: BLE001
            return [CheckResult("digilocker_verify", "mid", "failed",
                                {"error_class": e.__class__.__name__},
                                "DigiLookup call failed; other evidence still counts")]

    def _api_setu(self, text: str, ctx: JobContext) -> list[CheckResult]:
        key = os.environ.get("VERISAFE_APISETU_TOKEN")
        if not key:
            return [CheckResult("api_setu_lookup", "heavy", "unavailable", {},
                                "API Setu token not provisioned; official-API aggregation skipped")]
        import urllib.request
        base = os.environ.get("VERISAFE_APISETU_BASE", "https://apisetu.gov.in/api/v1")
        # find which service could answer based on detected doc type
        endpoint_map = {"pan_card": "/pan/status", "epf_statement": "/epf/membership"}
        ep = endpoint_map.get(_dt_of(text) or "", None)
        if not ep:
            return [CheckResult("api_setu_lookup", "heavy", "skipped", {},
                                "no applicable official API for this document class in current catalog")]
        req = urllib.request.Request(base + ep, headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read().decode())
            return [CheckResult("api_setu_lookup", "heavy", "ok",
                                {"matched_service": ep, "records_found": len(data.get("data", [])) if isinstance(data, dict) else "?"},
                                f"official {ep} service answered")]
        except Exception as e:  # noqa: BLE001
            return [CheckResult("api_setu_lookup", "heavy", "failed",
                                {"error_class": e.__class__.__name__, "endpoint": ep},
                                "API Setu call failed")]

    def _official_web_check(self, ctx: JobContext, doc_type: str | None, auth: str | None) -> list[CheckResult]:
        """Controlled access to official sites: Playwright first (gated), else
        SSRF-guarded urllib GET against allowlist only. Never arbitrary URLs."""
        if not ctx.browser_available and not OFFICIAL_DOMAINS:
            return [CheckResult("official_site_reachability", "heavy", "unavailable", {},
                                "no official-domain catalog configured")]
        target = _pick_official_domain(doc_type, auth)
        if not target:
            return [CheckResult("official_site_reachability", "heavy", "skipped", {},
                                "no matching official domain for detected issuer")]
        if ctx.browser_available:
            html = _playwright_get(target)
            method = "playwright_isolated"
        else:
            html = _guarded_http_get(target)
            method = "urllib_guarded"
        if html is None:
            return [CheckResult("official_site_reachability", "heavy", "degraded",
                                {"target": target, "method": method},
                                "official site unreachable right now; verdict relies on other evidence")]
        reachable = True
        has_verification_portal = any(k in html.lower() for k in (
            "verify", "validate", "check your ", "search application", "online services"))
        return [CheckResult("official_site_reachability", "heavy", "ok",
                            {"target": target, "reachable": reachable,
                             "has_public_verification_portal": bool(has_verification_portal),
                             "method": method},
                            "official site responded; portal for public verification "
                            + ("present" if has_verification_portal else "not identified in HTML head"))]

    def _rag_cache(self, text: str, ctx: JobContext, doc_type: str | None, auth: str | None) -> list[CheckResult]:
        """Versioned verified-template cache = RETRIEVAL CACHE ONLY."""
        from verisafe import rag_cache as rc
        index = rc.load(self.rag_cache_path)
        if not index:
            return [CheckResult("rag_template_cache", "cheap", "skipped",
                                {"cache_version": self.rag_version},
                                "no local template cache yet; nothing to compare against")]
        # Cache-level freshness gate (14d TTL + source-digest match): a stale
        # cache drops its confidence contribution SILENTLY — same surface as
        # an absent cache, never an error. evidence_gap token lets the report
        # layer name the drop on its evidence_missing line.
        cache_fresh, stale_reason = rc.cache_freshness(index)
        if not cache_fresh:
            return [CheckResult("rag_template_cache", "cheap", "skipped",
                                {"cache_version": self.rag_version,
                                 "cache_stale": True,
                                 "stale_reason": stale_reason,
                                 "evidence_gap": "gov-template-cache-stale"},
                                f"template cache failed freshness gate ({stale_reason}); "
                                "retrieval-cache signal dropped")]
        entry = rc.template_for(doc_type, index)
        if not entry or str(entry.get("version")) != str(self.rag_version):
            return [CheckResult("rag_template_cache", "cheap", "skipped",
                                {"cache_version": self.rag_version},
                                "no cached template for this doc type/version")]
        # Freshness: a stale entry DEGRADES confidence, never blocks.
        stale = not rc.freshness(entry, index=index)
        # cache-consistency: does extracted text resemble the known-good shape?
        kws = entry.get("required_fields", [])
        low = text.lower()
        hit = [k for k in kws if k in low]
        ratio = len(hit) / max(1, len(kws))
        if stale:
            status = "degraded"
            note = (f"layout signal STALE (cache older than staleness window; "
                    f"{ratio:.0%} required fields present) — reduced confidence")
        else:
            status = "ok"
            note = (f"layout matches previously-verified template "
                    f"({ratio:.0%} required fields present)") if ratio >= 0.6 else \
                   (f"layout DEVIATES from known-good template "
                    f"(only {ratio:.0%} required fields matched) — raise scrutiny")
        return [CheckResult("rag_template_cache", "cheap", status,
                            {"required_fields_matched": ratio,
                             "matched_fields": hit[:10],
                             "cache_version": self.rag_version,
                             "stale": stale,
                             "source_of_truth": False},
                            note + " [retrieval-cache signal only, not authoritative]")]


# ------------------------------------------------------------ helpers ----

def _find_docid(text: str) -> str | None:
    m = re.search(r"dl[-_]?docid[\"']?\s*[:=]?\s*[\"']?([0-9A-Za\-]{8,})", text, re.I)
    return m.group(1) if m else None


def _vpa_sane(v: str) -> bool:
    local, _, dom = v.partition("@")
    return 2 <= len(local) <= 15 and 2 <= len(dom) <= 15 and local.count(".") <= 1


def _ifsc_sane(code: str) -> bool:
    # standard: 4 letters + 0 + 6 alnum; bank codes start non-digit
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", code.upper())) and not code[4:5].isdigit()


def _kp_from(gpg_bin: str, good: bool) -> str | None:
    try:
        p = subprocess.run([gpg_bin, "--list-keys", "--with-fingerprints"],
                           capture_output=True, text=True, timeout=20)
        m = re.search(r"^\s+[0-9A-F]{40}$", p.stdout, re.M)
        return m.group(1) if m else None
    except Exception:
        return None


def _parse_ekyc(text: str) -> dict | None:
    if not text or "<" not in text[:80]:
        return None
    def g(tag: str):
        m = re.search(rf"<\s*{tag}\s*>(.*?)</\s*{tag}\s*>", text, re.I | re.S)
        return m.group(1).strip() if m else None
    uid = g("Uid")
    if not uid:
        return None
    sha1 = g("Hash")
    declared = g("SelfDeclarationSha1") or g("Signature")
    masked = (uid[:4] + "XXXXXX" + uid[-4:]) if len(uid) >= 8 else uid
    keys_present = [k for k in ("Name", "Gender", "DOB", "AddressLine1", "PostalCode")
                    if g(k)]
    return {"sha1": _try_sha1(bytes(uid, "ascii")),
            "declared_sha1": sha1,
            "aadhaar_masked": masked,
            **{f"{k}_present": bool(g(k)) for k in keys_present}}


def _try_sha1(b: bytes) -> str | None:
    import hashlib
    return hashlib.sha1(b).hexdigest()


def _dt_of(text: str) -> str | None:
    t, _, _, _ = identify_document(text)
    return t


def _pick_official_domain(dt: str | None, auth: str | None) -> str | None:
    auth_map = {
        "UIDAI": "digilocker.gov.in",
        "Income Tax Department": "incometax.gov.in",
        "RBI": "rbi.org.in",
        "EPFO": "epfindia.gov.in",
    }
    if auth and auth in auth_map:
        return auth_map[auth]
    if dt == "voter_id":
        return "voter.gov.in"
    if dt == "passport":
        return "passportindia.gov.in"
    if dt == "pm_kisan_letter":
        return "pmkisan.gov.in"
    return None


def _playwright_get(url: str) -> str | None:
    """Isolated-profile Playwright fetch (gated by browser_available elsewhere)."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        return None
    try:
        with sync_playwright() as pw:
            b = pw.firefox.launch(headless=True)
            page = b.new_context(ignore_https_errors=False,
                                 user_agent="Mozilla/5.0 (compatible; VeriSafeAudit/1.0)").new_page()
            page.goto(url, timeout=20_000)
            html = page.content()[:200_000]
            b.close()
            return html
    except Exception:
        return None


def _guarded_http_get(url: str) -> str | None:
    """SSRF-guarded GET limited to https + allowlisted hostnames only."""
    import socket
    import urllib.request
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme != "https" or u.hostname not in OFFICIAL_DOMAINS:
        return None
    ip = _resolve_public_only(u.hostname)
    if not ip:
        return None
    req = urllib.request.Request(f"https://{u.hostname}/",
                                 headers={"User-Agent": "Mozilla/5.0 (compatible; VeriSafeAudit/1.0)"})
    try:
        import ssl
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            return r.read(200_000).decode("utf-8", "ignore")
    except Exception:
        return None


def _resolve_public_only(hostname: str) -> bool:
    """Reject private/reserved ranges before any request is made."""
    import ipaddress
    import socket
    try:
        infos = socket.getaddrinfo(hostname, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
            return False
    return True
