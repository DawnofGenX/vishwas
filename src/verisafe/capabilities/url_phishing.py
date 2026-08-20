"""URL / phishing capability.

Progressive design:
  T0 cheap : normalization + SSRF guard + fetch + redirect chain + TLS basics
  T1 mid   : VirusTotal reputation (key-gated), offline DOM signal kit
             (forms/password fields, link-graph, brand impersonation,
              typosquat) on the fetched HTML
  T2 heavy : PhishLLM (weights-gated)
Downloaded file artifacts are re-validated through FileValidator and handed
back to the malware flow (cycle-guarded via ctx.extra['loop_depth']).
"""
from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from typing import Any

from ..events import Artifact, JobContext
from ..url_guard import NormalizedUrl, normalize_url, fetch_url, typosquat_signals
from .base import CheckResult

_BRAND_LEXICON = [
    "google", "paypal", "amazon", "flipkart", "swiggy", "zomato",
    # Indian banks / government (high-fraud brands)
    "icici", "hdfcbank", "sbi", "axisbank", "kotak", "yesbank",
    "upi", "netbanking", "aadhaar", "uidai", "incometax", "irdai",
    "epfo", "nsdl", "passport", "ibps", "sebi", "reservebankofindia",
]

_PASSWORD_FIELD_RE = re.compile(r"(?i)(password|pin\b|cvv|otp\s*code|one[- ]time|credit card number|card number)")


class _FormCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.forms = 0
        self.inputs = 0
        self.password_like_inputs = 0
        self.links = 0
        self.link_hosts: dict[str, int] = {}
        self.title = ""
        self._in_title = False
        self._title_parts: list[str] = []
        self.external_form_post = False
        self.form_actions: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form":
            self.forms += 1
            act = (a.get("action") or "").lower()
            if act.startswith("http"):
                self.form_actions.append(act)
                self.external_form_post = True
        elif tag == "input":
            self.inputs += 1
            itype = (a.get("type") or "text").lower()
            blob = f"{a.get('name','')} {a.get('placeholder','')} {a.get('autocomplete','')}".lower()
            if itype == "password" or any(k in blob for k in ("passw", "pwd", "cvv", "cardnum")):
                self.password_like_inputs += 1
            elif _PASSWORD_FIELD_RE.search(blob) and itype in ("text", "tel", "number"):
                self.password_like_inputs += 1
        elif tag == "a":
            href = a.get("href") or ""
            if href:
                self.links += 1
                m = re.match(r"https?://([^/:]+)", href, re.I)
                if m:
                    h = m.group(1).lower()
                    self.link_hosts[h] = self.link_hosts.get(h, 0) + 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self._title_parts.append(data.strip())


def analyze_dom(html: bytes, page_host: str) -> dict[str, Any]:
    """Offline DOM signal kit — no JS, pure Python, fully deterministic."""
    p = _FormCollector()
    try:
        p.feed(html.decode("utf-8", errors="replace"))
    except Exception:
        pass
    p.close()
    title = " ".join(p._title_parts)[:160]
    ext_links = sum(n for h, n in p.link_hosts.items() if not host_eq(h, page_host))
    textblob = html[:400_000].decode("utf-8", errors="ignore").lower()
    brand_hits = [b for b in _BRAND_LEXICON if re.search(rf"\b{re.escape(b)}\b", textblob)]
    return {
        "forms": p.forms,
        "inputs": p.inputs,
        "password_like_inputs": p.password_like_inputs,
        "links": p.links,
        "external_link_share": round(ext_links / max(1, p.links), 3),
        "has_external_form_post": p.external_form_post,
        "host_mismatch_in_forms": any(_form_target_other_host(a, page_host) for a in p.form_actions[:20]),
        "page_title": title,
        "brand_mentions": brand_hits[:12],
        "typosquat_signals": typosquat_signals(page_host, _BRAND_LEXICON),
    }


def host_eq(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return a.endswith(b) or b.endswith(a) or a == b


_AUTH_KEYWORDS = ("login", "secure", "verify", "account", "reset", "otp",
                 "netbanking", "payment", "activate", "update", "customer-care")
_RISKY_TLDS = {".com": False, ".in": False, ".org": False, ".gov.in": False,
               ".co.in": False, ".edu.in": False, ".net.in": False}


def host_level_signals(nurl, brand_lexicon: list[str]) -> tuple[float, list[str]]:
    """Host-string-only phishing score 0..1 — computable WITHOUT fetching the page.

    Catches classic patterns even for dead/unresolvable domains:
      brand+auth-keyword stuffing, deep subdomains, digit-dense names,
      rare TLDs, raw-IP hosts, lookalike brands (delegates to typosquat_signals).
    Returns (score, evidence_notes). Deterministic.
    """
    notes: list[str] = []
    s = 0.0
    host = (nurl.puny_host or "").lower().lstrip("www.")
    domain = (nurl.registrable_domain or "").lower()
    # raw IP host is almost never legit for a service site
    try:
        import ipaddress
        ipaddress.ip_address(host.split(":")[0])
        return 0.8, ["raw IP address as host"]
    except ValueError:
        pass
    # typosquat / brand-plus-extra via shared helper
    typos = typosquat_signals(domain, brand_lexicon)
    for t in typos[:2]:
        if t["signal"] == "typosquat_exact":
            s += 0.6
            notes.append(f"lookalike of brand '{t['brand']}' (character substitution)")
        elif t["signal"] == "brand_plus_extra":
            s += 0.3
            notes.append(f"brand '{t['brand']}' padded with extra tokens")
    # credential-harvest keyword stuffing anywhere in host
    kw_hits = [k for k in _AUTH_KEYWORDS if k in host]
    if kw_hits:
        s += min(0.45, 0.15 * len(kw_hits))
        notes.append("credential-keyword stuffing: " + ", ".join(sorted(kw_hits)[:4]))
    # brand AND keyword combination is the strongest static pattern
    # (letter-bounded so tiny brands like 'sbi' don't fire inside unrelated words)
    def _brand_in(host_s: str, b: str) -> bool:
        return re.search(r"(?<![a-z])" + re.escape(b.lower()) + r"(?![a-z])", host_s) is not None
    brand_in_host = any(_brand_in(host, b) for b in brand_lexicon)
    if brand_in_host and kw_hits:
        s += 0.25
        notes.append("brand name combined with credential keywords")
    # deep subdomain chains
    if host.count(".") >= 3:
        s += 0.1
        notes.append("deep subdomain chain")
    # digit-dense label (e.g. 1cicibank, x9secure)
    labels = host.split(".")
    dense = [lab for lab in labels if len(lab) >= 4 and sum(c.isdigit() for c in lab) >= 2]
    if dense:
        s += 0.15
        notes.append("digit-heavy label: " + dense[0][:24])
    # unusual TLD for a 'bank/gov' page
    if not any(domain.endswith(tld) for tld, _ in _RISKY_TLDS.items()):
        s += 0.05
    return round(min(1.0, s), 3), notes


def _form_target_other_host(action_url: str, page_host: str) -> bool:
    m = re.match(r"https?://([^/:]+)", action_url or "", re.I)
    if not m:
        return False
    return not host_eq(m.group(1), page_host)


def phish_heuristic_score(dom: dict[str, Any]) -> float:
    """Bounded 0..1 blend of offline signals. Deterministic; feeds fusion weights."""
    s = 0.0
    s += min(1.0, 0.5 * dom.get("password_like_inputs", 0))
    if dom.get("has_external_form_post"):
        s += 0.4
    if dom.get("host_mismatch_in_forms"):
        s += 0.2
    s += 0.15 * min(1.0, dom.get("external_link_share", 0.0))
    typos = dom.get("typosquat_signals", [])
    if typos:
        s += 0.5
    if any(t["signal"] == "typosquat_exact" for t in typos) and dom.get("brand_mentions"):
        s += 0.3
    age = dom.get("domain_age_days")
    if isinstance(age, (int, float)) and 0 <= age < 90:
        s += 0.2
    return round(min(1.0, s), 3)


class UrlPhishingCapability:
    """requires=() keeps core always-runnable; VT/PhishLLM gate per-check."""

    requires: tuple[str, ...] = ()

    def __init__(self) -> None:
        from ..file_validator import FileValidator
        self.validator = FileValidator()

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        urls_in_text = ctx.extra.get("urls_in_text") or []
        raw = art.path.read_text(errors="ignore").strip()
        url_str = (urls_in_text or [raw])[0]
        if not url_str:
            out.append(CheckResult("url_normalize", "cheap", "failed",
                                   {"raw": (art.original_filename or "")[:60]},
                                   "no URL found in message"))
            return out
        nurl = normalize_url(url_str)
        if nurl is None:
            out.append(CheckResult("url_normalize", "cheap", "failed",
                                   {"raw": url_str[:120]},
                                   "could not parse as http(s) URL; nothing analyzable"))
            return out
        out.extend(self._cheap(nurl, ctx))
        # downloaded artifact loop (depth-limited)
        loop = int(ctx.extra.get("loop_depth", 0))
        ct_path = ctx.extra.get("_downloaded_content_path")
        if ct_path and loop < 1:
            out.extend(self._revalidate_downloaded(ct_path, ctx))
        # mid tier: only when the cheap tier left doubt (progressive rule)
        conclusive = any(c.name == "vt_url_reputation" and c.usable()
                         and (c.signals.get("positives_ratio") or 0) > 0.5 for c in out)
        if not conclusive:
            out.extend(self._mid(nurl, ctx))
        return out

    # -------------------------------------------------------------- tiers --
    def _cheap(self, nurl: NormalizedUrl, ctx: JobContext) -> list[CheckResult]:
        res: list[CheckResult] = [
            CheckResult("url_normalize", "cheap", "ok",
                        {"scheme": nurl.scheme, "host": nurl.host,
                         "domain": nurl.registrable_domain},
                        duration_s=0.0),
        ]
        qdir = ctx.quarantine_root / "fetched"
        qdir.mkdir(parents=True, exist_ok=True)
        r = fetch_url(nurl, body_cap=2 * 1024 * 1024, note=lambda s: ctx.note(f"url:{s}"))
        if r.block_reason:
            res.append(CheckResult("ssrf_guard", "cheap", "degraded",
                                   {"blocked": 1.0, "reason": (r.block_reason or "")[:80]},
                                   "resolved address is non-public or redirects into blocked range; content NOT fetched"))
        elif r.error:
            res.append(CheckResult("ssrf_guard", "cheap", "ok",
                                   {"reachable": 0.0, "error_class": (r.error or "")[:40]},
                                   f"fetch failed ({r.error}); static signals limited"))
        else:
            (qdir / "page.html").write_bytes(r.body[:4_000_000])
            ctx.extra["_fetched_html_path"] = str(qdir / "page.html")
            susp_hops = 0
            final = normalize_url(r.final_url or nurl.url)
            if final and final.puny_host != nurl.puny_host:
                susp_hops = 1
            sig = {"reachable": 1.0, "status_code": r.status,
                   "tls_version": r.tls.get("version") or "",
                   "redirect_hops": len(r.redirect_chain),
                   "suspicious_hop": susp_hops}
            notes = "" if not r.redirect_chain else \
                "followed: " + " -> ".join((h["to"] or "")[:50] for h in r.redirect_chain[:3])
            res.append(CheckResult("ssrf_guard", "cheap", "ok", sig, notes))
            if susp_hops:
                res.append(CheckResult("url_redirects", "cheap", "degraded",
                                       {"suspicious_hops": susp_hops},
                                       "final host differs from the shared host"))
        return res

    def _mid(self, nurl: NormalizedUrl, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        vt_key = ctx.vt_api_key
        if not vt_key:
            out.append(CheckResult("vt_url_reputation", "mid", "unavailable",
                                   {}, "VirusTotal API key not provisioned; reputation skipped"))
        else:
            out.extend(self._vt(nurl, vt_key))
        # host-string scoring — always computable (no page required)
        host_score, host_notes = host_level_signals(nurl, _BRAND_LEXICON)
        # offline DOM kit
        dom: dict[str, Any] | None = None
        p = ctx.extra.get("_fetched_html_path")
        if p:
            try:
                html = open(p, "rb").read()[:4_000_000]
            except OSError:
                html = b""
            dom = analyze_dom(html, nurl.puny_host)
            dom["domain_age_days"] = _vt_age_hint(ctx)  # may be None until VT runs
        if dom is not None:
            score = phish_heuristic_score(dom)
            # blend: DOM evidence dominates when present; host strings fill gaps
            blended = round(min(1.0, max(score, host_score) + 0.2 * min(score, host_score)), 3)
            out.append(CheckResult("phish_heuristics", "mid", "ok",
                                   {"score_norm": blended,
                                    "host_string_score": host_score,
                                    "dom_score": score,
                                    "forms": dom["forms"],
                                    "password_like_inputs": dom["password_like_inputs"],
                                    "has_external_form_post": dom["has_external_form_post"],
                                    "young_domain": (bool(dom["domain_age_days"]) if isinstance(dom["domain_age_days"], (int, float)) else False),
                                    "suspicious": bool(blended > 0.55)},
                                   (_dom_notes(dom) + "; " if host_notes else "") + ("; ".join(host_notes))))
        else:
            notes_txt = "; ".join(host_notes) or "no page content and no strong host-string signals"
            out.append(CheckResult("phish_heuristics", "mid",
                                   "degraded" if host_score < 0.3 else "ok",
                                   {"score_norm": host_score,
                                    "host_string_score": host_score,
                                    "dom_score": None,
                                    "suspicious": bool(host_score > 0.55)},
                                   notes_txt + (" (page unreachable; scored from URL string only)" if not host_notes else "")))
        # NOTE: a weights-gated phishing-LLM pass was here historically
        # (VERISAFE_PHISHLLM_WEIGHTS). It was removed 2026-08-19 because no
        # open-weight project by that name exists (see docs/research/
        # VERIFY_SECURITY_STACK.md); host-string + DOM heuristics carry.
        return out

    def _vt(self, nurl: NormalizedUrl, api_key: str) -> list[CheckResult]:
        from .. import vt_client
        res = vt_client.check_url(nurl.url)
        if res.status == "ok" and res.counts:
            total = sum(res.counts.values()) or 1
            ratio = (res.counts.get("malicious", 0) + res.counts.get("suspicious", 0)) / total
            notes = "VirusTotal reports positive detections" if ratio > 0 \
                else "no positive detections in latest scan set"
            return [CheckResult("vt_url_reputation", "mid",
                                "ok" if total > 1 else "degraded",
                                {"positives_ratio": round(ratio, 3),
                                 "vt_total_engines": total,
                                 "category": res.category,
                                 "vt_verdict": res.verdict},
                                notes)]
        if res.status == "ok":  # 404 / no stats yet — clean negative
            return [CheckResult("vt_url_reputation", "mid", "degraded",
                                {"positives_ratio": 0.0, "vt_total_engines": 0,
                                 "category": "", "vt_verdict": "low"},
                                res.note or "no VT record for this URL")]
        # transport/rate-limit exhaustion -> structured failure (never raises)
        return [CheckResult("vt_url_reputation", "mid", "failed",
                            {"error_class": "VtUnavailable", "vt_note": res.note},
                            "VirusTotal lookup failed; other evidence still counts")]

    def _revalidate_downloaded(self, path_str: str, ctx: JobContext) -> list[CheckResult]:
        """Spec rule: files behind a URL go back through file validation."""
        from pathlib import Path
        from ..file_validator import make_artifact, InputType
        p = Path(path_str)
        if not p.exists():
            return []
        ctx.extra["loop_depth"] = int(ctx.extra.get("loop_depth", 0)) + 1
        sub = make_artifact(ctx.quarantine_root / "downloads", p.name, InputType.FILE)
        sub.path.write_bytes(p.read_bytes())
        kind, mismatch = self.validator.validate(sub)
        flags = _kind_flags(kind.value)
        return [CheckResult("url_download_revalidated", "mid", "ok",
                            {"verified_kind": kind.value, "ext_mismatch": bool(mismatch),
                             "sha256": sub.sha256, **flags},
                            "content behind URL was itself a file; it passed through file validation")]


# ------------------------------------------------------------ helpers ----

def _vt_age_hint(ctx: JobContext) -> float | None:
    for c in getattr(ctx, "_prior", []) if hasattr(ctx, "_prior") else []:
        pass
    return None


_KIND_FILE_FLAG = {
    "pe": {"looks_executable": True},
    "elf": {"looks_executable": True},
    "apk": {"is_apk": True},
    "zip": {"is_archive": True},
    "rar": {"is_archive": True},
    "7zip": {"is_archive": True},
}


def _kind_flags(kind_name: str) -> dict[str, Any]:
    return _KIND_FILE_FLAG.get(kind_name, {"generic_file": True})


def _dom_notes(dom: dict) -> str:
    bits = []
    if dom.get("password_like_inputs"):
        bits.append(f"{dom['password_like_inputs']} password-like input field(s)")
    if dom.get("has_external_form_post"):
        bits.append("form posts to an external host")
    if dom.get("typosquat_signals"):
        bits.append("lookalike-domain signal vs known brands")
    if dom.get("brand_mentions"):
        bits.append("mentions: " + ", ".join(dom["brand_mentions"][:4]))
    if not bits:
        bits.append("no strong harvest indicators in HTML")
    return "; ".join(bits)
