# Vendored: Lintshiwe/PhishingScanner — `phishing_scanner.py`

This directory vendors the MIT-licensed upstream **PhishingScanner** single-file
scanner for the `url_phishing` detection path in Vishwas. Under the operator
decision of 2026-08-26, PhishingScanner **alone** decides URL phishing here —
the previous VirusTotal (`vt_url_reputation`), offline-DOM heuristic
(`phish_heuristics`) and vendored-xgboost (`url_phishml`) signals were dropped
from the url fusion target.

## Source

- **Project:** https://github.com/Lintshiwe/PhishingScanner
- **License:** MIT (see `LICENSE` in this directory, copied verbatim from upstream —
  Copyright (c) 2025 PhishingScanner).
- **Commit:** `7dfd04dfa083ee4a77a2f078d53af9e3e8e312ba` («first commit»), snapshot
  taken 2026-08-26.
- **File(s) vendored here:**
  - `phishing_scanner.py` — the core detection module.
  - `LICENSE` — MIT license text.

## Class / API

The upstream module defines `class PhishingDetector(config_path: str = "config.json")`
with `scan_url(url: str) -> ScanResult`. (Note: upstream names the class
`PhishingDetector`, not `PhishingScanner` — the *project* is called
PhishingScanner. The capability wrapping this uses the check name
`url_phish_scanner`.)

`ScanResult` is a dataclass:
`(url, timestamp, risk_score: int 0..100, is_phishing: bool, indicators: list,
 details: dict, response_time: float)`. `is_phishing = risk_score >= 70`.

`scan_url` does live network work: urlparse structure analysis, whois domain-age
(`python-whois`), SSL cert via `cryptography`, `requests` HTTP fetch, BS4 content
patterns, and `validators.url` validation. Evidence: `analyze_dom` etc. remain in
the module; only the `PhishingDetector.scan_url` path is consumed here.

## Dependencies (runtime, non-optional for `scan_url`)

All imports resolve under the calling process's PYTHONPATH (serving tree is
`/home/hermes/pylibs:/home/hermes/docling-python`):

| dep                     | serving-tree status        |
|-------------------------|----------------------------|
| `requests`              | present (docling-python)   |
| `beautifulsoup4` (`bs4`)| present (docling-python)   |
| `urllib3`               | present (docling-python)   |
| `dnspython` (`dns`)     | present (global `.local`)  |
| `cryptography`          | present (pylibs)           |
| `validators`            | **installed 2026-08-26** → `/home/hermes/docling-python` (0.35.0) |
| `python-whois` (`whois`)| **installed 2026-08-26** → `/home/hermes/docling-python` (0.9.6) |
| `Pillow`, `numpy`, `pandas`, `scikit-learn`, `opencv`, `imagehash` | NOT required for `scan_url` core — the visual/ML tiers are unused; skipped. |

## Local rehome (deliberate deviation)

One import was rehomed for the serving tree: the module-level `import dns.resolver`
was moved **into** `_analyze_domain` (lazy, inside its existing try/except). Reason:
in this sandbox `dns` resolves to the global `~/.local` dnspython whose
`dns.quic -> aioquic -> service_identity -> pyopenssl` chain is broken
(`AttributeError: module 'lib' has no attribute 'GEN_EMAIL'`). A module-level
import crashes the whole scanner at load; DNS analysis is non-critical and the
try/except already degrades a DNS failure to the `details['dns_error']` +10-risk
branch. Net behavior: the scanner imports cleanly and a DNS failure is honest
(`DNS resolution failed` indicator, +10 risk) rather than a hard crash.

No other code was changed.