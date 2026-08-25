# Verification & Security Stack — Research & Evidence Note (P1)

**Vishwas** | generated 2026-08-19 by orchestrator direct verification. Subagent batch
deleg_02086fd4 (task 2) died on an upstream 504 before writing its note; every item below
was re-verified by the orchestrator against primary sources (GitHub, PyPI JSON API,
live endpoint probes). All claims carry their source inline; anything not verifiable is
marked **UNVERIFIED**.

**Re-verified 2026-08-19 (second pass):** subagent deleg_760fb613 (tooling fact-check)
finished all 12 research calls but 429'd on final delivery; salvage recovered its leads
and the orchestrator re-probed every affected item directly against primary sources
(GitHub API, PyPI JSON, arXiv API, HuggingFace API, OpenWA README). Deltas are noted
inline in §3–§5; items whose facts are unchanged carry no delta marker.

## 1. Document extraction

| Tool | Verified fact | Source |
|---|---|---|
| **docling** (IBM) | `docling` v**2.120.3**, requires-python >=3.10,<4.0 — "SDK and CLI for parsing PDF, DOCX, HTML, and more, to a unified document model". Repo `IBM/docling` (raw README probe inconclusive at verify time; PyPI record authoritative) | https://pypi.org/pypi/docling/json |
| **marker-pdf** | v**2.0.0**, python >=3.10,"convert documents to markdown with high speed and accuracy" — the fast Markdown-first alternative | https://pypi.org/pypi/marker-pdf/json |
| **olmocr** | v**0.4.27**, python >=3.11 — VLM-based OCR ("open visual language model") | https://pypi.org/pypi/olmocr/json |
| tesseract | binary NOT installed on box; PyPI name `tesseract` is a *different* package (astronomy) — use `pip pytesseract` only if system tesseract present | local scan 2026-08-19 |

**Verdict:** primary = docling (layout analysis + Indic-script OCR via Tesseract engines
underneath; degrades to pytesseract when full model absent). Fallbacks: marker-pdf
(speed), olmocr (quality when GPU later available). On this CPU-only box expect
**several seconds per page** — budgeted into the tier-2 timebox in P8.

## 2. Indian government document verification

Live probe results (HTTP status seen from this machine, 2026-08-19):

| Portal | Status | Notes |
|---|---|---|
| **API Setu** https://apisetu.gov.in/ | **200 OK** | aggregator of central/state govt APIs exists. `/apiDocs` path 404'd — docs live at https://apidocs.apisetu.gov.in (UNVERIFIED-path detail; the root is alive and our gov_document module treats API Setu as *discovery layer*, probing its public catalog JSON rather than hardcoding routes) |
| **DigiLocker** https://www.digilocker.gov.in/ | **200 OK** | No public developer API documented found in this pass → DigiLocker flows stay behind the app-level QR/sig checks; our module uses it as an official-portal fallback target, not an API dependency |
| **UIDAI** https://uidai.gov.in/ | **200 OK** | Aadhaar verification is service-provider-gated (no self-service citizen API) — we do **not** attempt Aadhaar number lookups; only format/consistency checks (documented limitation) |
| **Income Tax / PAN** https://www.incometax.gov.in/ (root 200; /pancard/ 503 at probe time) | mixed | NSDL's pcc.nin.gov.in unreachable from this network; treat both as Playwright-fallback targets, never hard deps |
| **PM-KISAN** https://pmkisan.gov.in/ | **200 OK** | benefits-check pages exist; candidate for the Playwright official-site fallback list |
| Ayushman (app.pmjay.gov.in) | URLError (TLS/SNI from this box) | listed as fallback-only; failure mode is graceful "couldn't reach" |

Digital-signature validation (`.p7s`/signed PDF): pypdf signature-object parse + OpenSSL
chain building works with bundled trust anchors; Indian CA roots (eMudhra/KCA/NTRO)
are obtainable from CRL/OCSP endpoints published by each CA — implementation in
`gov_document.py` validates cert chain, timestamps, and declared signer against the
extracted text, with OCSP treated as advisory (many Indian CAs have flaky OCSP).

**Ranked automation verdict:**
1. *Automate today:* magic-byte/format validation, field-schema extraction (Docling/OCR),
   cross-field consistency (dates, amounts, PAN-format, EPFO UAN format), declared-vs-extracted
   digest comparison, digital-signature chain where present.
2. *Playwright fallback:* PAN/land-reports/PMS-KISAN check pages (official portals, no login for
   format-level checks).
3. *Impossible without credentials/account:* Aadhaar, passport-application status,
   individual land-record downloads — reported honestly to user as "verify in person / at the office".

## 3. Malware-analysis stack (CPU-only, 15GB laptop)

| Tool | Verified identity | Install | License | Verdict on this box |
|---|---|---|---|---|
| **ClamAV** | **second pass:** canonical repo now `Cisco-Talos/clamav` (★7148, GPL-2.0, pushed 2026-08-18; **release clamav-1.5.4, 2026-08-07** → actively maintained 2026). The `clamav` **PyPI package is DEAD (v0.2, ~2014)** — do not `pip install clamav`; integrate via distro packaging / `clamdscan --fdump` JSON or clamd REST only | `apt install clamav clamav-daemon` (Ubuntu 24.04), `freshclam --all`, `clamdscan --json` | GNU GPL-2+ | **USABLE-NOW** (install pending; current box lacks it → gate reports unavailable until apt run) |
| **YARA-X** | repo now **`VirusTotal/yara-x`** (org renamed), ★1254, **BSD-3-Clause**, pushed 2026-08-17; Python bindings **`yara_x` v1.19.0** on PyPI (py>=3.9) | `pip install yara_x` (wheels ship Rust lib; no build needed) | BSD-3 | **USABLE-NOW after pip** — classic-YARA rule compatibility is partial (x-expressions differ); bundle conservative ruleset, UNVERIFIED full compat matrix |
| **MobSF** | real repo **`MobSF/Mobile-Security-Framework-MobSF`**, ★21610, **GPL-3.0**, active (pushed 2026-08-17) | Docker image `mobsec/mobile-security-framework` (tag unpinned here); headless CLI mode exists (`python3 mobsf/static-analyzer/main.py`) | GPL-3.0 | **NEEDS-DOCKER** for full static+dynamic; static analyzer alone runs from pip+deps (Java 21 present on box — JADX-backed decompile OK). 15GB RAM adequate for static only; dynamic emulator would thrash this box |
| **jadx** | **skylot/jadx** ★50119 **Apache-2.0** (pushed 2026-08-05) | standalone jar: `https://github.com/skylot/jadx/releases/latest` → `java -jar jadx-cli.jar -d out apk.apk` | Apache-2.0 | **USABLE-NOW** (java 21 present) |
| **apktool** | **iBotPeaches/Apktool** (repo live; name differs from old "Apktools" guess) | maven jar / sdkmanager-free decompile: `java -jar apktool.jar d apk.apk -o out/` | MIT | **USABLE-NOW** |
| **Quark-Engine** | **second pass:** canonical repo is **`ev-flow/quark-engine`** (GitHub 301 chain quarkslab→quark-engine→ev-flow; ★1710, GPL-3.0, pushed 2026-08-15) — direct `quarkslab/` fetch now 404s. PyPI **`quark-engine` v26.8.1** (py≥3.10, CalVer since v25.x, ~monthly releases), "Obfuscation-Neglect Android Malware Scoring System" | `pip install quark-engine` + rules download per its Quick Start | GPL-3.0 | NEEDS-SETUP; optional tier-3 signal behind a gate. Score semantics: higher = more malware-like |
| **CAPE** | **kevoreilly/CAPEv2** ★3433, license **NOASSERTION** (custom; effectively AGPL-ish — read before commercial deployment), pushed 2026-08-17 | docker-compose heavy (VM per sandbox) | custom | **TOO-HEAVY for 15GB laptop** → fallback: firejail mini-sandbox (gated) + strace behavior profiling script (strace present on box) |
| **PE static** | `pefile` **2024.8.26** (py>=3.6), `lief` 1.0.0 on PyPI (version string as served by index); `pydasm` **NOT on PyPI** (source-build only) | `pip install pefile lief` | BSD/Boost-ish | **USABLE-NOW after pip** — import-table heuristics, entropy scan, section anomalies all pure-Python |

**Chosen stack (one primary + fallback per family):**

| Family | Primary | Fallback | Justification |
|---|---|---|---|
| AV signatures | ClamAV (apt) | hash lookup vs VirusTotal (key-gated) | free DBs, zero-runtime-deps, deterministic |
| YARA rules | yara_x (pip) | skip gracefully | BSD, prebuilt wheels, modern perf |
| APK decompile | jadx (standalone jar) | apktool | Apache-2.0, one-shot CLI, fast |
| Android risk scoring | MobSF static analyzer (pip/deps) | Quark-Engine gate | GPLv3 isolation contained; static-only on this box |
| Dynamic PE behavior | strace + firejail mini-sandbox (script in repo) | skip with evidence gap recorded | CAPE too heavy; behavioral deltas still informative |
| PE static | pefile+lief | VT file-hash API | deterministic, instant, no native build |

## 4. Phishing / URL analysis

- **VirusTotal API v3**: auth header `x-apikey`; free-tier rate limits change over time —
  commonly quoted as **4 req/min, 450 queries/day** (marking **UNVERIFIED-this-pass** since the
  console didn't expose limits without auth; our code reads conservatively and caches). Response includes
  `last_analysis_stats` (positives count), `category`, `meaningful_name`, plus
  `whois`/`tls_certificate` objects on domain-related endpoints. **Second-pass correction:** the API is
  **v3** (`https://www.virustotal.com/api/v3/`) — there is **no public v4**; v4 references float around in
  blog posts only. Host resolves and answers 404-without-auth as expected; `api.virustotal.com` does not
  resolve at all from this network (not a real host).
- **PhishLLM**: **NOT FOUND as described.** Second pass probed four indexes with zero
  legitimate hits: **HuggingFace** (no model or dataset with "phishllm" in the ID), **arXiv** (the string
  appears only inside an unrelated E-PhishGen paper), **PyPI** (absent), **GitHub** (only two 0-star
  personal/coursework repos, e.g. `nbaliyan260/phishllm`). There is no open-weight PhishLLM model, no
  published accuracy table, and no serving recipe to build from — anyone advertising one is pointing at a
  placeholder or a private model. Our `url_phishing.py` gates any LLM-classifier behind
  `VISHWAS_PHISHLLM_WEIGHTS` **as a future slot** (a genuine phishing-LLM weights drop could land there
  later); today that path is inert by design, and the always-on tier is the offline DOM-signal set below.
  Nearest *real* open lines if a model ever gets provisioned: LLaMA/GPT-family fine-tunes from the
  phishing-email-detection literature (e.g. arXiv `2512.10104`, `2502.04759`) — each needs its own
  evaluation before being pointed at the env var.
- **Offline page signals implemented (Python stdlib only):** `<form>`/input inventory incl.
  password-like fields, link graph breadth, brand-name string matches, typosquat char-map
  distance, mixed-content/TLS anomaly flags — these feed fusion directly, no JS needed.

## 5. OpenWA (verified 2026-08-19 — replaces earlier guesses)

**Second-pass re-verification (same day):** GitHub API confirms `rmyndharis/OpenWA` ★12944,
**MIT**, TypeScript (NestJS), pushed 2026-08-18, **latest release v0.21.0 (2026-08-18)** — matches
the pinned image in `docs/DEPLOYMENT.md`. Its published `openapi.json` (title "OpenWA API", version
0.21.0) confirms: securitySchemes = `X-API-Key` (+ `metrics-bearer` for metrics only); webhook
registration `POST /api/sessions/{sessionId}/webhooks` body props = `url` (**required**), `events`,
`filters`, `headers`, `retryCount`, `secret`; plus `GET/PUT/DELETE …/webhooks/{id}` and a
`POST …/webhooks/{id}/test` endpoint. `fntlab/openwa` does not exist (404) — the subagent's original
candidate name in the dispatch brief was wrong; rmyndharis is the one. README guidance worth quoting
in ops docs: use the built-in `RATE_LIMIT_*` limiter ("a few messages/min/session sustainable"),
prefer opted-in recipients, mind datacenter-IP flagging (per-session proxy supported). MCP tool
surface exists (`MCP_ENABLED=true` → 25 read-only tools at `POST /mcp`) — not used by Vishwas.

- **Repo:** `rmyndharis/OpenWA` (NOT open-wa/openwa*; those names don't exist),
  ★≈13k, **MIT**, Node 22 LTS, NestJS, Docker-native.
- **Real REST surface (from the project's published `openapi.json`, v0.21.0, 157 paths)** — the
  Vishwas-relevant subset:
  - send text: `POST /api/sessions/{sessionId}/messages/send-text` body `{chatId:"E.164@c.us", text}`
  - inbound media bytes: `GET /api/sessions/{sessionId}/messages/{chatId}/{messageId}/media` → application/octet-stream
  - webhooks (per session): `POST /api/sessions/{sessionId}/webhooks` with `{url, events:[...], secret?, headers?}`
  - delivery envelope: `{"event":"message.received","timestamp":ISO,"sessionId":..,"idempotencyKey":"msg_{session}_{msgid}","deliveryId":"dlv_..","data":{ id, from, to, body, type, timestamp(epoch-sec), isGroup, kind, hasMedia, contact? }}`
  - HMAC: header **`X-OpenWA-Signature: sha256=<hex>`** over the RAW request body (constant-time compare);
    idempotency header **`X-OpenWA-Idempotency-Key`** (stable across retries — dedupe on it)
  - events catalog: message.received / message.sent / message.ack / message.failed / session.qr / session.status …
  - security scheme: **X-API-Key** (role-based OPERATOR/ADMIN)
  - big caveat from their own README: unofficial client (whatsapp-web.js or Baileys) → **non-zero ban
    risk**; use a dedicated number; media above 1MiB is *omitted from webhook payloads* (marker object)
    and must be fetched via the media GET above.
- **Alternatives** (comparison kept short on purpose — OpenWA is fit and maintained, no need to replace):
  WPPConnect (WA Web-based, TS), Baileys (pure-protocol, lower-level). For regulated use the official
  Meta Cloud API is the compliant option; OpenWA explicitly says so.

## 6. SSRF-safe fetch + browser

Pattern shipped in `url_guard.py`: resolve host → pin IP (reject private/loopback/link-local/CGNAT
ranges incl. IPv6 equivalents) → connect to pinned IP with Host header → validate redirect
targets against the same guard before following. Timeout + size caps on every hop.
**Playwright**: `playwright` **v1.62.0** on PyPI (py>=3.10); box has Firefox (no Chromium) →
`playwright install-deps firefox` + `playwright install firefox` gives a headless runner
(~300–700MB disk) — gated behind availability; isolated profile dir under quarantine, purged
after the job.

## UNVERIFIED / GAPS
1. Effort / VB+StA / Fake-Mamba papers — see DEEPFAKE_DETECTION.md §0.
2. ~~Quark-Engine repo move~~ **RESOLVED (second pass):** canonical `ev-flow/quark-engine`; PyPI `quark-engine` v26.8.1 is the install path.
3. VirusTotal "v4" — **resolved as non-existent** (public API is v3 at www.virustotal.com/api/v3); only the *exact* free-tier quotas remain unverified without an authenticated console.
4. **PhishLLM — NOT FOUND (confirmed second pass):** no paper, model, dataset, or package by that name anywhere probed. Treated in code as a *future* weights slot, never a real detector.
5. ~~DigiLocker developer-API specifics~~ **RESOLVED → see `INDIA_GOV_VERIFICATION.md` §1:** old developer hosts dead (NXDOMAIN); e-KYC access is partnership-gated (OAuth per UIDAI/DigiLocker partner agreement) — NO-PUBLIC-API for self-serve; Vishwas keeps user-assisted QR flow + discovery-only posture.
6. ~~Apisetu sub-docs path~~ **RESOLVED → see `INDIA_GOV_VERIFICATION.md` §2:** legacy `apisetu.api.gov.in`/`apisetu.dev.gov.in` NXDOMAIN; live catalog = `directory.apisetu.gov.in` (`/api/list?q={"query":…}` + single-use bearer from `/api/auth/generate-headers`; Meilisearch `apidirectory_v2`, 2930 orgs). Read-only discovery is account-free; consumer invocation needs partners.apisetu.gov.in OAuth.
7. CAPE license "NOASSERTION" — review before any non-personal deployment distribution.
