# Indian Government Document-Verification Infrastructure — Research Snapshot

**Captured:** 2026-08-19 (live probes from this machine; every claim below re-verified against primary sources)
**Provenance:** salvage + independent re-verification of delegation `deleg_7306c63c` (timed out at 50 iterations / Cloudflare 524 before writing its summary). Host DNS claims confirmed via Google DoH; API behavior replayed live; Docling facts pulled from PyPI/GitHub APIs.
**Evidence files:**
- `data/apisetu_catalog_digest_2026-08-19.json` — 461 KB digest of API Setu `/api/list` responses for aadhaar/pan/voter id/udyam/passport/income certificate/epf/ration (orgs, issuerIds, endpoints, per-API samples)
- `/tmp/apisetu_catalog.json` (transient) — raw page-0 of the empty-query catalog dump (Meilisearch shape)

---

## 1. DigiLocker (`digilocker.gov.in`)

**STATUS: NO-PUBLIC-API (self-service web only)**

| Claim (re-verified 2026-08-19) | Evidence |
|---|---|
| `services.digilocker.gov.in`, `developer.digilocker.gov.in`, `app./my./ekyc./share./partner./gateway.` subdomains → **NXDOMAIN** (DoH Status 3) | dead; the old public developer portal no longer exists |
| `www.digilocker.gov.in` → **200**, behind CloudFront (`d1dfndjwdsfipf.cloudfront.net`), custom header `Server: DigiLocker` | live consumer app only |
| `api.digilocker.gov.in`, `verify.digilocker.gov.in` → resolve (AWS NLB `dl-nlb-332ccffa0aa48d8b`, ap-south-1) but their roots **301 to www** | hosts exist but expose no API root — not a public programmatic endpoint |
| `https://www.digilocker.gov.in/share` → **403** to bare curl UA | share/verify flows are web-session oriented (QR + OTP user-in-the-loop); bot-gated |

**What it actually offers (public knowledge, consistent with the site):** citizens share documents by generating a time-limited verification link/QR; third parties *view or verify* through DigiLocker's own UI, where DigiLocker holds the issuing authority's original (e-KYC for Aadhaar, certificates for others). The e-KYC spec (DigiLocker ↔ UIDAI) is documented publicly (`digilocker.gov.in` partner docs), but access is **contract/partnership-gated**: you must be an authorized service provider (DLP/verifier registration with MoHA/UIDAI), get OAuth2/OIDC credentials issued, and pass a due-diligence process. There is **no anonymous self-serve key** like API Setu has for its directory search.

**Implication for VeriSafe:** do NOT build direct DigiLocker calls into the default pipeline. Two viable shapes when a user *has* a DigiLocker QR/link: (a) treat the QR payload as a DigiLocker verification URL → hand the user a plain-language instruction ("open this in your phone's DigiLocker app / verify with your OTP"), i.e. **user-assisted verification**; (b) if VeriSafe later gets partner credentials, a gated capability `digilocker_verify(qr_or_link)` using the published e-KYC flow (OAuth token grant, DigiLocker's JSON eKYC response). Until then: **NO-PUBLIC-API**.

## 2. API Setu / India.gov.in open-data portal

**STATUS: PARTIAL (public READ/discovery API = YES; actual gov-service invocation = registered-consumer OAuth)**

The platform has been **re-homed since pre-2024 docs**: `apisetu.api.gov.in` and `apisetu.dev.gov.in` are **NXDOMAIN** (dead — many older blog posts point at them). Live stack (all re-resolved today via DoH, all AWS ELB ap-south-1):

```
apisetu.gov.in            portal + developer/SOP/marketing pages (Next.js SSR)
directory.apisetu.gov.in  searchable catalog  ← the one VeriSafe needs for discovery
docs.apisetu.gov.in       Mintlify docs ("document-central", explore-apisetu, SOP for API Access)
partners.apisetu.gov.in   consumer/partner signup
```

### Discovery API (replayed live twice today — works without an account)

- **Token:** `GET https://directory.apisetu.gov.in/api/auth/generate-headers` → `{authorization: "Bearer <one-shot>", nonce, ts}`. Token is **single-use** — regenerate immediately before each list call; send `Authorization` + `x-nonce` + `ts` headers. A `Referer: <base>/search` header is sent by the browser (probe worked with it included; treat as required until proven otherwise).
- **Search:** `GET /api/list?q=<urlencoded JSON>&size=N&page=0`. **`q` must be a JSON object.** Free-text field is **`query`** (`{"query":"aadhaar"} → 15 hits`). Wrong field names → 400 with the exact message `"Unexpected field '<name>' in 'q'"` (usable as a live schema oracle).
- **Engine:** Meilisearch, index `apidirectory_v2`, **2930 org-level entries / 147 pages @ size 20**. Response envelope: `{status:"Success", statusCode:200, result:{nbHits, nbPages, hitsPerPage, hits:[…]}}`.
- **Hit fields:** `objectID/orgId, orgName, orgType (State Government|Central…|…), orgState, issuerId (in.gov.<dept>.<name>), categories[], subdomainName, apiCount, docApiCount, Collections[]` where each Collection carries `publishedThrough (e.g. "DigiLocker"), apiType ("Docs"/…), gateway_firewall, Apis[]`.
- **API entry fields:** `apiName, endpoint, available_docs, apiEndpId, spec_code, modified_on` etc.
- **Verified keyword counts (today):** aadhaar 15 · pan 13 · voter id 1 · udyam 2 · passport 4 · income certificate 36 · ration 32 · epf 0. Full per-org sample in the digest file.

### What the endpoints mean

Two families appear in the results:
1. **E-KYC/certificate distribution over DigiLocker** — pattern `/certificate/v3/<subdomain>/<spec>` (e.g. `/certificate/v3/civilsupplies/ratcr`, Passport Seva `/certificate/v3/passportindia/psprt`). These are the *document-type* identifiers of the IndiaStack e-KYC layer: a citizen authorizes release, the verifier pulls signed content. Invoking them requires a registered API-Setu consumer identity.
2. **Plain service APIs** (UMANG-style `/umang/apisetu/dept/…`, department-specific ws1 paths, private vendors' OCR/CKYC suites — e.g. Think Analytics `think360`, Baldor `idfy`: full commercial ID-document OCR/face/liveness offerings).

**Consumer auth model** (from the SOP-for-API-Access doc on docs.apisetu.gov.in — page present but body JS-rendered; corroborated by the directory's `gateway_firewall` flag): register at partners.apisetu.gov.in → get an **OAuth2 client credentials pair** (client_id/secret) → sign requests to the respective `<subdomain>.apisetugateway.in` / `gateway.apisetu.gov.in` endpoints. Rate limits are negotiated per consumer (not published as fixed numbers). Data license: government-published datasets follow India.gov.in data policies; **personal KYC data returned by these APIs may only be processed under the consuming entity's legal authorization and the user's consent** — a WhatsApp-based product must keep the zero-retention + consent posture or it will violate those ToS.

**Best-practice integration pattern for VeriSafe:** use the **read-only discovery surface only** (token → `/api/list?q={"query": …}`) to maintain a versioned cache of *which document types exist, which issuers serve them, and their spec codes*. Never call the e-KYC endpoints anonymously; they'll fail or misbehave without a registered consumer + user-initiated consent. This makes the discovery API our **RAG template-cache refresher** (see §7).

## 3. Docling (`docling-project/docling`)

**STATUS: PUBLIC-API (OSS library — local execution, MIT)**

Re-verified today from PyPI + GitHub APIs:
- Latest **v2.120.3**, `requires_python >=3.10,<4.0`. **MIT** (GitHub repo `docling-project/docling`, LICENSE first line "MIT License"; PyPI `license` field blank — GitHub is the truth here). ★≈65k, pushed daily.
- Install footprint: wheel ~3 MB, but with `docling[full]` (layout models + RapidOCR ONNX) the isolated dir we run lands at **~5.5 GB** — that's why VeriSafe runs it from `/home/hermes/docling-python` with an explicit `PYTHONPATH` gate rather than polluting site-packages. First convert ≈22–34 s (model load), cached after (~1–2 s per page).
- Extraction path used by VeriSafe: `DocumentConverter().convert(input=DocumentStream(name=…, stream=BytesIO(bytes)))` → `.document.export_to_markdown()` (**str**) — see `gov_document.py` step 1b. Tables render as markdown tables; forms are read as structured layout blocks inside the Document object (JSON export available via `.export_to_dict()` if we ever want machine-checked field boxes).
- **OCR bundled:** `rapidocr-onnxruntime` (the `easyocr`/`tesseract` backends are optional extras; tesseract remains our separate branch-3 fallback for scanned images outside docling's success path).
- **QR hints on scanned certificates:** docling can rasterize embedded images but does **not** decode QR itself — QR decoding stays in our planned cv2/zbar branch (item 4). No conflict: same raster, two consumers.

## 4. QR-coding schemes on Indian govt documents

**STATUS: PARTIAL (formats documented publicly; no central registry of every scheme)**

Known schemes (public documentation; each should go into the RAG cache with its source link):
- **Aadhaar (UIDAI):** card front QR encodes **masked** UID (XXXX XXXX 1234 format) — privacy-by-design, not the 12 digits; back-side QR (post-2020 cards) carries masked UID + name. Decodable with any standard QR reader → gives us the *presence* check and partial-number cross-check without holding full PII.
- **Voter ID (EPIC, NIC):** QR encodes EPIC number (10-digit, e.g. ABC1234567) + photo hash (NIC's "photo verification" QR for polling booths).
- **EPFO member card:** QR carries UAN + member name (UAN is 12-digit numeric).
- **PAN (NSDL/UTI):** modern PAN cards carry a QR with PAN + name.
- **DigiLocker certificates & UDYAM:** QR = deep link to the issuing portal's verify page (URL, not JSON) — this is the *user-assisted verification* vector from item 1.
- **IndiaStack e-KYC responses:** signed JSON with base64-encoded PDF/A image; the QR on physical copies typically points to the issuer's verification endpoint.

Formats observed in the wild: mostly **UTF-8 text (plain or JSON)**, occasionally **base64 blobs** wrapping issuer payloads. Recommendation: decoder returns *raw string*; a small rule table (regex per doc class: UAN `^\d{12}$`, PAN `^[A-Z]{5}\d{4}[A-Z]$`, EPIC `^[A-Z]{3}\d{7}$`, masked-Aadhaar `^\d{4} ?\d{4} ?\d{4}$`) classifies without trusting the payload — untrusted-input discipline applies (QR content never parsed as instructions).

## 5. PAdES/AdES PDF digital-signature verification in Python

**STATUS: PARTIAL (workable with light deps; no pure-stdlib path)**

- **Stdlib reality check:** `hashlib`/`hmac` cover hashing; there is **no** ASN.1/CMS/X.509 parser in stdlib (a common misconception — `ssl` exposes nothing for CMS SignedData, `binascii` doesn't help). So a true PAdES verify cannot be pure-stdlib.
- **Recommended light stack (all pip, CPU-friendly, no GNU tools):**
  - `asn1crypto` (pure python, actively maintained) → parse `CMS SignedData` + X.509 cert chains extracted from the PDF's `/Contents` signature dictionary (PDF spec 32.10; signatures live as PKCS#7/CMS objects in the document).
  - **`pyhanko`** (MIT, pure python, ≥3.9) — the single most useful tool here: it *reads* PAdES/AdES signatures, validates signing-time policy, exposes timestamp tokens (RFC 3161), and even signs. Its `pyhanko.pdf_utils.signature.SignatureField` + `validate` machinery maps directly onto "is this govt PDF genuinely signed by the claimed authority".
  - **`cryptography`** (Rust-backed, tiny) → ECDSA/RSA signature verify + chain building once asn1crypto has handed us certs. For CAs: bundle the relevant government root CAs in a versioned trust store (e.g. IndiaSign/Meghna, Sify, T-Systems roots + per-ministry CA certs) — **trust-store-as-data, versioned, in-repo**.
- **GnuPG is the wrong tool** (OpenPGP ≠ CMS/PKCS#7); the old plan is retired.
- **VeriSafe mapping:** new optional stage in `gov_document.py` ladder *after* extraction: `signature_check(pdf_bytes)` → CheckResult {signed: bool, valid_chain: bool?, signer_cn, signature_type(PAdES-B/L/ES), error}. Weight it strongly but not decisively (many legit state certificates are unsigned/scanned — absence ≠ fraud).

## 6. Playwright headless vs .gov.in anti-bot (observed today)

| Site | HTTP | WAF/anti-bot signals | Practical implication |
|---|---|---|---|
| www.digilocker.gov.in | 200 | none (CloudFront CDN only) | polite GET OK |
| www.digilocker.gov.in/**share** | **403** | bot-filter on protected routes | don't scrape; use user-assisted QR flow instead |
| directory.apisetu.gov.in | 200 | none visible (Next.js SPA) | fine with real UA; rate-limit ourselves |
| epfo.gov.in | 200 | **Cloudflare** (`cf` marker) | expect challenges on deeper paths; headless may hit CF JS challenge — budget retries w/ backoff, don't loop |
| voter.gov.in | unreachable (-1) | (conn failure during probe) | flaky upstream; retry w/ timeout ladder |
| incometax.gov.in | 200 | **Akamai** markers | Akamai bot-manager on some paths; keep traffic minimal, cache aggressively |

**Politeness settings to bake into any browser stage:** real Chrome UA string, viewport 1366×768, `locale=hi-IN/en-IN`, `timezone_id=Asia/Kolkata`; ≥2 s jitter between navigations; max 1 retry per resource with 5–10 s backoff; respect robots.txt per-host; absolute cap of N=3 unique URLs per job; abort whole stage on any CAPTCHA/challenge page detected (never solve challenges — that crosses from "polite client" into active circumvention). This keeps us on the right side of every host's ToS and makes the browser a *last resort*, matching the architecture's offline-first posture.

---

## 7. Implications for offline fallback design (authoritative API unavailable → what to cache)

When DigiLocker/e-KYC consumer access is absent (our baseline case), VeriSafe runs **controlled browser + versioned RAG template cache**. The cache is a *retrieval aid*, never a source of truth (architectural invariant). Contents to version:

1. **Document-template fingerprints** (per doc class): expected field inventory + label variants (EN/HI + common states' language), layout zones (header logos, QR position box, serial patterns), watermarks/security-feature descriptions. Sources: official specimen PDFs from ministry sites (fetch once, hash, store), plus API Setu catalog metadata (doc types per issuer — from the digest above).
2. **Issuer-trust data**: `issuerId` → orgName/type/state, official domain(s), logo hashes, known QR scheme per doc class (item 4 table). Source: API Setu discovery API (refresh cadence: monthly, ~20 calls).
3. **Signature trust store**: pinned government CA roots + known signer CN lists per ministry (item 5).
4. **Official-content baselines**: short canonical strings/phrases per form (e.g. "Permanent Account Number" appears verbatim on every PAN card) — cheap tamper heuristics; each entry carries source URL + fetch date.
5. **Negative cache**: last-known-good versions of each cached artifact with SHA-256; anything stale >90 days degrades confidence automatically (version stamp in every verdict: "template cache vN, fetched YYYY-MM-DD").

Everything is **rebuildable from public sources by a human following the recipe in this doc** — no proprietary data, no retained user documents (zero-retention still governs: templates are generic, keyed by document class, never by a specific user's content).
