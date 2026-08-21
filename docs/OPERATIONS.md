# VeriSafe Operations

## Stale-quarantine sweep (`scripts/stale_sweep.sh`)

Every job works inside a quarantine dir under `~/verisafe/quarantine`; if the
process crashes before its purger runs, that dir lingers. The sweep calls
`scan_stale_quarantines()` to delete any quarantine whose manifest `ts` is
older than `VERISAFE_STALE_TTL_S` (default 7200s) and appends an audit line
per removal. It is a silent watchdog: empty output + exit 0 on a quiet tick,
a swept-count line only when something was removed, errors on stderr with a
non-zero exit.

- **Cadence:** run every 15 minutes via cron (TTL is 2h, so worst-case residue
  lives ~2h15m). Suggested line:
  `*/15 * * * * /home/hermes/verisafe/scripts/stale_sweep.sh`
- **Manual run:** `/home/hermes/verisafe/scripts/stale_sweep.sh` (add
  `VERISAFE_QUARANTINE=...` / `VERISAFE_AUDIT_LOG=...` to point elsewhere).
- **Audit log:** `logs/purge_audit.log` (one JSON line per purge/sweep;
  override with `VERISAFE_AUDIT_LOG`).

## Rich `/health` (service observability)

`GET /health` on the webhook service returns: `status`, `uptime_s`,
`jobs_total` / `jobs_ok` / `jobs_failed` (thread-safe counters incremented in
the `MessageProcessor.process()` outcome path), `quarantines_open`
(subdirs under the quarantine root), and a `deps` summary
(`{"available": [...], "count": N}`) plus the backward-compat flat
`deps_available` list.

- **Counters reset on restart** by design (no DB) — treat them as
  since-last-restart figures, not lifetime totals.
- **Daily check:** `curl -s localhost:2785/health | python3 -m json.tool`.
- **Systemd:** see `deploy/verisafe.service.example` for a ready unit
  (`Restart=on-failure`, `RestartSec=5`); install with the comment-block
  steps at the top of that file.

## Government API registration checklist (operator steps)

Source of truth: `docs/research/INDIA_GOV_VERIFICATION.md` §1–§2 (live-probed
2026-08-19). We cannot perform these registrations for you; everything below
is operator work totalling **~30 min** once you sit down to it — *external*
approval/due-diligence waits are called out separately and are not part of
that estimate. Current enablement state lives in the
`digilocker`/`setu` row of `docs/GAPS_AND_ENABLEMENT.md`.

### Ground truth first (2 min — avoids dead ends)

- The old public developer portals are **dead**: `services.digilocker.gov.in`,
  `developer.digilocker.gov.in`, `apisetu.api.gov.in`, `apisetu.dev.gov.in`
  are all NXDOMAIN. Ignore older blog posts pointing at them.
- Live stack (re-resolved 2026-08-19): `www.digilocker.gov.in` (consumer app
  only), `apisetu.gov.in` (portal), `directory.apisetu.gov.in` (catalog),
  `docs.apisetu.gov.in` (docs incl. "SOP for API Access"),
  `partners.apisetu.gov.in` (signup).

### (a) DigiLocker partner program — partnership-gated OAuth (~15 min operator work)

There is **no anonymous self-serve key**. Access requires becoming an
authorized service provider (DLP/verifier registration with MoHA/UIDAI),
passing a due-diligence process, and being issued OAuth2/OIDC credentials.

1. **Locate the current partner/e-KYC documentation** on
   `digilocker.gov.in` (partner docs exist publicly; the exact application
   URL is not captured in our research doc — verify at the official portal).
   *(~5 min)*
2. **Apply as authorized service provider** (DLP/verifier registration with
   MoHA/UIDAI) and submit the due-diligence material the portal asks for.
   *(~10 min to submit; approval is an external wait — unbounded)*
3. **On credential issuance**, set in the service environment:
   - `VERISAFE_DIGILOCKER_KEY` — required; without it the pipeline emits
     `digilocker_verify = unavailable` and skips authoritative DigiLookup.
   - `VERISAFE_DIGILOCKER_URL` — optional override; code default is
     `https://apis.digilocker.gov.in/dl/v1/verDoc`
     (`src/verisafe/capabilities/gov_document.py::_digilocker`). Confirm the
     correct verify endpoint against the partner docs before relying on the
     default.

**What it unlocks:** the gated `digilocker_verify(qr_or_link)` capability —
OAuth token grant → DigiLocker's JSON e-KYC response — i.e. *authoritative*
verification when a user supplies a DigiLocker QR/link, instead of the
credential-free fallback ("open this in your phone's DigiLocker app",
user-assisted verification). Note `/share` routes are bot-gated (403):
never scrape them; the QR/link flow is the supported shape.

### (b) API Setu consumer credentials — client-credentials OAuth (~15 min operator work)

1. **Register at `partners.apisetu.gov.in`** (consumer/partner signup).
   Read the "SOP for API Access" on `docs.apisetu.gov.in` first — note its
   body is JS-rendered, so use a browser. *(~10 min)*
2. **Receive the OAuth2 client-credentials pair** (`client_id`/`secret`);
   rate limits are negotiated per consumer, not published as fixed numbers.
3. **Set in the service environment:**
   - `VERISAFE_APISETU_TOKEN` — required; without it the pipeline emits
     `api_setu_lookup = unavailable`.
   - `VERISAFE_APISETU_BASE` — set this to the gateway host from your
     partner onboarding (`<subdomain>.apisetugateway.in` or
     `gateway.apisetu.gov.in`). The code default
     (`https://apisetu.gov.in/api/v1`) predates the platform re-homing —
     verify the correct base at the portal.
4. **Legal posture (non-negotiable):** personal KYC data returned by these
   APIs may only be processed under the consuming entity's legal
   authorization **and** the user's consent; keep VeriSafe's zero-retention
   posture or the ToS are violated.

**What it unlocks:** registered-consumer invocation of the IndiaStack
e-KYC/certificate endpoints (pattern
`/certificate/v3/<subdomain>/<spec>`): a citizen authorizes release, the
verifier pulls signed content. Code routes `pan_card → /pan/status` and
`epf_statement → /epf/membership`. Directory coverage as of 2026-08-19
(keyword → catalog hits): income certificate 36 · ration 32 · aadhaar 15 ·
pan 13 · passport 4 · udyam 2 · voter id 1 · **epf 0** (no directory
entries — expect the EPF route to stay unproven even with creds).

### Anonymous discovery surface — no registration required (works today)

The directory search needs no account: single-use bearer per call.

```bash
python3 - <<'PY'
import json, urllib.parse, urllib.request
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
b = "https://directory.apisetu.gov.in"
req = urllib.request.Request(b + "/api/auth/generate-headers",
                             headers={"User-Agent": UA, "Referer": b + "/search"})
h = json.load(urllib.request.urlopen(req, timeout=30))
q = urllib.parse.urlencode({"q": json.dumps({"query": "aadhaar"}), "size": 3, "page": 0})
req2 = urllib.request.Request(b + "/api/list?" + q, headers={
    "Authorization": h["authorization"], "x-nonce": h["nonce"], "ts": h["ts"],
    "Referer": b + "/search", "User-Agent": UA})
print(json.load(urllib.request.urlopen(req2, timeout=30))["result"]["nbHits"])
PY
```

Rules (from §2 of the research doc): the token from `generate-headers` is
**single-use** — regenerate immediately before each list call and send
`Authorization` + `x-nonce` + `ts`; `q` must be a JSON object whose
free-text field is `query`; keep the browser UA + `Referer` (bare-UA calls
get 403 — re-verified 2026-08-21). This surface feeds the RAG template-cache
refresher (`scripts/build_rag_cache.py`; refresh cadence monthly, ~20 calls).

### Verification once creds exist

| Credential | Ad-hoc live proof | Expected |
|---|---|---|
| none (discovery) | snippet above | `status: Success`, `nbHits: 15` for aadhaar |
| `VERISAFE_APISETU_BASE`+`TOKEN` | `curl -s "$VERISAFE_APISETU_BASE/pan/status" -H "Authorization: Bearer $VERISAFE_APISETU_TOKEN"` | JSON response, **not** 401/403 (exact auth-header shape per the SOP — verify at `docs.apisetu.gov.in`) |
| `VERISAFE_DIGILOCKER_URL`+`KEY` | `curl -s "$VERISAFE_DIGILOCKER_URL" -H "Authorization: Bearer $VERISAFE_DIGILOCKER_KEY"` | non-401/403 per the partner-docs OAuth flow (verify at portal) |

End-to-end: run one real job through the pipeline and confirm the verdict no
longer lists `digilocker_verify` / `api_setu_lookup` as `unavailable`, then
flip the `digilocker`/`setu` row in `docs/GAPS_AND_ENABLEMENT.md` ❌→✅ with
the evidence date (same pattern as the VirusTotal row).

### Not sourced from the research doc — verify at the official portals

- Exact DigiLocker partner-application URL and form steps (research doc only
  establishes that partner docs exist on `digilocker.gov.in` and that access
  is MoHA/UIDAI-gated).
- Exact API Setu token-exchange call and auth-header shape for consumer
  requests (the SOP page body was JS-rendered at capture time; only the
  partners.apisetu.gov.in → client-credentials model is documented).
- The correct production gateway base host for `VERISAFE_APISETU_BASE`.
