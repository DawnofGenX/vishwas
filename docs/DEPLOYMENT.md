# Deployment — Vishwas + OpenWA

Reference wiring for **OpenWA v0.21.0** as the sole WhatsApp transport. OpenWA is a self-hosted gateway that exposes per-session REST APIs (port 2785 default) backed by SQLite; Vishwas talks to it over plain HTTP and receives inbound events via an HMAC-signed webhook.

## Topology

```
WhatsApp
   │
OpenWA container ──(REST /api/sessions/{id}/...)──────────► vishwas webhook server (:2785, in-container)
   │                                                            │
   └────────────◄── reply: POST /api/sessions/{id}/messages/send-text ◄──┘
```

Both run on the same box here (laptop, zero-cloud constraint). The committed
compose file puts them on one compose network: OpenWA reaches Vishwas's
webhook at `http://vishwas:2785`, Vishwas reaches OpenWA's REST at
`http://openwa:2785`; only OpenWA's REST is published to the host (loopback).

## Committed deployment artefacts (Roadmap T4.4)

| Artefact | Purpose |
|---|---|
| `deploy/docker-compose.yml` | Two-service stack: `openwa` (pinned `rmyndharis/openwa:v0.21.0`) + `vishwas` (built from `deploy/Dockerfile`). Secrets read from `deploy/.env`: `OPENWA_API_KEY` (required), `OPENWA_WEBHOOK_SECRET` (recommended), optional `VISHWAS_MODELS_HOST_DIR` to relocate the weights mount. |
| `deploy/Dockerfile` | Thin runtime image: `python:3.12-slim` + `curl` (healthcheck) + `numpy==2.4.6` — the one hard third-party import on the boot path (`capabilities/__init__` eagerly imports `deepfake_video`). Heavy capability trees stay host-side by design; gated stages report `unavailable` in `/health` and degrade gracefully. |
| `.dockerignore` | Keeps the build context to `src/` + `scripts/`. |
| `deploy/webhook.example.json` | OpenWA webhook-registration body pointing at Vishwas's route (see "Webhook registration"). |
| `deploy/vishwas.service.example` | Bare-metal systemd unit (T2.3). |

Wiring summary (matches the code; older drafts of this doc said `/healthz`, `/openwa` and port 8480 — all stale):

- Vishwas routes: **`POST /webhook/inbound`** (events) and **`GET /health`** (rich JSON snapshot). Code default port is `8899` (`VISHWAS_WEBHOOK_PORT`); the compose service pins **2785** container-internally to match this host's ops convention.
- OpenWA REST stays on its own container's 2785, reached as `http://openwa:2785` over the compose network; published to the host loopback-only (`127.0.0.1:2785`). Vishwas's webhook port is deliberately NOT published — no ingress needed, OpenWA pushes events out.
- Model weights are volume-mounted **read-only** at `/opt/vishwas/models`; the container entrypoint runs `scripts/provision_weight_env.sh --quiet` at start (absent gates skip safely), then execs `scripts/run_vishwas.sh webhook --port 2785`.
- Quarantine and audit log bind-mount to `deploy/quarantine` and `deploy/logs`; OpenWA session state to `deploy/openwa-data`.

### Bring up / down

```bash
cd deploy
printf 'OPENWA_API_KEY=%s\nOPENWA_WEBHOOK_SECRET=%s\n' \
    "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" > .env && chmod 600 .env
docker compose up -d --build        # vishwas flips to healthy once /health answers
docker compose logs -f openwa       # QR appears here → WhatsApp → Linked Devices
# ...then register the webhook once the session exists (next section)
docker compose down                 # add -v only if you want to drop session state
```

> **Operator review required before pairing:** nothing in these artefacts is
> connected to any WhatsApp number. Scanning the QR links a real device — do
> NOT connect a personal or production number without explicit operator
> review; once paired, every inbound message on that session is analyzed and
> answered by the pipeline.

If running Vishwas bare-metal instead of in a container (this laptop's actual mode), just export the env block above and run `python3 -m vishwas.app webhook`.

### Enablement (bare-metal, this host)

**Preferred: use the launcher** — `scripts/run_vishwas.sh <cli|webhook> [args…]`.
It prepends the isolated Python trees to `PYTHONPATH`, applies the thermal-safe
`VISHWAS_FFMPEG_THREADS` default, and execs `python3 -m vishwas.app`:

```bash
scripts/run_vishwas.sh webhook --port 8899 --openwa-url http://localhost:2785
scripts/run_vishwas.sh cli --greet
```

Manual equivalent (if you prefer plain commands):

```bash
# System-wide visibility shim (installed 2026-08-20):
#   $(python3 -c 'import site;print(site.getsitepackages()[0])')/vishwas-pylibs.pth
#   -> adds /home/hermes/pylibs to every python3 process, so pefile, lief,
#      yara_x, cv2 import without any env work.
# docling is NOT in the .pth by design (isolated 5.5 GB tree); add manually or
# just use the launcher:
export PYTHONPATH=/home/hermes/docling-python:/home/hermes/vishwas/src
#   /home/hermes/pylibs           = pefile, lief, yara-x, opencv-python-headless (+deps)
#   /home/hermes/docling-python   = docling 2.120.3 (RapidOCR ONNX + layout models; ~5.5 GB)
#   src/                          = the project itself (no pip install needed)

# OS-level tools are on PATH (apt-installed): tesseract, firejail, clamscan/freshclam, gpg, ffmpeg
# ClamAV signature DB lives at /var/lib/clamav (default); keep it fresh:
sudo freshclam                      # run periodically or on a cron
```

With these set, `detect_available_deps()` reports `browser, clamav, cv2, docling, dynamic-sandbox, media-tools, ocr, pe-lief, pe-static, strace, yara` plus anything you additionally enable below — every gated stage then self-upgrades from *degraded* to *live* automatically. No code change is needed to turn a gate on; that is the whole point of the availability-gate design (see `GAPS_AND_ENABLEMENT.md`).

Capability gates you still choose per deployment (all optional, all degrade gracefully when unset):

```bash
# External reputation lookups (need a key):
#   export VISHWAS_VT_API_KEY=***            # VirusTotal v3 file+URL lookup
# LLM narration layer (any OpenAI-compatible endpoint; the client sends a real
# browser User-Agent by default because several proxies 403-block python-urllib):
#   export VISHWAS_LLM_BASE_URL=https://<host>/v1
#   export VISHWAS_LLM_API_KEY=***
#   export VISHWAS_LLM_MODEL=qwen3.8-27b
#   # optional: export VISHWAS_LLM_USER_AGENT="..."
# RAG template cache (populated by scripts/build_rag_cache.py):
#   export VISHWAS_RAG_CACHE=/home/hermes/rag-cache
#   export VISHWAS_RAG_VERSION=1
# Model weights (only the ones you have downloaded; see MODEL_WEIGHTS_MANIFEST.json):
#   export VISHWAS_MODEL_DIR=/home/hermes/vishwas-models
#   # each named detector keys off its own VISHWAS_<NAME>_WEIGHTS path
```

Everything else is deterministic stdlib and runs as-is.

### Container image

Committed as `deploy/Dockerfile` (build context = repo root, trimmed by
`.dockerignore`): `python:3.12-slim`, `curl` for the healthcheck, and a pinned
`numpy==2.4.6`. The entrypoint/command are wired in `deploy/docker-compose.yml`
(weight provisioning → `scripts/run_vishwas.sh launcher`), so the image stays
runnable standalone too:

```bash
docker run --rm -p 2785:2785 vishwas \
    bash scripts/run_vishwas.sh webhook --port 2785
```

Heavy optional wheels (opencv, pefile, lief, yara-x, docling, playwright)
intentionally stay OUT of the image — this host provisions them into isolated
host-side trees (PEP-668 box) instead of baking multi-GB layers; inside the
container the corresponding gates simply report unavailable and those stages
degrade gracefully (see `GAPS_AND_ENABLEMENT.md`). Append them to the
Dockerfile only if you accept the size and want an all-in-one image.

## First-run pairing & session setup

1. Start OpenWA; its console prints a **QR code** — scan from WhatsApp → Linked Devices.
2. Once paired, the default session id is `main` (visible under `/api/sessions`). If you create additional sessions, point `OPENWA_SESSION_ID` at the one Vishwas should own.
3. Verify REST works: `curl -H "X-API-Key: $OPENWA_API_KEY" http://127.0.0.1:2785/api/sessions/main/status`
   (OpenWA authenticates via the **X-API-Key** header, not `Authorization: Bearer` — see its openapi.json securitySchemes; Vishwas's client sends X-API-Key in `channels.py::_req`.)

## Webhook registration (per session)

Registration is a MANUAL operator step (Vishwas has no self-registration code
path). The ready-made body is `deploy/webhook.example.json`; from the `deploy/`
dir after `up`:

```bash
curl -fsS -X POST \
     -H "X-API-Key: $OPENWA_API_KEY" -H 'Content-Type: application/json' \
     --data-binary @webhook.example.json \
     http://127.0.0.1:2785/api/sessions/main/webhooks
```

Its fields, mapped to Vishwas's parse expectations:

| `webhook.example.json` field | Value | Why |
|---|---|---|
| `url` | `http://vishwas:2785/webhook/inbound` | Vishwas's only event route (service name resolves on the compose network; use a host IP/LAN name for non-compose setups). Field is `url`, not `target` — verified against the OpenWA v0.21.0 README, 2026-08-19. |
| `events` | `["message.received"]` | The only event `parse_openwa_webhook()` acts on (both casings accepted); anything else is answered 200-no-op. |
| `secret` | same value as `OPENWA_WEBHOOK_SECRET` in `deploy/.env` | OpenWA signs each delivery `X-OpenWA-Signature: sha256=<hex>` (HMAC-SHA256 over the RAW request bytes); `verify_openwa_signature()` in `channels.py` recomputes and constant-time-compares exactly that. Retries carry `X-OpenWA-Idempotency-Key` for dedupe. |

(Optional top-level `"filters"` object enables smart pre-dispatch; `RATE_LIMIT_*` env vars on the OpenWA side bound outbound webhook+send throughput.)

(Inbound side of Vishwas verifies each delivery with header `X-OpenWA-Signature: sha256=<hex>` computed over the **raw** request bytes using `OPENWA_WEBHOOK_SECRET`, and dedupes retries on `X-OpenWA-Idempotency-Key`.)

If both sides are started with the same secret, deliveries validate; if `OPENWA_WEBHOOK_SECRET` is empty on the Vishwas side, unsigned deliveries are accepted — a documented operator choice for air-gapped single-box setups only, never for anything reachable beyond your LAN.

## Secret handling

- Generate: `openssl rand -hex 32` for both `OPENWA_API_KEY` and `OPENWA_WEBHOOK_SECRET`; put them in a `.env` next to the compose file (`chmod 600`).
- Never bake secrets into images; the compose file reads them from the environment.
- Rotate by updating both containers' env and restarting in order (OpenWA first, then re-run the webhook-registration curl above).

## Operations

- **Health**: `GET /health` on the webhook port — rich JSON snapshot (status, uptime, job counters, open quarantines, available deps). The compose healthcheck curls it every 30s. From the host: `docker compose -f deploy/docker-compose.yml exec vishwas curl -fsS localhost:2785/health` (compose mode) or `curl -s localhost:2785/health` (bare-metal). The audit log at `VISHWAS_AUDIT_LOG` records every job's verdict, confidence, wall time, and stage timings (P8 fields included).
- **Disk hygiene**: quarantine auto-purges per job; the stale sweep (default TTL in `VISHWAS_STALE_TTL_S`) catches crashed-job leftovers. Monitor the quarantine mount size regardless — this is where disk grows if a sweep fails.
- **Thermal**: this deployment targets an i5-8250U-class CPU. Keep `VISHWAS_FFMPEG_THREADS ≤ 2`, avoid overlapping other heavy jobs; see `docs/PERFORMANCE.md` for the full budget model and the incident-response notes.
- **Logs**: `VISHWAS_LOG_LEVEL=info` default; drop to `debug` only when diagnosing routing questions (it echoes the full route decision).

## Validation status (honest, 2026-08-21)

Validated **live** on this host (docker 29.7.1 + compose v5.4.0, via passwordless sudo):

- `docker compose -f deploy/docker-compose.yml config -q` — clean.
- `deploy/Dockerfile` builds; the built image boots through the compose entrypoint (`provision_weight_env.sh --quiet` → `run_vishwas.sh webhook --port 2785`).
- In that container: `GET /health` returned the rich JSON snapshot and the compose healthcheck reached `healthy`; a correctly HMAC-signed `message.received` POST to `/webhook/inbound` returned **200** with a real pipeline reply; a tampered signature returned **401**.

NOT yet exercised end-to-end (needs operator credentials/review by design): pulling and pairing the real `rmyndharis/openwa:v0.21.0` container, QR linking, and live WhatsApp traffic. The smoke run used `compose run --no-deps`, so the openwa service itself has never been started here.

## Rollback / failure modes

| Failure | Behaviour |
|---|---|
| OpenWA down | Inbound impossible; Vishwas stays up, replies unavailable until gateway returns (no crash loop — the client has bounded timeouts) |
| Webhook signature mismatch | Delivery rejected with 401, logged, dropped; no processing |
| Duplicate webhook delivery (retry storm) | Idempotency key dedupe: first delivery wins, rest answered 200-no-op |
| Job crashes mid-stage | Stage exception isolated, remaining stages continue, job purged, verdict reflects reduced coverage (reliability gate may force `unable_to_verify`) |
| Hard time budget hit | Remaining stages recorded as gaps; partial report sent with explicit "could not fully check" framing |
