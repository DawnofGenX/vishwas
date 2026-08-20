# Deployment — VeriSafe + OpenWA

Reference wiring for **OpenWA v0.21.0** as the sole WhatsApp transport. OpenWA is a self-hosted gateway that exposes per-session REST APIs (port 2785 default) backed by SQLite; VeriSafe talks to it over plain HTTP and receives inbound events via an HMAC-signed webhook.

## Topology

```
WhatsApp
   │
OpenWA container ──(REST :2785 /api/sessions/{id}/...)──► verisafe webhook server (:8480)
   │                                                            │
   └────────────◄── reply: POST /api/sessions/{id}/messages/send-text ◄──┘
```

Both run on the same box here (laptop, zero-cloud constraint). Use `host` networking or a shared compose network so OpenWA can reach the webhook and vice-versa.

## docker-compose.yaml

```yaml
services:
  openwa:
    image: rmyndharis/openwa:v0.21.0          # pin the exact version you validated against
    restart: unless-stopped
    environment:
      API_KEY: ${OPENWA_API_KEY}              # sets the Bearer key VeriSafe authenticates with
      DATA_DIR: /data                          # SQLite + session state live here
    volumes:
      - ./openwa-data:/data
    ports:
      - "2785:2785"                            # REST API; keep bound to 127.0.0.1 if not LAN-exposed
    # NOTE: no public ingress needed for the webhook direction below —
    # OpenWA pushes webhook events OUT to verisafe; see registration step.

  verisafe:
    build: .
    # or: image: your-registry/verisafe:1.0
    depends_on: [openwa]
    environment:
      OPENWA_BASE_URL: http://openwa:2785
      OPENWA_SESSION_ID: main                  # session created after first QR pairing
      OPENWA_API_KEY: ${OPENWA_API_KEY}        # MUST match openwa.API_KEY above
      OPENWA_WEBHOOK_SECRET: ${OPENWA_WEBHOOK_SECRET}
      VERISAFE_WEBHOOK_HOST: 0.0.0.0
      VERISAFE_WEBHOOK_PORT: "8480"
      VERISAFE_QUARANTINE: /var/lib/verisafe/quarantine
      VERISAFE_AUDIT_LOG: /var/log/verisafe/audit.log
      # capability gates — leave unset until provisioned; see GAPS_AND_ENABLEMENT.md
      # VERISAFE_VT_API_KEY: ...
      # VERISAFE_FFMPEG_THREADS: "2"           # thermal-safe default
    volumes:
      - ./quarantine:/var/lib/verisafe/quarantine
      - ./models:/opt/verisafe/models          # weight files referenced by VERISAFE_*_WEIGHTS
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8480/healthz"]
      interval: 30s
      timeout: 5s
      retries: 3
```

If running VeriSafe bare-metal instead of in a container (this laptop's actual mode), just export the env block above and run `python3 -m verisafe.app webhook`.

### Enablement (bare-metal, this host)

**Preferred: use the launcher** — `scripts/run_verisafe.sh <cli|webhook> [args…]`.
It prepends the isolated Python trees to `PYTHONPATH`, applies the thermal-safe
`VERISAFE_FFMPEG_THREADS` default, and execs `python3 -m verisafe.app`:

```bash
scripts/run_verisafe.sh webhook --port 8480 --openwa-url http://localhost:3000
scripts/run_verisafe.sh cli --greet
```

Manual equivalent (if you prefer plain commands):

```bash
# System-wide visibility shim (installed 2026-08-20):
#   $(python3 -c 'import site;print(site.getsitepackages()[0])')/verisafe-pylibs.pth
#   -> adds /home/hermes/pylibs to every python3 process, so pefile, lief,
#      yara_x, cv2 import without any env work.
# docling is NOT in the .pth by design (isolated 5.5 GB tree); add manually or
# just use the launcher:
export PYTHONPATH=/home/hermes/docling-python:/home/hermes/verisafe/src
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
#   export VERISAFE_VT_API_KEY=***            # VirusTotal v3 file+URL lookup
# LLM narration layer (any OpenAI-compatible endpoint; the client sends a real
# browser User-Agent by default because several proxies 403-block python-urllib):
#   export VERISAFE_LLM_BASE_URL=https://<host>/v1
#   export VERISAFE_LLM_API_KEY=***
#   export VERISAFE_LLM_MODEL=qwen3.8-27b
#   # optional: export VERISAFE_LLM_USER_AGENT="..."
# RAG template cache (populated by scripts/build_rag_cache.py):
#   export VERISAFE_RAG_CACHE=/home/hermes/rag-cache
#   export VERISAFE_RAG_VERSION=1
# Model weights (only the ones you have downloaded; see MODEL_WEIGHTS_MANIFEST.json):
#   export VERISAFE_MODEL_DIR=/home/hermes/verisafe-models
#   # each named detector keys off its own VERISAFE_<NAME>_WEIGHTS path
```

Everything else is deterministic stdlib and runs as-is.

### Dockerfile (minimal)

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml setup.py* ./            # adapt to your layout
COPY src/ src/
RUN pip install --no-cache-dir .
# Optional heavy deps (omit to stay light):
# RUN pip install --no-cache-dir opencv-python-headless pefile lief yara-x docling playwright
# This host instead provisions them into an isolated target dir (PEP-668 box)
# and points PYTHONPATH there — see the "Enablement (bare-metal)" section below.
ENV PYTHONPATH=/src
EXPOSE 8480
ENTRYPOINT ["python3", "-m", "verisafe.app", "webhook"]
```

## First-run pairing & session setup

1. Start OpenWA; its console prints a **QR code** — scan from WhatsApp → Linked Devices.
2. Once paired, the default session id is `main` (visible under `/api/sessions`). If you create additional sessions, point `OPENWA_SESSION_ID` at the one VeriSafe should own.
3. Verify REST works: `curl -H "X-API-Key: ***" http://<host>:2785/api/sessions/main/status`
   (OpenWA authenticates via the **X-API-Key** header, not `Authorization: Bearer` — see its openapi.json securitySchemes; VeriSafe's client sends X-API-Key in `channels.py::_req`.)

## Webhook registration (per session)

VeriSafe registers itself against the session at boot:

```
POST /api/sessions/main/webhooks
{ "url": "http://<verisafe-host>:8480/openwa", "events": ["message.received"], "secret": "<hmac-secret>" }
```

(Note: the field is `url`, not `target` — verified against the OpenWA v0.21.0 README, 2026-08-19. Optional top-level `"filters"` object enables smart pre-dispatch; `RATE_LIMIT_*` env vars on the OpenWA side bound outbound webhook+send throughput.)

(Inbound side of VeriSafe verifies each delivery with header `X-OpenWA-Signature: sha256=<hex>` computed over the **raw** request bytes using `OPENWA_WEBHOOK_SECRET`, and dedupes retries on `X-OpenWA-Idempotency-Key`.)

If both sides are started with the same secret, deliveries validate; if `OPENWA_WEBHOOK_SECRET` is empty on the VeriSafe side, unsigned deliveries are accepted — a documented operator choice for air-gapped single-box setups only, never for anything reachable beyond your LAN.

## Secret handling

- Generate: `openssl rand -hex 32` for both `OPENWA_API_KEY` and `OPENWA_WEBHOOK_SECRET`; put them in a `.env` next to the compose file (`chmod 600`).
- Never bake secrets into images; the compose file reads them from the environment.
- Rotate by updating both containers' env and restarting in order (OpenWA first, then re-register the webhook from VeriSafe's boot path).

## Operations

- **Health**: `GET :8480/healthz` (compose healthcheck). The audit log at `VERISAFE_AUDIT_LOG` records every job's verdict, confidence, wall time, and stage timings (P8 fields included).
- **Disk hygiene**: quarantine auto-purges per job; the stale sweep (default TTL in `VERISAFE_STALE_TTL_S`) catches crashed-job leftovers. Monitor the quarantine mount size regardless — this is where disk grows if a sweep fails.
- **Thermal**: this deployment targets an i5-8250U-class CPU. Keep `VERISAFE_FFMPEG_THREADS ≤ 2`, avoid overlapping other heavy jobs; see `docs/PERFORMANCE.md` for the full budget model and the incident-response notes.
- **Logs**: `VERISAFE_LOG_LEVEL=info` default; drop to `debug` only when diagnosing routing questions (it echoes the full route decision).

## Rollback / failure modes

| Failure | Behaviour |
|---|---|
| OpenWA down | Inbound impossible; VeriSafe stays up, replies unavailable until gateway returns (no crash loop — the client has bounded timeouts) |
| Webhook signature mismatch | Delivery rejected with 401, logged, dropped; no processing |
| Duplicate webhook delivery (retry storm) | Idempotency key dedupe: first delivery wins, rest answered 200-no-op |
| Job crashes mid-stage | Stage exception isolated, remaining stages continue, job purged, verdict reflects reduced coverage (reliability gate may force `unable_to_verify`) |
| Hard time budget hit | Remaining stages recorded as gaps; partial report sent with explicit "could not fully check" framing |
