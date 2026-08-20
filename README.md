# VeriSafe

WhatsApp-first verification & safety platform. Users message a WhatsApp number; VeriSafe checks what they send — **URLs, files (APK/PE/ELF/documents), images, audio, video, government documents** — and replies in plain language with a verdict, a **confidence band**, and practical next steps. Built for constrained CPU-only hardware (designed around an i5-8250U laptop), zero-cloud, stdlib-first, zero-retention by contract.

## What it does

| You send | It checks | Verdict vocabulary |
|---|---|---|
| A link | normalisation → SSRF guard → VirusTotal reputation → host-string + DOM phishing heuristics | trust / caution / do_not_use / unable_to_verify |
| An executable or archive (APK, PE, ELF/.so, .jar, zip…) | magic-byte validation → file entropy → ClamAV → YARA-x → Quark-Engine → PE static analysis → APK statics (MoBFS-lite) → (gated) dynamic sandbox (Cape/firejail+strace) | same |
| A document (PAN, DL, certificate, PDF/image scan) | extraction/OCR → document-type ID → field-level validation → (gated) DigitalLocker API / SETU API / GPG signature / RAG template cache (retrieval cache, not source of truth) | same |
| Photo/video/audio | ffprobe probe → frame/audio heuristics → (gated) named deepfake detectors: EFFORT (ICML 2025 Oral spatial), DeMamba (degradation-robust general), Fake-Mamba, AASIST, SSL-audio; cross-modal AV consistency (HAVIC); transform-battery robustness (no raw-score averaging) | same |

Deterministic routing end-to-end: input type → MIME/magic bytes → capability set. No LLM-gated decisions anywhere; the LLM (when provisioned) only *narrates* over already-computed evidence, behind a prompt-injection guard.

## Quick start (this box)

```bash
cd /home/hermes/verisafe
export PYTHONPATH=$PWD/src

# CLI — simulate a WhatsApp user locally
python3 -m verisafe.app cli --text "https://bank-secure-login.example.net"
python3 -m verisafe.app cli --file ./path/to/file.apk --media-type file

# Webhook server (pair with OpenWA below)
VERISAFE_WEBHOOK_HOST=0.0.0.0 VERISAFE_WEBHOOK_PORT=8480 \
OPENWA_BASE_URL=http://localhost:2785 OPENWA_SESSION_ID=main \
OPENWA_API_KEY=<key> OPENWA_WEBHOOK_SECRET=<secret> \
python3 -m verisafe.app webhook

# Full test suite (offline, no network needed): 131 tests
python3 -m pytest tests/ -q
```

## Architecture (one line per layer)

See `docs/ARCHITECTURE.md` for the full diagram.
- **channels.py** — OpenWA v0.21.0 client (REST `/api/sessions/{id}/...`, port 2785, SQLite-backed) + inbound webhook with `X-OpenWA-Signature` HMAC verification and idempotency dedupe.
- **router.py + file_validator.py** — deterministic classification: extension hypothesis, then magic bytes as ground truth (`ext_mismatch` flagged when they disagree).
- **capabilities/** — seven specialist modules (url_phishing, malware_file, gov_document, deepfake_video, deepfake_audio, cross_modal, image_facecheck). Each declares its dependency `requires=` tuple and emits `CheckResult(name, cost, status, signals, notes)` — evidence only, never verdicts.
- **fusion.py** — per-target logistic regression stacking (LR stack if a trained checkpoint exists under `VERISAFE_FUSION_DIR`; documented explicit weights otherwise), temperature calibration, disagreement metric, and a reliability gate that can force `unable_to_verify` when evidence quality is insufficient.
- **report.py + i18n.py** — plain-language replies with confidence bands; 7 languages (en hi ta te ml kn bn), auto-detected from the user's message.
- **quarantine.py** — every job gets an isolated dir under `VERISAFE_QUARANTINE`; original + every derived artifact are purged on completion **and** on failure, with a stale-sweep TTL (`VERISAFE_STALE_TTL_S`). Zero retention is enforced, not optional.
- **orchestrator.py** — wall-clock budget (default 300s via `VERISAFE_BUDGET_S`), 10s stage floor, conservative short-circuit after confirmed positives (P8), per-stage timing records, crash isolation per stage.

## Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `OPENWA_BASE_URL` | `http://localhost:2785` | OpenWA REST base |
| `OPENWA_SESSION_ID` | `main` | Which session handles traffic |
| `OPENWA_API_KEY` | — | Bearer auth for OpenWA REST calls |
| `OPENWA_WEBHOOK_SECRET` | — | Expected HMAC for inbound webhooks; empty = unsigned accepted (documented operator choice) |
| `VERISAFE_WEBHOOK_HOST/PORT` | 127.0.0.1 / 8480 | Inbound listener bind |
| `VERISAFE_VT_API_KEY` | — | Enables VirusTotal (files + URLs) |
| `VERISAFE_LLM_BASE_URL` / `_API_KEY` / `_MODEL` (or `OPENAI_API_KEY`) | — | Optional LLM narration layer |
| `VERISAFE_FFMPEG_BIN/FFPROBE_BIN/THREADS` | ffmpeg / ffprobe / **2** | Media processing; thread cap is thermal-safe (see `docs/PERFORMANCE.md`) |
| `VERISAFE_CLAMSCAN_BIN` / `CLAMD_DB` | clamscan / `/var/lib/clamav` | Local AV |
| `VERISAFE_YARA_RULES` | — | YARA rule pack path |
| `VERISAFE_TESSERACT_BIN` / `DOCLING` / `GPG_BIN` | tesseract / docling / gpg | OCR / document parsing / signature verification |
| `VERISAFE_DIGILOCKER_URL` / `_KEY`, `VERISAFE_APISETU_BASE` / `_TOKEN` | — | Government live-API verification (India: DigitalLocker, SETU) |
| `VERISAFE_QUARK_PY` / `MOBSF_CLI` / `CAPE_CMD` | — | Quark-Engine wrapper, MoBFS CLI, dynamic sandbox command |
| `VERISAFE_<MODEL>_WEIGHTS` (EFFORT, DEMAMBA, FAKEMAMBA, AASIST, SSL_AUDIO, HAVIC, IMAGE_FACE) | — | Path to a downloaded model weight set; enables that detector (canonical sources: `docs/research/MODEL_WEIGHTS_MANIFEST.json`) |
| `VERISAFE_FUSION_DIR` | — | Trained LR-stacking checkpoints (from `fusion_train.py`) |
| `VERISAFE_RAG_CACHE` / `_VERSION` | — | Document-template retrieval cache (NOT a source of truth) |
| `VERISAFE_QUARANTINE` / `STALE_TTL_S` / `WORKDIR` / `AUDIT_LOG` / `LOG_LEVEL` / `BUDGET_S` | see `app.py` | storage paths, sweep TTL, audit trail, hard time budget |

## Deployment

Real-world wiring with OpenWA (docker compose, webhook registration, secret management, health checks) is in **`docs/DEPLOYMENT.md`**. Non-technical behaviour guide for users and operators is in **`docs/USER_GUIDE.md`**; adversary-facing design rationale is in **`docs/THREAT_MODEL.md`**; the honest capability inventory and how to enable every gated feature is in **`docs/GAPS_AND_ENABLEMENT.md`**.
