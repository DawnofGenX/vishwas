<p align="center"><img src="https://raw.githubusercontent.com/DawnofGenX/vishwas/master/docs/og-banner.png" alt="vishwas" width="100%"/></p>

# Vishwas

WhatsApp-first verification & safety platform. Users message a WhatsApp number; Vishwas checks what they send — **URLs, files (APK/PE/ELF/documents), images, audio, video, government documents** — and replies in plain language with a verdict, a **confidence band**, and practical next steps. Built for constrained CPU-only hardware (designed around an i5-8250U laptop), zero-cloud, stdlib-first, zero-retention by contract.

## What it does

| You send | It checks | Verdict vocabulary |
|---|---|---|
| A link | normalisation → SSRF guard → VirusTotal reputation → host-string + DOM phishing heuristics | trust / caution / do_not_use / unable_to_verify |
| An executable or archive (APK, PE, ELF/.so, .jar, zip…) | magic-byte validation → file entropy → ClamAV → YARA-x → Quark-Engine → PE static analysis → APK statics (MoBFS-lite) → (gated) dynamic sandbox (Cape/firejail+strace) | same |
| A document (PAN, DL, certificate, PDF/image scan) | extraction/OCR → document-type ID → field-level validation → GPG signature / PAdES verification / QR payload + native check / RAG template cache (retrieval cache, not source of truth) | same |
| Photo/video/audio | ffprobe probe → frame/audio heuristics → (gated) named deepfake detectors: EFFORT (ICML 2025 Oral spatial), aasist3 (Spectra-AASIST3), XLSR-Mamba, NYUAD (image, 2nd signal); cross-modal AV consistency (HAVIC); transform-battery robustness (no raw-score averaging) | same |

Deterministic routing end-to-end: input type → MIME/magic bytes → capability set. No LLM-gated decisions anywhere; the LLM (when provisioned) only *narrates* over already-computed evidence, behind a prompt-injection guard.

## Measured results

All figures below were re-measured on **2026-10-21** by re-scoring the raw
media with the real checkpoints (not by replaying stored feature vectors).
Raw records are in `docs/research/`.

**Per-detector accuracy**, on the corpora named:

| Channel | Detector | Corpus | n | AUC | EER |
|---|---|---|---|---|---|
| Audio | aasist3 (Spectra-AASIST3) | official ASVspoof2019-LA eval | 300 | **1.0000** | **0.00%** |
| Audio | xlsr-mamba | official ASVspoof2019-LA eval | 300 | **1.0000** | **0.00%** |
| Image | SPAI (spectral) | raw image corpus (30 AI / 30 real) | 60 | 0.7433 | 26.67% |
| Video | EFFORT (FF++-trained) | raw FaceForensics++ subset (52 fake / 50 real) | 102 | 0.6654 | 38.46% |

Posterior medians: aasist3 bonafide **0.0000** / spoof **0.9831** (perfectly
separated, 0/150 misclassified at 0.5) · SPAI real **0.0015** / AI **0.6790**
· EFFORT real **0.6209** / fake **0.7227**.

**Where each detector still falls short** — stated because a security tool
that only publishes its wins is not telling you the truth:

- **SPAI misses 8 of 30 AI images outright** (scores below 0.05): 3 flux,
  2 midjourney v6.1, 2 stable-diffusion-3, 1 gigagan. It also false-highs
  6 of 30 real photos. This is why a second, independent image signal
  (NYUAD, 3-class ViT) corroborates SPAI rather than trusting it alone.
- **EFFORT barely separates on this corpus** (AUC 0.6654). The 87-clip
  calibration corpus behind the older operating point has **no audio stream
  on 83 of its 87 rows**, so the two cross-modal gates (AV-sync, weight 3.5;
  HAVIC, weight 0.8) were never actually measured on 95% of the set the
  weights were tuned on. The older "42/42 reals not flagged" figure is true
  only because the clean-side floor is set to avoid false positives — it is
  not evidence that EFFORT separates. Treat the video channel as
  single-signal until a corpus with audio exists.
- **The video channel has no clean/low floor for silent clips in this
  corpus**, so real videos read CAUTION rather than LOW. That is deliberate:
  an aggressive LOW posture on a detector that weak would be unsafe.

**Test suite:** 530 passed / 6 skipped (bare env, hermetic, Python 3.12;
the 2 failures + 2 errors are the pre-existing numpy-ABI `cv2` and
missing-`torchvision` SPAI issues). With the learned detectors installed the
suite collects 544 tests — the torch-gated
architecture modules skip wholesale when torch is absent, so a bare run is
not evidence that the detectors work. CI asserts both.

### Components rejected on evidence

A learned fusion stack trained on out-of-fold predictions scored **OOF AUC
0.512** against a **0.502** heuristic baseline, so it is not wired into the
serving path. A gradient-boosted alternative was rejected at OOF AUC 0.499
at n=84 — chance.

The audio detector was vendored, loaded cleanly (1022/1022 checkpoint keys)
and **rejected on a 240-clip validation slice** at **AUC 0.54** against a
bar of **AUC ≥ 0.85** fixed before the run. That result was later
**superseded**: the same checkpoint was re-measured on the *official*
ASVspoof2019-LA evaluation set at **AUC 0.9967 / EER 0.70%** (vendor) and
re-measured here at **AUC 1.0000 / EER 0.00%** on 300 official-eval
samples. The detector is now **wired in** as the audio channel, with clean
audio reading LOW and a spoof reading HIGH. The 0.54 figure was a bad
*validation slice*, not a bad model — but it was a real measurement at the
time, and the rejection was honest given the evidence available then.

Image verdicts deliberately **cap at CAUTION**: a single learned-model read
never escalates a photo to DO_NOT_USE, because that detector false-highs
6/30 real images on our own check set.

## Quick start (this box)

```bash
cd /home/hermes/vishwas
export PYTHONPATH=$PWD/src

# Use the launcher, NOT a bare `python3` — it selects the interpreter that
# can actually load the learned detectors (see "Runtime" below). Calling
# `python3 -m vishwas.app` directly works for the cheap tiers, but the
# torch-gated detectors silently degrade to "unavailable" and the audio and
# image channels fail OPEN.
#
# CLI — simulate a WhatsApp user locally
scripts/run_vishwas.sh cli --text "https://bank-secure-login.example.net"
scripts/run_vishwas.sh cli --file ./path/to/file.apk

# Webhook server (pair with OpenWA below)
OPENWA_BASE_URL=http://localhost:2785 OPENWA_SESSION_ID=main \
OPENWA_API_KEY=<key> OPENWA_WEBHOOK_SECRET=<secret> \
scripts/run_vishwas.sh webhook --port 2790

# Full test suite (offline, no network needed): 471 tests, hermetic
python3 -m pytest tests/ -q
```

## Runtime

The learned detectors need a specific interpreter, and using the wrong one
degrades the platform **silently**:

- The wav2vec2 frontend behind the audio gate is a 1.27 GB
  `pytorch_model.bin` pickle, and `transformers` ≥ 4.57 refuses
  `torch.load` on `torch` < 2.6 (CVE-2025-32434). The system `torch` is
  2.4.1, so the audio gates cannot load under it at all.
- `pylibs` (the legacy local dependency tree) must **not** be on
  `PYTHONPATH`: its `cryptography` v50 no longer ships the `lib` submodule
  that pyOpenSSL imports, so Python resolves `lib` to a top-level namespace
  package and the whole `accelerate`/`transformers` import chain dies.

`scripts/run_vishwas.sh` handles both: it execs the dedicated venv
(`/home/hermes/.venv-ambient`) and keeps `PYTHONPATH` to the repo's own
`src/`. **If the venv is missing the launcher refuses to start**, printing
the rebuild command, rather than starting a verifier whose detectors are all
dead.

Rebuild it with:

```bash
uv venv --python 3.11 /home/hermes/.venv-ambient
VIRTUAL_ENV=/home/hermes/.venv-ambient uv pip install --python $VIRTUAL_ENV/bin/python \
  -e ".[learned,binary]"
```

Runtime dependencies are pinned in `pyproject.toml`.

## Architecture (one line per layer)

See `docs/ARCHITECTURE.md` for the full diagram.
- **channels.py** — OpenWA v0.21.0 client (REST `/api/sessions/{id}/...`, port 2785, SQLite-backed) + inbound webhook with `X-OpenWA-Signature` HMAC verification and idempotency dedupe.
- **router.py + file_validator.py** — deterministic classification: extension hypothesis, then magic bytes as ground truth (`ext_mismatch` flagged when they disagree).
- **capabilities/** — seven specialist modules (url_phishing, malware_file, gov_document, deepfake_video, deepfake_audio, cross_modal, image_facecheck). Each declares its dependency `requires=` tuple and emits `CheckResult(name, cost, status, signals, notes)` — evidence only, never verdicts.
- **fusion.py** — per-target logistic regression stacking (LR stack if a trained checkpoint exists under `VISHWAS_FUSION_DIR`; documented explicit weights otherwise), temperature calibration, disagreement metric, and a reliability gate that can force `unable_to_verify` when evidence quality is insufficient.
- **report.py + i18n.py** — plain-language replies with confidence bands; 7 languages (en hi ta te ml kn bn). The reply language is the user's **saved choice** (set once, persisted) — it is **never auto-detected from message content** (operator decision).
- **quarantine.py** — every job gets an isolated dir under `VISHWAS_QUARANTINE`; original + every derived artifact are purged on completion **and** on failure, with a stale-sweep TTL (`VISHWAS_STALE_TTL_S`). Zero retention is enforced, not optional.
- **orchestrator.py** — wall-clock budget (default 300s via `VISHWAS_BUDGET_S`), 10s stage floor, conservative short-circuit after confirmed positives (P8), per-stage timing records, crash isolation per stage.

## Configuration reference

| Env var | Default | Purpose |
|---|---|---|
| `OPENWA_BASE_URL` | `http://localhost:2785` | OpenWA REST base |
| `OPENWA_SESSION_ID` | `main` | Which session handles traffic |
| `OPENWA_API_KEY` | — | Bearer auth for OpenWA REST calls |
| `OPENWA_WEBHOOK_SECRET` | — | Expected HMAC for inbound webhooks; empty = unsigned accepted (documented operator choice) |
| `VISHWAS_WEBHOOK_HOST/PORT` | 127.0.0.1 / 8899 | Inbound listener bind (the systemd unit uses 2790) |
| `VISHWAS_VT_API_KEY` | — | Enables VirusTotal (files + URLs) |
| `VISHWAS_LLM_BASE_URL` / `_API_KEY` / `_MODEL` (or `OPENAI_API_KEY`) | — | Optional LLM narration layer |
| `VISHWAS_FFMPEG_BIN/FFPROBE_BIN/THREADS` | ffmpeg / ffprobe / **2** | Media processing; thread cap is thermal-safe (see `docs/PERFORMANCE.md`) |
| `VISHWAS_CLAMSCAN_BIN` / `CLAMD_DB` | clamscan / `/var/lib/clamav` | Local AV |
| `VISHWAS_YARA_RULES` | — | YARA rule pack path |
| `VISHWAS_TESSERACT_BIN` / `DOCLING` / `GPG_BIN` | tesseract / docling / gpg | OCR / document parsing / signature verification |
| `VISHWAS_QUARK_PY` / `MOBSF_CLI` / `CAPE_CMD` | — | Quark-Engine wrapper, MoBFS CLI, dynamic sandbox command |
| `VISHWAS_<MODEL>_WEIGHTS` (EFFORT, AASIST, XLSRMAMBA, SSL_AUDIO, HAVIC, IMAGE_FACE, NYUAD) | — | Path to a downloaded model weight set; enables that detector (canonical sources: `docs/research/MODEL_WEIGHTS_MANIFEST.json`) |
| `VISHWAS_FUSION_DIR` | — | Trained LR-stacking checkpoints (from `fusion_train.py`) |
| `VISHWAS_RAG_CACHE` / `_VERSION` | — | Document-template retrieval cache (NOT a source of truth) |
| `VISHWAS_QUARANTINE` / `STALE_TTL_S` / `WORKDIR` / `AUDIT_LOG` / `LOG_LEVEL` / `BUDGET_S` | see `app.py` | storage paths, sweep TTL, audit trail, hard time budget |

## Deployment

Real-world wiring with OpenWA (docker compose, webhook registration, secret management, health checks) is in **`docs/DEPLOYMENT.md`**. Non-technical behaviour guide for users and operators is in **`docs/USER_GUIDE.md`**; adversary-facing design rationale is in **`docs/THREAT_MODEL.md`**; the honest capability inventory and how to enable every gated feature is in **`docs/GAPS_AND_ENABLEMENT.md`**.

More writing on this stack: https://dev.to/dawnofgenx
