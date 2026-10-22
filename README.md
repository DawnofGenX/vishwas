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
| Video | EFFORT (FF++-trained, face crop) | leak-free corpus (30 fake / 30 real) | 60 | **0.7789** | 50.00% |

Posterior medians: aasist3 bonafide **0.0000** / spoof **0.9831** (perfectly
separated, 0/150 misclassified at 0.5) · SPAI real **0.0015** / AI **0.6790**
· EFFORT real **0.4619** / fake **0.6881**.

**EFFORT was being fed the wrong pixels.** It is a CLIP ViT-L/14, trained on
RGB, but the serving path handed it BGR (`cv2.imread`) with no conversion —
and then normalised it with ImageNet statistics instead of CLIP's. Every
sibling arch (`spai.py`, `nyuad.py`, `effnet_image.py`) does its own
BGR→RGB conversion; EFFORT was the only one relying on the shared
preprocess, which is why the omission was invisible. Fixing the input is
worth **+0.070 AUC** (0.6511 → 0.7189) and it also fixed the operator's
*silent* real video, which used to read MEDIUM and now reads LOW.

**EFFORT is a face-forensics model and was being fed whole frames.** The
upstream paper applies it to face regions, and the cross-modal capability
already loads a YuNet face detector for the HAVIC first-face crop. Cropping
to the detected face (with a whole-frame fallback, because a clip may have no
detectable face) is worth a further **+0.060 AUC** (0.7189 → 0.7789) and
lifts the operator's AI videos from 0.32–0.37 to 0.54–0.65 on effort.

The fusion offset was re-derived twice to match (‑0.55 → ‑0.30 → ‑0.40),
because the old offset was fitted to the broken score distribution. See
`docs/research/` and the measurement notes.

### The video channel was re-measured on a corpus that had a label leak

The cross-modal gates were originally weighted on the 87-clip calibration
corpus, which has **no audio stream on 83 of its 87 rows** — so AV-sync
(weight 3.5) and HAVIC (weight 0.8) were a `known_gap` on 95% of the set the
weights were tuned on, and their weights were never actually tested.

The first cross-modal corpus built to fill that gap (`rows_crossmodal_v1`)
**muxed bonafide audio onto real clips and spoof audio onto fake clips** —
so the audio label was a deterministic function of the video label. That
inflated the cross-modal gates by a large margin. A leak-free rebuild
(`tools/build_crossmodal_corpus_v2.py`, which distributes the same audio pool
across both classes and asserts the marginals match) measured the truth:

| Signal | leaked corpus AUC | leak-free AUC | verdict |
|---|---|---|---|
| effort (video-only) | 0.6328 | 0.6511 | the only video signal with real separation |
| frameheur (video-only) | 0.5233 | 0.5350 | noise |
| av_risk | 0.6422 | **0.4744** | below chance once the leak is gone |
| havic | 0.9461 | **0.6272** | the 0.9461 was label leakage |

**Fixes applied (2026-10-21):**

- `frameheur.prob`'s affine calibration was **amplifying a noise signal 1.97x**,
  so it contributed up to 0.76 while the strongest real signal (EFFORT)
  contributed at most 0.43 — the noise outweighed the evidence. Slope is now
  1.0, with no invented amplification.
- `av_risk_addition`'s weight dropped 3.5 → 1.5, and a **corroboration gate**
  now discounts the *decorrelated* band (0.45) when no video-side signal
  agrees with it. The *anti-correlated* band (0.5) is untouched: that is the
  genuine deepfake signature — all 3 of the operator's own AI videos read
  0.5, and all 4 genuine operator clips read 0.0/0.1.
- The `disagreement` metric compared **calibrated** values across two
  differently-scaled detectors, so two detectors that agreed produced a
  spread of 0.39 and tripped the abort on genuine clips. With the
  calibrations now on comparable scales the calibrated spread is meaningful
  again.

**Measured outcome** on the leak-free corpus (60 rows, production code
paths): **0 genuine clips read DO_NOT_USE** (was 2), **0 fakes read TRUST**,
and the operator's 3 AI videos now read `do_not_use` end-to-end through the
CLI (they previously read `trust`).

**The honest limit:** this corpus cannot yield both zero false positives and
full detection. `real_006` and `fake_006` differ by 0.003 in calibrated
EFFORT, so no threshold on the video-side signals separates them. The gate
therefore keeps 3 of 30 fakes flagged and misses the rest to CAUTION. Treat
the video channel as **single-signal plus a cross-modal corroborator**, not
as a detector with demonstrated separation.

**Where each detector still falls short** — stated because a security tool
that only publishes its wins is not telling you the truth:

- **SPAI misses 8 of 30 AI images outright** (scores below 0.05): 3 flux,
  2 midjourney v6.1, 2 stable-diffusion-3, 1 gigagan. It also false-highs
  6 of 30 real photos. This is why a second, independent image signal
  (NYUAD, 3-class ViT) corroborates SPAI rather than trusting it alone.
  Measured: **AI images reading clean drop from 11/30 to 3/30** with NYUAD,
  because the two detectors miss *different* images.
- **`freqband` is not "dead noise" — it is INVERTED.** The old comment
  claimed it was uninformative; measured on the raw 60-image corpus it is
  anti-correlated: real photos score **higher** than AI images (median 0.4625
  vs 0.3990, AUC(fake=high) **0.1222**). Its positive weight therefore
  pushes genuine photos toward more risk. It is also **inert**: flipping the
  sign, flipping it via calibration, and removing it outright all give
  **identical verdicts on all 60 images** (0 rows changed), because the image
  channel caps at CAUTION and the clean-side guard never fires (max freqband
  on any image is 0.5150, below its 0.60 threshold). It is left in place
  deliberately, but it is a landmine: a "tighter" 0.45 threshold would block
  **21 of 30 genuine photos and 1 of 30 AI images** — the wrong way round.
- **EFFORT still barely separates** (AUC 0.7789 after the input fix and the
  face crop). The
  87-clip calibration corpus behind the older operating point has **no audio
  stream on 83 of its 87 rows**, so the two cross-modal gates (AV-sync,
  weight 3.5; HAVIC, weight 0.8) were never actually measured on 95% of the
  set the weights were tuned on. The older "42/42 reals not flagged" figure
  is true only because the clean-side floor is set to avoid false positives
  — it is not evidence that EFFORT separates.
  **Treat the video channel as single-signal plus a cross-modal
  corroborator.** EFFORT is a *face*-forensics model and is weak on
  full-scene AI video; the upstream paper applies it to face regions, which
  this pipeline does not do.
- **With the corrected input, effort alone still cannot separate the
  operator's own AI videos** from some genuine clips — the AI videos score
  0.54–0.65 on the face crop while some genuine clips score higher. The
  **cross-modal** signal is what flags them (their audio is
  anti-correlated, av=0.5). That is the documented design, and the real AI
  videos do read `do_not_use` end-to-end (3/3 HIGH). But on the leak-free
  corpus 11 of 30 fakes still read TRUST, because a muxed audio track
  decorrelates rather than anti-correlates and never reaches the
  anti-correlated band. **A corpus with naturally synchronised audio is
  the missing instrument.**
- **The face crop also raises SOME genuine clips** (a low-quality phone
  face can read as synthetic: ffc23 real 451 went 0.421 → 0.775 on the
  whole frame vs the crop). That is a real false-positive cost, and it is
  why the offset was re-derived a second time.

**Test suite:** **605 passed / 1 skipped** on a developer machine (the service
venv: Python 3.11, torch 2.14.1+cu130, all learned detectors installed, and
the legacy `pylibs` tree present). In **CI** — a clean checkout with only the
declared dependencies — the same suite measures **506 passed / 31 skipped**
(bare, no torch) and **582 passed / 21 skipped** (with torch). The bare run
skips the six torch-gated architecture modules wholesale, so a bare run is
*not* evidence that the detectors work; CI asserts both counts separately
with environment-specific floors (495 bare / 575 full).

**Run the suite with `scripts/run_tests_split.sh` on a memory-constrained
host.** The full suite in ONE process is OOM-killed by the cgroup on a 25 GB
machine (each torch checkpoint costs ~4 GB RSS; the kill appears as exit -9
with an empty pytest output and `dmesg` reports "Memory cgroup out of
memory"). The script splits the run into two processes that between them
cover every test file (54 + 7 = 61, no gap) and prints the usual pytest
output for each.

Two CI gates run on every push, and a third checks the shipping invariants
(the aasist3 helper path exists, shell scripts are LF, the launcher keeps
pylibs off the PYTHONPATH, no secret file is tracked). `master` is
branch-protected: a direct push is rejected, and a PR must pass all three
checks to merge.

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
