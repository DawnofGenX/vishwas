# Performance & Thermal Operations Guide

Target hardware profile this document is written for: **Intel i5-8250U** (4 cores / 8 threads, ~15W sustained TDP), laptop-class memory bandwidth, CPU-only inference (no GPU by design — zero-cloud constraint). The system has hard-tripped at ~100C when two concurrent Hermes tasks drove sustained load; everything below keeps steady-state work comfortably inside that envelope.

## 1. Timing budget model

Every job runs under one wall-clock deadline:

| Knob | Default | Source | Meaning |
|---|---|---|---|
| `VERISAFE_BUDGET_S` / `--budget-s` | **300s** | env / CLI flag | Hard deadline from job start; checked before every stage (`JobContext.expired()`) |
| Stage floor | **10s** | `_run()` in `orchestrator.py` | A stage is never *started* with <10s remaining — it records `*.timeout` (status `skipped`) instead, so partial work can't straddle the deadline |
| Subprocess timeouts | per-call | each capability | e.g. clamscan 90s, ffprobe 15s, browser fetch ≤ 2× redirect hops × 6s |

Per-stage wall time is recorded into `JobOutcome.fusion_trace.stage_timings_s` and the early-stop marker into `fusion_trace.short_circuited_at`. These fields are what the ops dashboard / incident log consume.

### Budget allocation guidance (default 300s)

| Stage family | Typical cost | Shares of budget |
|---|---|---|
| Magic-byte validation | <1ms | ~0 |
| URL normalise + SSRF guard (DNS, 2 hops max) | 0.3–3s | ~1% |
| Phishing heuristics (stdlib only) | <50ms | ~0 |
| VirusTotal lookups (file or URL) | 1–4s each | ~2% |
| ClamAV scan | 1–60s (size-dependent) | up to ~20% |
| YARA-x scan | 0.1–5s | ~1% |
| Media probe (ffprobe) | 0.2–1s | <1% |
| Transform battery (ffmpeg ×N variants) | 1–15s per variant set | up to ~15% |
| Offline audio features (librosa-free numpy path) | 1–4s | ~1% |
| AASIST learned audio (WavLM-Large, 3 crops) | **~15–23s** (measured 2026-08-21) | ~8% |
| EFFORT learned video (ViT-L/14 303M + OrthAlign) | **~1.7s/frame** (measured 2026-08-21); ~5s for a 3-frame T2 stage | ~2–4% |
| Deepfake video models (when weights available) | EFFORT ~1.7 s/frame (measured); HAVIC cross-modal **~5.1 s/clip** (measured 2026-08-21) | now minor, not dominant |
| LLM interpretation (gated, optional) | 5–30s | ≤10% |

**Rule:** heavy-model stages only start if ≥ their own expected runtime remains in the budget; the 10s floor above makes this automatic rather than ad-hoc.

## 2. Thread caps (thermal-safe)

| Binary / process | Env var | Default | Rationale |
|---|---|---|---|
| ffmpeg | `VERISAFE_FFMPEG_THREADS` | **2** | >2 threads measurably pushes the 8250U toward its thermal ceiling under sustained encode/decode; 2 gives most of the throughput on single-file workloads |
| ffprobe | inherits | 2 (same flag) | probing barely uses threads anyway |
| Python pool for heavy stages | `_HEAVY_POOL` in `orchestrator.py` | 2 workers | bounds concurrent model runs to what the i5-8250U can sustain without tripping (~89C was observed as the max in the Aug-7 incident; we want <80C steady) |
| Webhook server | stdlib `ThreadingHTTPServer` daemon threads | unbounded accept, 2 compute pool | accepts fast, computes bounded |

The thermal monitor script (`~/.hermes/scripts/thermal-monitor.sh`) logs core temps every 5s; the cron watchdog alerts on CRITICAL. If you see sustained >80C during VeriSafe batches, drop `VERISAFE_FFMPEG_THREADS` to 1 and/or lower `_HEAVY_POOL` before scaling volume.

## 3. Conservative short-circuit (P8)

Once a **confirmed** positive finding settles the user-facing answer, remaining *heavy/unknown* stages are skipped and marked `skip_early_stop`. "Confirmed" is deliberately stricter than a single heuristic:

| Trigger (from the just-finished stage's batch) | Why it qualifies as confirmed |
|---|---|
| `clamscan` with `detected=True` | signature match = AV ground truth for known malware |
| `yara_x` `hits_norm > 0.6` **plus** `quark_engine > 0.4` or `pe_statics.packed` | two independent static-analysis families agree |
| `vt_reputation` `prob_malicious ≥ 0.5` | multi-engine reputation consensus |
| `phish_heuristics` `score_norm > 0.7` **plus** corroborating `ssrf_guard` degraded or suspicious redirects | strong phish signal with independent network-side evidence |

Stages that still run after a trigger (light analysis families, currently `GovDocumentCapability` and `UrlPhishingCapability`) are cheap enough that their evidence still improves the report. The verdict is computed over *all* checks regardless — the short-circuit saves CPU heat, it does not delete evidence or alter scoring.

**When it does NOT fire:** a single medium-confidence heuristic alone (e.g. yara 0.6 with no corroboration, or phish 0.65 with clean network evidence). In those cases all stages run and disagreement is surfaced through the reliability gate instead.

## 4. Short-circuits elsewhere (pre-existing)

- **SSRF block** in URL normalisation aborts the URL pipeline immediately (no redirect chasing beyond the block point).
- **Near-silence audio** (< RMS threshold) abstains from audio-deepfake scoring instead of burning transform-battery cycles on an empty signal.
- **Dependency gates**: every capability declares `requires=(...)`; missing binaries/API keys turn a whole stage into an `unavailable` record in <1ms instead of failing mid-run.
- **Subprocess isolation**: hung codecs die at their per-call timeout; the job continues on other branches.

## 5. Incident-response notes (this box)

- Symptom: machine suddenly powers off with no OS-level shutdown log → silicon thermal trip (~100C), not an OOM or driver crash.
- First thing to check: `tail ~/.hermes/logs/thermal_power_monitor.log`.
- Two concurrent Hermes sessions were the Aug-7 trigger; VeriSafe batches share that risk. Before any bulk job: run the precheck script, keep `VERISAFE_FFMPEG_THREADS=2`, avoid overlapping with other CPU-heavy Hermes work.

## 6. Measured model timings (this box, CPU-only)

| Model | Fixture | Load | Inference | Tier | Date |
|---|---|---|---|---|---|
| AASIST (HABLA_WavLM_AASIST: WavLM-Large 315M + HtrgGAT) | `tests/fixtures/audio/smoke_5s.wav` (5 s @16 kHz mono) | 15.7 s (3.79 GB checkpoint → RAM + arch build) | **15.3 s** single inference (1 crop); ~23 s for the full 3-crop T2 stage in E2E | normal (cap 120 s; not SLOW) | 2026-08-21 |
| EFFORT (CLIP-style ViT-L/14 303M + OrthAlign rank-1 residuals) | synthetic 224×224 frames + 2 known-real photos (label-order check) | 2.6 s skeleton build + 1.1 s torch.load (1.21 GB checkpoint) + 0.3 s apply_state (681/681 keys, 0 missing/unexpected) | **~1.7 s** per 224×224 frame; ~5 s for a 3-frame T2 stage | normal (cap 180 s; not SLOW) | 2026-08-21 |

Notes: thermal held at 37 °C (acpitz) throughout the AASIST smoke — no SLOW-tier marking needed per Decision #3. The 3-crop T2 stage (`aasist_detector`) ran inside the default 300 s job budget with ~8% share. EFFORT verify ran at 55 °C peak (normal tier); label order confirmed [real, fake] on known-real photos (fake posterior 0.29–0.35). Evidence: `docs/research/ARCH_VENDOR_EVIDENCE_2026-08-21.md`.

## 7. End-to-end CLI re-measure — full current build (all learned stages live)

Re-measured 2026-08-21 (Phase 5 Task A) with all three weight gates provisioned
(AASIST + EFFORT + HAVIC real checkpoints), `VERISAFE_FFMPEG_THREADS=1`,
sequential runs, via `scripts/run_verisafe.sh cli`. Fixture: fresh 4 s
`testsrc`+`sine` clip (640×360 @ 8 fps, h264+AAC) — the earlier P1 fixture is
purged by zero-retention design, so it was re-synthesized to the same recipe.

| Run | Input | E2E wall (CLI) | `wall_s` | Stage timings (`stage_timings_s`) | Learned scores seen |
|---|---|---|---|---|---|
| (a) audio | `tests/fixtures/audio/smoke_5s.wav` | **23.9 s** | 22.97 | `DeepfakeAudioCapability` 22.97 | aasist 0.997 (3 crops); degradation battery ok |
| (b) AV clip | 4 s testsrc+sine mp4 | **35.8 s** | 34.77 | `DeepfakeVideoCapability` 21.68, `CrossModalCapability` 13.08 | effort 0.378 (8 frames), havic 0.993 |
| (c) text URL | `--text "https://example.com"` | **0.58 s** | 0.18 | — (light path only) | n/a |

Full hermetic suite same day: **313 passed, 3 skipped in 18.84 s** pytest wall
(19.7 s incl. interpreter start), thermal ≤50 °C throughout all runs.

Reading of the numbers:
- Audio E2E dropped vs the earlier-today 42.1 s P1 proof run (→ 23.9 s): that
  run paid first-process cold page-in for the 3.79 GB WavLM checkpoint; the
  stage itself (3-crop AASIST + offline features + degradation battery) is
  unchanged at ~23 s. The §1 budget row (~15–23 s) stands.
- Video side splits as expected: EFFORT dominates `DeepfakeVideoCapability`
  (8 frames × ~1.7 s/frame ≈ 14 s of the 21.7 s stage), HAVIC ~5 s/clip inside
  the 13.1 s cross-modal stage (rest is frame/wav extraction + heuristic probe).
- **No stage exceeded its budget tier** against the default 300 s job deadline
  (worst case 35.8 s ≈ 12% share) — no budget-bump proposal required. The §1
  allocation table remains accurate as written.

Adversarial sensitivity of these learned scores under codec/scale/framerate
transforms is measured separately in
`docs/research/DEEPFAKE_DETECTION.md` §4.1 (same date).

- Undervolt config `thermald-undervolt.xml` is staged for `sudo` install if idle temps creep above ~53C.

## 8. Measured model timings — RTX 5090 Laptop, CUDA (2026-08-24)

Hardware/runtime change: migration to the new disk brought a GeForce RTX 5090
Laptop GPU (24 GB, sm_120). Commit `5c80af7` added a device-resolution seam
(`src/verisafe/device.py`: `VERISAFE_DEVICE` env > cuda-if-available > cpu);
all learned-stage loads and inference now run on CUDA by default. Weights were
re-downloaded fresh (AASIST 3.79 GB, HAVIC 859 MB); the fetcher's size-verify
unit bug (MiB vs decimal MB) was fixed in `scripts/fetch_model_weights.sh`.

| Measurement | CPU era (§6–7) | RTX 5090 (this table) |
|---|---|---|
| AASIST model load (3.79 GB ckpt) | ~16 s | **7.2 s** |
| AASIST single score (2 s noise probe) | ~1.7 s/frame-class | **2.82 s total incl. first-CUDA-init** |
| Full DeepfakeAudioCapability E2E (smoke_5s.wav) | ~23.9 s | **6.51 s** (incl. model load + 6-variant degradation battery); re-run 2026-08-25: 9.6 s stage wall incl. offline-features check |
| EFFORT model load (1.21 GB ckpt, ViT-L/14 303M) | — | **1.6 s**; 5-crop batch score **0.29 s** (probe /tmp/probe_effort.py recipe, 2026-08-24) |
| XLSR-Mamba load (1.28 GB safetensors, strict 565/565) | — | **2.7 s**; 4 s waveform score **0.65 s cuda / 1.24 s cpu-first-call** (`scripts/verify_xlsrmamba.py`, commit fa07bcc) |
| Text URL light path | <1 s | <1 s (unchanged) |

Gate posture after this table's date: ALL FOUR learned gates live on CUDA —
AASIST (audio), EFFORT chameleon (video/face, fetched 2026-08-24 via Drive API
`files.copy` bypass of the public download-quota wall), HAVIC (av-crossmodal),
XLSR-Mamba-LA (audio 2nd opinion, MIT, replaces RawBMamba as Mamba-slot
primary; label-order inversion bonafide=1 documented in its spec header X5).
RawBMamba demoted to eval-grade ONNX fallback (no license). Webhook defaults to
`VERISAFE_DEVICE=cpu` (risk-6: LLM serving shares the 5090); GPU is opt-in per
invocation and `/health` reports the resolved `device`.

Notes:
- Interpreter decision: docling-python's torch 2.13.0+cu130 initializes CUDA
  correctly on sm_120 (verified: device name + capability + matmul), so
  `run_verisafe.sh` PYTHONPATH is unchanged; no vllm-env switch needed.
- The webhook systemd unit previously had NO weight-gate env at all (gates
  lived in ~/.bashrc, invisible to systemd — same failure class as the VT key).
  Fixed durably: provision output inlined into `deploy/verisafe-secrets.env`.
- New gate registered: `VERISAFE_XLSRMAMBA_WEIGHTS` (MIT, XLSR-Mamba-LA,
  replaces RawBMamba as Mamba-slot primary once its arch module is vendored —
  separate commit cycle). RawBMamba demoted to eval-grade ONNX fallback.
- EFFORT checkpoint pending (Google Drive quota window on the only public
  host); env line staged, gate will light up when uncommented.
- Suite same day: **339 passed, 3 skipped** (baseline 333+3 +6 device-seam tests).
