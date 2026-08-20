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
| Deepfake video models (when weights available) | 60–240s | dominant remainder |
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
- Undervolt config `thermald-undervolt.xml` is staged for `sudo` install if idle temps creep above ~53C.
