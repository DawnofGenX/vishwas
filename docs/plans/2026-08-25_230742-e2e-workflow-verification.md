# End-to-End Workflow Verification Plan (post-fixes)

> **For Hermes:** Use subagent-driven-development to implement task-by-task. Each task verifies one stage of the pipeline with real execution and recorded evidence.

**Goal:** Systematically verify every layer of the Vishwas pipeline — unit suite, each detector gate, fusion logic, calibration behavior, CLI E2E, webhook health, and the live WhatsApp round trip — so the operator has proof the whole workflow works after today's fixes, not just green unit tests.

**Architecture:** A staged verification ladder: (1) hermetic unit suite → (2) per-gate detector proofs on real weights → (3) fusion decision-table replay over measured corpora → (4) CLI end-to-end on known media → (5) service-level checks (webhook + openwa) → (6) live WhatsApp round trip. Each stage produces a pass/fail artifact under `/tmp/verification/` that the final report links. No fix is trusted until its stage passes; a failed stage blocks all later stages that depend on it.

**Tech Stack:** pytest (existing suite), bash + curl for service probes, ffmpeg for synthetic media, the production capabilities via `PYTHONPATH="/home/hermes/pylibs:/home/hermes/docling-python:src"`, GPU (`VISHWAS_DEVICE=cuda`) for weight-bearing gates, `scripts/run_vishwas.sh cli` for E2E, openwa gateway API for the live leg.

---

## Current context / assumptions

- HEAD `f627e70`; suite baseline **365 passed / 7 skipped**; webhook restarted after Fusion-v2 + recalibration commits.
- Fixes to be verified (this session's commits): `7efa7fb` prenorm frontend config; `c2dfe14` XLSR zero-pad; `86dd969`+`ca30446`+`3d40116` Fusion v2 + coherent gate + i18n patterns; `394aa08`+`f627e70` real-video recalibration + demamba retirement; XLSR score unwrap.
- Measured corpora available: 240-clip ASVspoof-2019 audio set (`/home/hermes/fusion_audio/asv19_la/clips/`, protocol.csv), 84-row video feature corpus (`rows_video_v2.jsonl`), FF++ subsets (`subset/real2|fake2`), AI seed video `/tmp/ops_video.bin`.
- The AASIST-fixer agent (deleg_6e3fd88d/task-0) is still running its final 240-clip proof. **Stage G below is written conditionally**: it executes either (a) the agent's landed fix, or (b) documents the checkpoint-swap path if the conclusion is "checkpoint degenerate."
- Known-true negatives to expect: HAVIC saturated (~1.0 everywhere); EFFORT cannot rank FF++ face-swaps (AUC 0.385 inverted). Verification asserts *documented* behavior, not ideal behavior — e.g., FF++ fakes land CAUTION (not DNU) is a PASS.
- Operator-side items that stay manual: VT key rotation, `/opt` sudo mv, physical second phone for the V2 round trip.

---

## Proposed approach

Ten verification stages, strictly ordered. Every stage writes `/tmp/verification/<stage>.txt` with full output and a final line `STAGE_<X>: PASS` or `STAGE_<X>: FAIL <reason>`. Stage 0 writes a manifest consumed by the final summary script. Any FAIL stops the ladder at that point; diagnosis happens before proceeding.

---

## Step-by-step plan

### Task A: Stage 0 — environment & repo manifest
**Objective:** snapshot exact state being verified (commit, suite env, weights present, services up).
**Steps:**
1. Record `git rev-parse HEAD` + `git status --short` into `/tmp/verification/stage0.txt`.
2. Verify weight files exist: `$VISHWAS_AASIST_WEIGHTS`, `$VISHWAS_EFFORT_WEIGHTS`, `$VISHWAS_HAVIC_WEIGHTS`, `$VISHWAS_XLSRMAMBA_WEIGHTS` (source `deploy/vishwas-secrets.env` + provisioner first).
3. Record `/health` JSON.
**Pass:** all four weight paths exist; `/health` = ok; tree clean or only known-untracked artifacts.

### Task B: Stage 1 — full hermetic unit suite
**Objective:** prove the committed code is internally consistent.
**Command:** `cd ~/vishwas && PYTHONPATH=src python3 -m pytest tests/ -q`
**Pass:** `365 passed, 7 skipped` (or more if new tests added by then; zero failures).

### Task C: Stage 2 — detector gates on real weights (input-sensitivity quartet)
**Objective:** each learned gate responds to input and matches documented polarity.
**Steps:** run `tests/test_34_audio_input_sensitivity.py` with weights sourced on CUDA (expect 3 passed); plus direct probes: AASIST(silence) ≠ AASIST(noise) ≠ AASIST(sine440); XLSR same trio distinct; EFFORT(AI frame crop) > EFFORT(testsrc control) — assert effort separates the seeded AI/control pair; HAVIC probe recorded as *saturated-by-documentation* (no assertion beyond "loads and returns float").
**Files:** reuse `/tmp/aasist_invariance.py` pattern; write `/tmp/verification/gate_probes.py`.
**Pass:** test_34 green + three distinct audio posteriors per model + EFFORT AI>control.

### Task D: Stage 3 — fusion decision table replay
**Objective:** prove the recalibrated fusion produces the exact expected verdict per named scenario, including both Task-1 contract halves.
**Steps:** run tests/test_35 + test_30 + test_05 verbosely; THEN replay all 84 rows of `rows_video_v2.jsonl` through the real `FusionEngine` and diff against the accepted operating-point distribution: reals NOT-DNU ≥80% (34/41), fakes ≥CAUTION 100%, AI row DO_NOT_USE + fully_generated.
**Pass:** all scenario tests green AND corpus distribution within tolerance (±1 row).

### Task E: Stage 4 — CLI end-to-end on known media (GPU)
**Objective:** prove the shipped CLI path produces correct verdicts on ground-truth files.
**Steps:** run `bash scripts/run_vishwas.sh cli --file /tmp/ops_video.bin --media-type video` (AI video) → assert verdict DO_NOT_USE, confidence band high/moderate, reply contains the fully_generated explanation sentence. Run the two synthetic-real controls (`fusion_av/seeded/*.mp4`) → assert NOT do_not_use. Extract timings for the perf note.
**Pass:** AI=DNU w/ pattern sentence; controls ∈ {CAUTION, TRUST, UNABLE}.

### Task F: Stage 5 — service level (webhook + openwa bridge)
**Objective:** prove the deployed service runs this code and can talk to WhatsApp infra.
**Steps:** restart `vishwas-webhook`; poll `/health` until ok (deps=12, device reported); POST a synthesized URL-check request through the local webhook auth path (HMAC-signed, localhost) for a benign domain and assert a verdict JSON returns; verify openwa :2785 session status `ready` via its status endpoint (read-only).
**Pass:** health ok post-restart; signed local check returns valid verdict envelope; openwa reports connected session.

### Task G: Stage 6 — audio gate resolution (conditional on agent report)
**Objective:** close out the AASIST saturation work with whichever outcome is true.
**Branch (a) fix landed:** apply/review agent diff → independently rerun its 240-clip proof script → require AUC ≥0.85 AND non-degenerate quantiles (real p25..p75 not overlapping fake medians) → run suite → commit.
**Branch (b) checkpoint degenerate:** document in docs/research/FUSION_DATASETS_2026-08-25.md §AASIST: root cause statement + exact checkpoint-swap steps (candidate sources, license check, drop-in env path) — no code change.
**Pass:** branch (a) merged w/ independent proof, OR branch (b) doc committed with actionable swap plan. Either way the audio story is closed with evidence.

### Task H: Stage 6b — ASVspoof 2021 extraction readiness check (optional, non-blocking)
**Objective:** confirm the downloaded pool is intact for future scaling.
**Steps:** `tar -tzf /home/hermes/fusion_audio/asv21_la_eval/ASVspoof2021_LA_eval.tar.gz | head -5` + count entries; record in verification log. Do NOT extract fully unless disk/time allow.
**Pass:** tar lists >10k flac entries without error.

### Task I: Stage 7 — live WhatsApp round trip (operator-assisted)
**Objective:** the true E2E — a real message from the operator's phone gets a concrete verdict back.
**Steps:** operator sends one AI-generated video (with speech) + one ordinary real photo to the paired number; assistant polls outcomes log (`/tmp/vishwas-work/outcomes.jsonl`) for the two new jobs; asserts video→DO_NOT_USE-family reply with pattern sentence lands in WhatsApp, photo→non-DNU reply; captures message timestamps as evidence.
**Pass:** both replies delivered and logged; verdicts consistent with Stages D/E expectations.

### Task J: Final consolidated report
**Objective:** one artifact the operator can read.
**Steps:** aggregate all stage files into `/tmp/verification/REPORT.md`: table stage→result→evidence pointer→runtime; list any FAILs with diagnosis; list remaining known limitations (HAVIC saturation, EFFORT-vs-FF++ limitation, audio-channel state per Stage G outcome).
**Pass:** every prior stage has an entry; open items explicitly listed with owner (assistant vs operator).

---

## Files likely to change

- Possibly `src/vishwas/model_archs/aasist.py` (+`_wavlm/`) — only via Stage G branch (a)
- `docs/research/FUSION_DATASETS_2026-08-25.md` — Stage G branch (b) documentation
- `/tmp/verification/*` — new evidence artifacts (not committed)
- Skill `verisafe-operations` — version bump after ladder completes

## Tests / validation

- Hermetic: existing 365-test suite is Stage B's gate.
- Weighted: test_34 (audio sensitivity) re-run on GPU in Stage C.
- Corpus: 84-row replay distribution check in Stage D (tolerance ±1 row).
- E2E: CLI verdict assertions in Stage E; live round trip in Stage I.
- Every stage emits machine-readable PASS/FAIL to block the ladder.

## Risks, tradeoffs, open questions

- **GPU contention:** the AASIST agent is still running CUDA jobs; Stages C/E must wait for it to finish or serialize behind it — do not run heavy gates concurrently (VRAM pressure + timing skew).
- **Stage I depends on the operator's phone availability**; it is the only stage that can't be autonomous. Everything else can complete without it.
- **Tolerance choice in Stage D:** ±1 row allows borderline-threshold flips; tighter tolerance would make the ladder brittle to float nondeterminism on CPU vs GPU.
- **If Stage G concludes checkpoint-degenerate,** the audio channel stays calibration-only until a known-good checkpoint is sourced — the plan records this honestly rather than shipping an unverified model swap.
- **Open question:** should the live round trip use a fresh AI video from the operator (preferred — adds an unseen sample) or the same one already verified? Default: fresh if available, else reuse.
