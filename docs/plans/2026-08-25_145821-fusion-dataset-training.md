# Dataset-Driven Fusion Training — Implementation Plan

> **For Hermes:** Use subagent-driven-development to implement this plan task-by-task after the `video` dataset research report lands.

**Goal:** Assemble a labeled real/fake dataset, run Vishwas's local detectors over it to build per-detector feature vectors, and train a real logistic fusion stack (`stack_<target>.json`) that FusionEngine auto-loads — replacing the single-sample calibration heuristics with empirically-fit posterior combination.

**Architecture:** The base deepfake detectors (AASIST, XLSR-Mamba-LA, EFFORT, HAVIC, frame/cross-modal heuristics) are already fine-tuned and locally vendored. We do **not** retrain them. We gather labeled clips (real vs fake, and for AV: *which modality* was faked), run each clip through the detectors to get a posterior vector, and feed those (vector, ground-truth) rows into the existing stdlib trainer (`fusion_train.py --dataset`) which does leave-subset-out cross-validation + temperature/bias calibration and writes `fusion/training/stack_<target>.json`. FusionEngine already supports a per-target LR-stack override on top of the calibrated weighted baseline.

**Tech Stack:** Python 3.12 stdlib (`fusion_train.py` needs only numpy optionally) · wget/curl + HF `huggingface-cli` for dataset pulls · pyarrow+soundfile for ASVspoof parquet slicing · ffmpeg for audio extraction/16 kHz normalization · our existing detector adapters (`model_archs/`) for feature extraction.

---

## Current context / assumptions

- Box: WSL Linux, RTX 5090 (24.5 GB VRAM), 712 GB free on `/`. Detectors are input-sensitive now (AASIST/XLSR prenorm fix landed `7efa7fb`); Fusion v2 pattern-aware fusion live (`86dd969`, `ca30446`, `3d40116`).
- Trainer verified working: `PYTHONPATH=src python3 -m vishwas.fusion_train --synthetic 200 --target deepfake_video` wrote `fusion/training/stack_deepfake_video.json` (deleted test artifact; only real `stack_url_phishing.json` remains). Pipeline is ready; only **data** is missing.
- **License boundary:** ASVspoof 2019/2021 are **ODC-BY** (local inference + reading fine). Common Voice is **CC0**. FakeAVCeleb / AV-Deepfake1M are **research-only / CC-BY-NC** — fine for *evaluation/training-internal* on this private operator box, but must be documented as research-only and /not/ shipped in any commercial release. VoxCeleb2 mirrors are CC-BY-SA.
- **Detector separation already observed** (single-sample, 2026-08-25): EFFORT AI-video 0.677 vs control 0.302; frame_heuristics 0.281 vs 0.078; HAVIC saturates ~1.0 (near-uninformative, deweighted); AASIST/XLSR now input-sensitive but clean-side sample was a synthetic sine (not speech) → need real-speech bona-fide to calibrate audio reliably.
- **Fusion-gap already fixed** (deepfake_video now weights av_risk_addition + havic.prob_inconsistent); this plan's trained stack will *replace* the hand-set weights + pattern thresholds with empirically-fit coefficients per target.
- The `video` dataset research report is still in flight (3rd agent). **This plan intentionally front-loads the audio+AV work that does not depend on it**; the video-focused tasks (Task 7) will use whichever video dataset the pending report recommends (likely DFDC-preview or FaceForensics++ for face-swap, plus labeled full-AI clips).

---

## Proposed approach

1. **Acquire the two bootstrappable labeled sets now** (no approval, same-day):
   - **ASVspoof 2019 LA** via HF parquet mirror (`Bisher/ASVspoof_2019_LA`, validation parquet ~1.58 GB) — balanded real/fake 16 kHz audio for AASIST/XLSR. (ODC-BY.)
   - **Common Voice 'hi'** (CC0) — real human speech in the operator's language, for the bona-fide side and cross-lingual false-positive probing.
2. **Background-download the bigger labeled audio pool** — ASVspoof 2021 LA eval (7.6 GB Zenodo, ODC-BY, all clips labeled).
3. **Request access now (blocking, run in parallel):** FakeAVCeleb (Google form) and AV-Deepfake1M (HF EULA gate) for the 4-class AV set (real / face-only / voice-only / both-fake). Don't block the pipeline on approval — use VoxCeleb2 mirror (real AV, ungated) as the interim real-AV baseline.
4. **Build a feature-extraction harness** that runs the local detectors over a directory of labeled clips and writes `rows.csv` = `path,label,subclass,[post_<detector>...],gt`.
5. **Train + validate the stacks** per target (`deepfake_audio`, `deepfake_video`) with OOF cross-val + calibration; assert AUC/ECE improvement over the current calibrated-heuristic baseline.
6. **Regression-anchor:** operator's own 10.9 s AI video stays a scenario — must remain DO_NOT_USE high-confidence 'fully_generated'.
7. **Wire in + restart webhook**, document licenses in `docs/research/`, and record the trained-stack model_ids surfaced in fusion reasons.

---

## Step-by-step plan

### Task 0: Stability checkpoint (before any download)
**Objective:** confirm the tree and detectors are stable so a bad dataset doesn't corrupt state.
- Run: `cd /home/hermes/vishwas && git status --short` → expect clean.
- Run: `PYTHONPATH=src timeout 280 python3 -m pytest tests/ -q` → expect `363 passed, 7 skipped`.
- **Auto-skip risk:** if any test fails, fix or revert BEFORE data work.

### Task 1: Download ASVspoof 2019 LA labeled slice (HF parquet mirror)
**Objective:** get ~240 labeled 16 kHz real/fake clips same-day.
- **Files:**
  - Create: `/opt/fusion_audio/asv19_la/asv19_val.parquet`
  - Create: `/opt/fusion_audio/scripts/slice_asv19.py`
- **Step 1** mkdir + wget (resume-capable):
  `mkdir -p /opt/fusion_audio/{asv19_la,asv21_la_eval,feat_vectors,scripts}`
  `cd /opt/fusion_audio && wget -c -O asv19_la/asv19_val.parquet 'https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA/resolve/main/data/validation-00000-of-00001.parquet'`
- **Step 2** write `scripts/slice_asv19.py`: read parquet (pyarrow), take balanded 120 bonafide + 120 spoof rows, decode `audio` (16 kHz) + `key` (0=bonafide) + `system_id`, write WAVs under `asv19_la/clips/{bonafide|spoof}/` and a `asv19_la/protocol.csv` {path,label,subclass}.
- **Step 3** run it; verify ~240 WAV files exist and are 16 kHz (`ffprobe` a few).
- **Verify:** `wc -l /opt/fusion_audio/asv19_la/protocol.csv` ≈ 241 header-inclusive; file sizes sane.

### Task 2: Background-download ASVspoof 2021 LA eval (main pool)
**Objective:** start the large labeled pool; don't block on it.
- Run as background terminal (`notify_on_complete=true`):
  `cd /opt/fusion_audio && wget -c -O asv21_la_eval/ASVspoof2021_LA_eval.tar.gz 'https://zenodo.org/api/records/4837263/files/ASVspoof2021_LA_eval.tar.gz/content'`
- **Step** after download: extract; fetch `keys.tsv` + `meta.tsv` from asvspoof.org (2021 eval protocol). Park; process a slice in Task 4 optionally.

### Task 3: Download Common Voice 'hi' (CC0 real speech)
**Objective:** real human speech in the operator's language for bona-fide + FP probing.
- Download per-language archive from `https://commonvoice.mozilla.org/en/datasets` (or HF `mozilla-foundation/common_voice_17_0`).
- Extract; keep `validated.tsv` mapping + audio dir. Pick ~100 'hi' clips.
- **Verify:** `ffprobe` clipping shows 16 kHz/32-bit; protocol has `client_id` for speaker variety.

### Task 4: Feature-extraction harness → fusion rows
**Objective:** turn labeled clips into (vector, gt) rows the trainer consumes.
- **Files:**
  - Create: `/opt/fusion_audio/scripts/extract_rows.py`
  - Create (output): `/opt/fusion_audio/feat_vectors/rows_audio.csv` and later `rows_av.csv`
- **Behavior** (read-only vs repo; sets PYTHONPATH to vishwas `src` + pylibs + docling-python; uses `resolve(<ENV>).load(path)` + `model.predict(waveform)` like the verified `audio_recal.py`):
  - input: `protocol.csv` (or `protocol_av.csv`)
  - for each clip: run AASIST → `post_aasist`, XLSR-Mamba → `post_xlsr`, offline heuristics → `post_heur`; append `clip,label,subclass,post_aasist,post_xlsr,post_heur,gt`.
  - For AV rows later: also EFFORT, HAVIC, cross_modal_av.
- **Verify:** rows count == protocol rows; sample a few posteriors by hand; confirm labels present (gt in {0,1}, subclass token).

### Task 5: Train per-target fusion stacks
**Objective:** emit `stack_deepfake_audio.json` / `stack_deepfake_video.json` from real data.
- Run: `PYTHONPATH=src python3 -m vishwas.fusion_train --dataset /opt/fusion_audio/feat_vectors/rows_audio.csv --target deepfake_audio`
  and `... --dataset rows_av.csv --target deepfake_video`
- Trainer does OOF LR + calibration + metrics (AUC/F1/ECE/Brier). 
- **Acceptbar:** `test_roc_auc >= 0.85`, `ece <= 0.15` on the validation split; if the single-sine clean-side poisoned audio, add the CommonVoice 'hi' real clips and warn if AUC drops (document).
- Artifacts go to `fusion/training/stack_<target>.json`. Keep a copy under `/opt/fusion_audio/artifacts/` for provenance.

### Task 6: Wire trained stacks + regression-anchor
**Objective:** serve the trained stacks safely; the operator AI-video case must not regress.
- `FusionEngine.load_trained("fusion/training")` already auto-loads stacks. Confirm serve path (orchestrator builds `FusionEngine` — verify it calls `load_trained` from `VISHWAS_FUSION_DIR`).
- Add/add-to tests: `tests/test_36_fusion_trained.py` asserts `decide("deepfake_video", <operator_video_signals>)` is DO_NOT_USE conf>=0.45 + pattern 'fully_generated', AND that the trained stack changes at least one non-degenerate decision vs heuristic baseline (guard against no-op).
- Run full suite → `363+ passed, 7 skipped`.
- Restart webhook; `/health` deps 12.

### Task 7: Video dataset (face-foot + full-AI-video class)
**Objective:** add labeled video for the face-forensics (EFFORT/frame_heuristics) AND the AV-sync foot.
- **Zero-cred now:** HF `bitmind/FaceForensicsC23` (17.9 GB, no token) → 10 real + 10 fake MP4s for the **face-swap** class. ⚠️ FF++ ships **no audio** → cross_modal/HAVIC degrade to `failed` on it; use it ONLY to validate EFFORT separation + row plumbing, not the AV dataset (report §Tier0).
- **AV foot needs audio-bearing video, all gated:** DFDC (Kaggle key), Celeb-DF v2 (Google form → gdown), AV-Deepfake1M (HF EULA gate). Pick whichever the operator's credential unlocks first; non-silent clips only (`ffprobe` filter) so AV-sync has signal.
- **Full-AI-video class (the one datasets under-serve):** fetch ~12–20 officially-released Sora/Veo/Kling/Runway demo clips (honest label `AI-generated`, label 1) + ~15 real talking-head clips from a CC source (label 0). Merge FF++ face-swap + AV-manip + full-AI rows into one JSONL so the LR stack sees all three failure modes.
- Add `effort/HAVIC/cross_modal_av` columns to `rows_av.csv`; re-run Task 5 for deepfake_video with enriched rows.

### Task 8: License + results documentation
**Objective:** make the provenance + license boundaries auditable.
- Create `docs/research/FUSION_DATASETS_2026-08-25.md`: table of datasets used, exact license links, sizes, how many clips/rows, model-IDs, AUC/ECE, and the explicit note that research-only/NC corpora are training/eval-internal only and must not ship in a commercial release.
- Commit.

---

## Files likely to change

- `src/vishwas/fusion_train.py` — (likely no change; verify `--dataset` accepts rows.csv format directly)
- `src/vishwas/fusion.py` / `src/vishwas/model_adapters.py` — only if train artifact format needs a tweak
- `src/vishwas/orchestrator.py` — verify `FusionEngine.load_trained(...)` is invoked at build
- `tests/test_36_fusion_trained.py` — NEW regression + trained-stack guard
- `docs/research/FUSION_DATASETS_2026-08-25.md` — NEW
- `/opt/fusion_audio/scripts/{slice_asv19,extract_rows}.py` — NEW harness (outside repo, data-landing dir)

## Tests / validation

- `tests/test_35_fusion_v2_scenarios.py` — must stay green (operator video anchor).
- NEW `tests/test_36_fusion_trained.py` — trained stack present ⇒ `decide()` uses it and operator-video stays confident DNU; absent ⇒ skip gracefully.
- Full suite after each data/train task: `PYTHONPATH=src timeout 280 python3 -m pytest tests/ -q`.
- CV metrics asserted in Task 5 (AUC/ECE thresholds) — printed by the trainer JSON.

## Risks, tradeoffs, open questions

- **Bandwidth unknown**: large pulls (7.6 GB+) run in background with `wget -c`; bootstrap uses the small parquet slice so the session never blocks.
- **ASVspoof 2019 eval keys are sealed** — use train+dev only; ASVspoof 2021 eval keys are public.
- **Research-only/NC licenses** (FakeAVCeleb, AV-Deepfake1M): fine for private training/eval; must NOT ship commercially. VoxCeleb2 mirror licenses unverified (CC-BY-SA claimed). Document all.
- **HAVIC saturation (~1.0 everywhere)** and **AASIST clean-side was a sine** are known data-quality gaps: Task 3 (CommonVoice real speech) + VoxCeleb2 real AV are the intended fixes; if separation is still poor, the trainer's calibration must broaden, not trust a tight fit.
- **Small-N risk**: ~240 audio / ~60 AV clips is thin for a production stack. Forward the plan to scale to the 2021 pool (Task 2) + full DFDC/FF++ subset (Task 7) once the small stacks validate.
- **Open question:** should the trained stack *replace* the pattern classifier entirely, or only the raw-weight aggregation (keeping patterns for explainability)? Recommendation: keep patterns for the `reasons`/reply, but let the LR stack own the raw→verdict mapping; confirm with the user before removal.
- **Full-AI-video class** is under-served by face-swap datasets; needs a small labeled set of pubbench full-AI generator clips (Sora/Veo/etc.) or self-generated fakes — Task 7 should name the source explicitly.

---

## First actions (this session)

1. Run Task 0 stability checkpoint.
2. Kick off Task 1 (ASVspoof 2019 parquet wget) — ~1.58 GB, resume-safe.
3. Kick off Task 2 (ASVspoof 2021 eval wget) in background.
4. Kick off Task 3 (Common Voice 'hi') in background.
5. File FakeAVCeleb + AV-Deepfake1M access requests (operator action; parallel).
6. Write + run `scripts/slice_asv19.py` and `scripts/extract_rows.py`, then Task 5 training.