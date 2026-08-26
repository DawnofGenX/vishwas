# Fix Detector Overfitting + Stand Up Image Detection & Multimodal Repair — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. Three parallel workstreams with disjoint file ownership; the orchestrator merges shared files (fusion.py, secrets env) between waves.

**Goal:** Eliminate the false-RISK-HIGH verdicts on real videos and images (root cause: EFFORT-chameleon checkpoint scores every face 0.70–0.86 and dominates fusion at weight 2.5), stand up a proper AI-image detector (SPAI, CVPR 2025), and repair the cross-modal system (HAVIC preprocessing bug), with every change calibrated against a FRESH live-derived corpus and held-out validated before deploy.

**Architecture:** Three workstreams. **W1 (video overfit):** swap the EFFORT gate to the FF++-trained checkpoint, re-extract a fresh live feature corpus (anchored on the actual false-HIGH failures), retune `deepfake_video` weights on it. **W2 (images):** vendor SPAI (spectral AI-image detector, weights from Google Drive via the proven Drive-API bypass), wire it as the image_facecheck heavy gate, tune image weights on a real-photo vs AI-image mini-corpus. **W3 (cross-modal):** repair HAVIC preprocessing (official weights already on disk; weight-hunt evidence: saturation is a face-crop/fps preprocessing bug, their pipeline = 25fps face-crops + 16kHz audio crops), re-verify separation on real vs AI clips.

**Tech Stack:** Python 3.12, torch (docling-python tree), ffmpeg, pytest; repos: `YZY-stack/Effort-AIGI-Detection` (ffpp ckpt, Drive), `mever-team/spai` (GitHub, weights on Drive), `JielunPeng/HAVIC` preprocessing reference (`tuffy-studio/HAVIC` GitHub, MIT).

---

## Current context / assumptions

- HEAD `ec6eb19`, tree CLEAN (partial fusion.py weight change reverted 2026-08-26 — it was tuned against the stale corpus and failed live on clips 904/540). Suite 370/7. Webhook live :2790 (12 deps, cpu).
- **The bug (live-verified):** real FF++ originals `904.mp4` (effort 0.861), `570.mp4` (0.699), `245.mp4` (0.691), `540.mp4` (0.812) → DNU/HIGH. EFFORT-chameleon scores EVERY face 0.70–0.86. `deepfake_video` weights: effort 2.5 (dominant), frameheur 1.0, av_risk 2.0, havic 1.2.
- **Image path has the same bug:** `image_facecheck` weights `faceforensics.prob: 2.5` dominant; the gate resolves to the SAME EFFORT-chameleon checkpoint via `VISHWAS_EFFORT_WEIGHTS` fallback (`_resolved_weights_env()` in image_facecheck.py). Fallback heuristic = spectral band anomaly (works, weak).
- **HAVIC:** weights official + on disk, but `prob_inconsistent` ≈ 1.0 on everything → saturation. Weight-hunt §4: their pipeline feeds AudioMAE+MARLIN specific 25fps face-crops + 16kHz audio crops via `video_data_engine/preprocess_pt_dataset.py`; ours likely feeds wrong crops. MIT license.
- **The stale-corpus trap (why the last fix failed):** `rows_video_v2.jsonl` stores effort ≈0.6 for reals; live scoring gives 0.7–0.86. All weight tuning MUST use fresh features.
- **Corpus sources on disk:** 41 real FF++ originals + 43 fakes in `/tmp/fusvid_*/original/` (83 clips, numbered NN.mp4 = real, named = fake), seeded AI clips `~/fusion_av/seeded/*.mp4`, govdoc fixtures. Real photos for W2: reuse test fixtures + operator's own photos (need ≥20 real + ≥20 AI images; AI images can be generated locally — flux1-schnell is in ~/.cache/huggingface — or drawn from GenImage/DMimageDetection repos).
- **User decisions (locked):** checkpoint swap + live recalibration; SPAI for images; fix existing cross-modal system; parallel execution; fresh live corpus with held-out validation; delete the uncommitted partial change (DONE).
- Network is UP (github/hf 200). Drive downloads work via rclone remote "gdrive" (scope=drive) or Drive API files.copy bypass (proven for effort_chameleon).

## Proposed approach

Parallel subagents (max 2 concurrent per API cap), disjoint file ownership:
- **W1 owns:** `deploy/vishwas-secrets.env` (effort path flip), `/opt/verisafe/models/effort/ffpp/`, `~/fusion_av/feat_vectors/rows_video_v3.jsonl` (new), fusion.py deepfake_video block (LAST, after W3 lands its numbers — orchestrator merges).
- **W2 owns:** `src/vishwas/model_archs/spai.py` + `_spai/` vendor, `src/vishwas/capabilities/image_facecheck.py` (gate wiring), `VISHWAS_IMAGE_FACE_WEIGHTS`, image corpus dir `~/fusion_img/`.
- **W3 owns:** `src/vishwas/model_archs/_havic/` + `havic.py` (preprocessing repair only), cross-modal eval artifacts `~/fusion_av/crossmodal/`.
- Orchestrator owns: fusion.py final weight merge, tests that span workstreams, webhook restart, commits per wave.

Every workstream ends with a measured gate (numbers, not vibes) before its code reaches fusion weights.

---

## Workstream 1 — Video overfit fix (ffpp swap + live corpus)

### Task W1.1: Fetch effort_ffpp.pth
**Files:** Create `/opt/verisafe/models/effort/ffpp/effort_ffpp.pth` + `PROVENANCE.md`
1. Try direct: the 3 checkpoints live on Google Drive per manifest D5 notes; find the ffpp file ID from `docs/research/` (grep `effort_ffpp` for the Drive link; the YZY-stack README lists them). If the link is dead, use rclone `gdrive:` remote to search/copy, else Drive API files.copy bypass (proven pattern for chameleon).
2. Verify: expected size 1,213,769,519 B, sha256 first12 `8d86711f098d` (from MODEL_WEIGHTS_MANIFEST.json).
3. PROVENANCE.md: source URL, sha256, license CC BY-NC 4.0, date.
**Pass:** byte-exact match or documented deviation.

### Task W1.2: Probe ffpp on faces (before any wiring)
1. Script `/tmp/w1_probe.py`: load ffpp via the existing effort adapter (`resolve("VISHWAS_EFFORT_WEIGHTS")` with env pointed at the new file), score 8 frames from each of: 4 real FF++ clips (904, 570, 245, 540), 4 fake (Deepfakes/FaceSwap/Face2Face/NeuralTextures one each), seeded AI clips.
2. **HARD BAR:** mean(real) ≤ 0.45 AND mean(fake) ≥ 0.55 OR AUC(real∪fake) ≥ 0.75 on those 12 clips. If the ffpp ckpt ALSO fails → STOP, report; fallback = W1 pivots to DeepfakeBench UCF (`ucf_best.pth`, direct URL verified in WEIGHT_HUNT §5).
3. Record per-clip posteriors to `/tmp/w1_probe.json`.
**Pass:** bar met, numbers recorded.

### Task W1.3: Flip the gate + smoke
1. `deploy/vishwas-secrets.env`: `VISHWAS_EFFORT_WEIGHTS=/opt/verisafe/models/effort/ffpp/effort_ffpp.pth` (old line commented `# rollback: chameleon — overfits real faces`).
2. Live CLI on the 4 failing reals: `bash scripts/run_vishwas.sh cli --file ...` → all 4 must be NOT do_not_use (CAUTION or lower) BEFORE any weight change. If verdicts still HIGH, the weight retune in W1.5 handles it; record both states.
**Pass:** effort check shows new posteriors; runs green.

### Task W1.4: Fresh live corpus extraction (rows_video_v3)
1. Extend `/home/hermes/fusion_av/scripts/extract_video_rows.py` (READ its args first; add `--out rows_video_v3.jsonl --ffpp` mode or env-driven) and score ALL 83 FF++ clips + 6 seeded AI clips through the live pipeline (ffpp effort, frameheur, av, havic).
2. Split: 80% tune / 20% held-out (stratified by label AND subclass; seed 42).
3. **Pass:** rows_video_v3.jsonl exists, 89 rows, features match current live output byte-for-byte on 3 spot-checked clips.

### Task W1.5: Retune deepfake_video weights on v3 tune-split
1. Grid search (offline, FusionEngine replay like /tmp/vid_calib_final.py pattern but reading v3): effort {0.8,1.0,1.2,1.5}, av {2.0,2.5,3.0}, havic {1.2,1.5,2.0} + the effort affine cal (see `_calibrate` table).
2. Select by: reals NOT-DNU ≥ 90% (tune split) AND fakes ≥CAUTION 100% AND AI anchors DNU 100% (all 6 seeded AI clips).
3. **Validate on held-out 20%:** same bars. Fail → widen grid, re-select; document every candidate's numbers.
**Pass:** chosen weights beat bars on BOTH splits; table of top-5 candidates recorded in `/tmp/w1_tuning.md`.

### Task W1.6: Land + suite + live gate
1. Apply weights to fusion.py deepfake_video block (with the same style of provenance comment as existing).
2. `env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS PYTHONPATH=src python3 -m pytest tests/ -q` → 370+ passed (test_30/test_35 fusion scenario tests may need updating to the new expected operating point — update EXPECTED constants, don't delete assertions).
3. Live: the 4 failing reals → NOT-DNU; `ai_crf45.mp4` → DO_NOT_USE; 2 fake FF++ → ≥CAUTION.
4. `systemctl --user restart vishwas-webhook`, health ok, one WA smoke (operator).
5. Commit: `fix(fusion): effort ffpp swap + live-corpus recalibration — real videos no longer false-HIGH`.

## Workstream 2 — SPAI image detector

### Task W2.1: Vendor SPAI code + fetch weights
**Files:** Create `src/vishwas/model_archs/_spai/` (code vendored from github mever-team/spai, copy the model def + spectral preprocess; license MIT per repo), weights to `/opt/verisafe/models/spai/`.
1. Weights: Google Drive file `1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI` (SPAI trained ckpt) + MFM ViT-B/16 backbone `mfm_pretrain_vit_base.pth` (`1qgMuODAxAapwXZXbH2Bgo5s5O_zc1bdl` from Jiahao000/MFM README). Use rclone gdrive or Drive API bypass.
2. Note SPAI preprocessing: any-resolution spectral learning — READ their inference block only (cap reading: model def + infer script), replicate exactly.
**Pass:** files on disk, sizes recorded, PROVENANCE.md written.

### Task W2.2: Structure probe + sensitivity
1. Load ckpt onto arch; key coverage ≥95%; forward a real photo + a locally-generated flux image → distinct finite posteriors (spread > 0.3).
**Pass:** probe numbers recorded in `/tmp/w2_probe.md`.

### Task W2.3: Wire into image_facecheck
**Files:** Modify `src/vishwas/capabilities/image_facecheck.py` (the heavy-gate `_resolved_weights_env` section: prefer `VISHWAS_SPai...` no — use `VISHWAS_IMAGE_FACE_WEIGHTS` pointing at SPAI; keep spectral fallback), Create `src/vishwas/model_archs/spai.py` (ArchSpec family `spai`), Modify `src/vishwas/model_adapters.py` (register family).
1. ArchSpec contract: copy aasist3.py pattern (build loads ckpt + applies state, score returns p_fake with polarity documented in header).
2. Router unchanged (image_facecheck already routes). Fusion `_SIGNAL_SOURCES` `faceforensics.prob` maps to the check name `image_face_forensics` — W2 keeps that name, swaps the MODEL behind it.
**Pass:** CLI on a real photo → image_facecheck runs with SPAI gate ok.

### Task W2.4: Image corpus (real + AI) + weight tune
1. Build `~/fusion_img/` with BOTH classes, each split 80/20 (seed 42):
   - **REAL photos** (≥20; the false-HIGH class — must NOT flag HIGH): operator's own photos + GOV/AUDIO test fixtures that are real photographs. Add several unmodified phone/DSLR photos the operator provides (in-distribution real-world).
   - **AI-generated images** (≥20; must be detected): labeled samples pulled from pinned sources — `grip-unina/DMimageDetection` (241★, canonical diffusion-image set; eval data per its `docs/data.md`) and GenImage HF mirrors (`e8035669/GenImage`, verified fetchable). Record per-image provenance (repo/id) in `~/fusion_img/ai/sources.csv`.
   - protocol.csv same format (`path,label,subclass`; label 0=real photo, 1=AI).
2. Score all through the pipeline → `rows_image_v1.jsonl`; tune `image_facecheck` weights (`freqband.prob`, `faceforensics.prob`) on the tune split.
3. **Dual validation bars (BOTH real and AI, held-out split):**
   - **REAL photos → NOT-DNU ≥90%** (no false-HIGH — the overfit regression guard; ≥1 unmodified real photo must land TRUST/CAUTION with the SPAI gate ok, not gated off).
   - **AI images → ≥CAUTION 100%, DO_NOT_USE ≥50%**.
   - Failure on real-photo bar = overfit guard tripped → widen threshold / adjust calibration, re-select; document every candidate's numbers.
4. Apply weights to fusion.py image_facecheck block; suite green (test_14 adapter tests extended with a spai-family hermetic case).
5. Commit: `feat(image): SPAI (CVPR'25 spectral) AI-image gate + calibrated image fusion weights`.

## Workstream 3 — Cross-modal (HAVIC) repair

### Task W3.1: Reproduce their preprocessing
1. Fetch reference: `git clone --depth 1 https://github.com/tuffy-studio/HAVIC /tmp/havic_ref` (MIT); READ ONLY `video_data_engine/preprocess_pt_dataset.py` + the AudioMAE/MARLIN input specs.
2. Document the exact expected input: face-crop geometry (detector? margins?), fps (25), audio crop (16kHz, alignment), frame count, normalization.
3. Diff against our `_havic` vendor's current input path (READ `src/vishwas/model_archs/havic.py` score() + `_havic/` preprocess).
**Pass:** written diff `/tmp/w3_preprocess_diff.md` naming every mismatch.

### Task W3.2: Fix our preprocessing
**Files:** Modify `src/vishwas/model_archs/_havic/` (preprocess only) and/or `havic.py` score path.
1. Apply fixes for each named mismatch (likely: face-crop via cv2 face detector at 25fps equivalent sampling, audio aligned to the SAME window, correct normalization).
2. Sensitivity: real speech-video vs AI video (seeded clips + operator's) → havic posterior must differ (real < 0.5, AI > 0.7 target; saturation broken = any non-constant separation).
**Pass:** posteriors recorded; no longer ~1.0 constant.

### Task W3.3: Cross-modal separation eval
1. Score 20 real (FF++ originals WITH audio? — most FF++ clips have no audio; use seeded real controls + operator clips that carry audio) + 6 seeded AI clips: report havic + av_risk distributions.
2. If HAVIC separates: keep weight 1.5 (or retune in W1.5's grid — W3 numbers feed W1.5's havic candidates).
3. If HAVIC still saturates: document honestly, leave weight modest, av_risk carries (existing behavior).
**Pass:** eval table in `/tmp/w3_eval.md`; fusion impact decided by evidence.

## Workstream 4 — Orchestrator closeout
1. Merge fusion.py weight changes from W1/W2/W3 into one coherent WEIGHTS table; verify no contradictions (single final replay over v3 corpus + image corpus).
2. Full suite + `test_30`/`test_35` scenario expectations updated to the new operating point.
3. Webhook restart + health + live smoke: 1 real video → not-HIGH; 1 AI video → HIGH; 1 real photo → not-HIGH; 1 AI image → ≥CAUTION.
4. Update `docs/research/FUSION_FINAL_2026-08.md` (new operating point, SPAI, HAVIC outcome), MODEL_WEIGHTS_MANIFEST.json (effort ffpp fetched-real, spai gate added, havic preprocessing note), skill verisafe-operations v1.4.0.
5. Commit + final report to user with all measured numbers.

---

## Files likely to change

- `deploy/vishwas-secrets.env` — effort path flip, VISHWAS_IMAGE_FACE_WEIGHTS→SPAI
- `src/vishwas/fusion.py` — deepfake_video + image_facecheck weight blocks (+_EXPECTED_PROB_DET if signal set changes)
- `src/vishwas/model_archs/spai.py` + `_spai/` (new), `havic.py`/`_havic/` (preprocess), `model_adapters.py` (spai family)
- `src/vishwas/capabilities/image_facecheck.py` (gate wiring)
- `tests/test_14_model_adapters.py`, `tests/test_30_fusion_trust.py`, `tests/test_35_fusion_v2_scenarios.py` (expectations)
- `/opt/verisafe/models/{effort/ffpp,spai}/` + PROVENANCE.md files
- `~/fusion_av/feat_vectors/rows_video_v3.jsonl`, `~/fusion_img/` (data, outside repo)
- `docs/research/FUSION_FINAL_2026-08.md`, `docs/research/MODEL_WEIGHTS_MANIFEST.json`

## Tests / validation

- Hermetic suite green after every wave (370+ baseline; new tests added).
- W1 probe bar (real ≤0.45 / fake ≥0.55 or AUC ≥0.75); W1.5 dual-split bars (reals NOT-DNU ≥90%, fakes ≥CAUTION 100%, AI DNU 100%).
- W2 sensitivity spread >0.3; image corpus bars (real NOT-DNU ≥90%, AI ≥CAUTION 100%).
- W3 saturation broken (non-constant, separated posteriors).
- Final live CLI gates on the 4 originally-failing clips + AI anchors + webhook health.

## Risks / tradeoffs / open questions

- **ffpp checkpoint may also misbehave** → fallback to DeepfakeBench UCF (verified URLs, CC BY-NC) is pre-planned (W1.2 branch).
- **SPAI Drive weights may be gated/vanished** → fallback: chameleon/genimage EFFORT ckpt for AI-images (correct domain for chameleon!) as image gate — chameleon is AIGC-image-trained, which is the RIGHT domain for image AI-detection (the overfit was applying it to faces/video).
- **License note:** ffpp + SPAI-adjacent MFM weights carry CC BY-NC / research terms — fine for current deployment posture (local, non-commercial), flag for any commercial pivot.
- **FF++ clips lack audio** → cross-modal eval uses only audio-bearing clips (seeded + operator's); av_risk remains gap-flagged honestly on silent clips.
- **Parallel waves touch fusion.py** → merge discipline: workstreams record candidate weights, ONLY the orchestrator's final task writes fusion.py.
- **Open Q:** if HAVIC repair succeeds, should it become the primary cross-modal signal over av_risk? Decide on W3.3 separation numbers.

## Execution summary

Parallel: W1+W2 first wave (disjoint: secrets/effort+corpus vs model_archs/spai+capability), W3 second wave (havic files untouched by W1/W2), orchestrator closeout last. Every fusion.py write happens once, at closeout, on fresh-corpus numbers.
