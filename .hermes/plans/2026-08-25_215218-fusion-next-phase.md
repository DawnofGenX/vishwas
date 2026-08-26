# Fusion Layer — Next-Phase Plan (post dataset-training campaign)

> **For Hermes:** Use subagent-driven-development to implement task-by-task. Each task is independently verifiable; commit after each.

**Goal:** Convert the dataset-training campaign's honest findings into a production-quality fusion layer: fix the heuristic over-flagging bug (shipping), land the principled AASIST saturation fix (in flight), rebuild the video training set with audio-bearing AI/reals, and re-evaluate a learned stack only when separating evidence exists.

**Architecture:** Keep Fusion-v2's pattern classifier as the serving mechanism (validated). Fix detector-level correctness first (heuristic thresholds, AASIST frontend seam, EFFORT C23 inversion), then expand data where signals exist (full-AI-with-audio class), and only then revisit a learned GBDT head with a fair bar.

**Tech Stack:** existing vendored detectors (`model_archs/`), `fusion_train.py`, sklearn GBDT (available in docling-python tree), ffmpeg corpus tooling, the proven feature harnesses under `/home/hermes/fusion_av/scripts/` + `/home/hermes/fusion_audio/scripts/`.

---

## Current context / assumptions

- HEAD `8859d34`, tree clean except untracked `fusion/training/stack_deepfake_audio.json` (provenance artifact, keep untracked or move to /opt).
- **Dataset campaign conclusions (all OOF, honest):**
  - Audio: production path AUC 0.4875 (saturated); AASIST backend preserves signal until attention/GAT tail; scaling frontend ×20–130 restores AUC 0.94→1.0 → hypothesis: HABLA trained on UN-normalized frontend output; our extract_features applies final LN. AASIST-fixer agent in flight verifying against upstream DeepFense code.
  - Video: at n=84, GBDT 0.499 ≈ LR 0.492 ≈ heuristic 0.442 — all chance. **EFFORT is inverted at C23** (single-feature AUC 0.385); frameheur flat (0.497); av_risk/havic structurally absent on FF++ (no audio ⇒ known gaps). The one AI-video row is caught by all heads but via a 1-sample gap fingerprint.
  - **SHIPPING BUG (§8 of gbdt_report):** production heuristic returns `do_not_use` for ALL 84 clips including 40 real talking-heads (effort ~0.62–0.67 baseline × weight 2.5 saturates logistic-6 squash; clean-bonus never fires because effort never ≤0.10).
- Suite baseline: **363 passed / 7 skipped**. Webhook live on Fusion-v2 patterns.
- Datasets on disk: ASVspoof 2019 slice (240 clips), ASVspoof 2021 eval tar (7.76 GB, unextracted), FaceForensicsC23 zip (17.88 GB, subsets extracted).

---

## Task queue (priority order)

### Task 1 — P0 SHIPPING FIX: heuristic over-flags all talking-heads
**Objective:** stop the engine from returning DO_NOT_USE for every FF++-style real video.
**Evidence:** gbdt_report.md §8 — raw weighted score ≈0.63–0.67 regardless of class; effort weight 2.5 dominates; clean-bonus requires effort ≤0.10 which real footage never hits.
**Approach (pick empirically, verify on rows_video_v2.jsonl):**
1. Recalibrate effort's affine seed using the 40 FF++ reals as the real-class reference (current seed maps control 0.302→~0.36; real talking-heads sit 0.62–0.87 → they map INTO the risk band). Target: median real ≈0.35–0.45 post-calibration so weighted score lands CAUTION/TRUST boundary, not DNU.
2. AND/OR raise the clean-bonus epsilon from 0.10 to a calibrated value (e.g. effort ≤0.45 when frameheur also low) — but ONLY if it keeps the operator AI-video anchor DO_NOT_USE.
**Files:** `src/vishwas/fusion.py` (CALIBRATION table effort entry; clean-bonus condition), tests.
**Tests:** new `tests/test_36_real_video_not_dnu.py` — FF++-real feature vectors must NOT produce do_not_use (expect caution/unable band); test_35 operator-AI anchor MUST stay DO_NOT_USE conf ≥0.45 pattern fully_generated; full suite green.
**Verify:** run decide() over all 84 rows of rows_video_v2.jsonl via checks_from_vector reconstruction; report verdict distribution before/after. Accept: ≥80% of FF++ reals NOT do_not_use; AI row still DO_NOT_USE.

### Task 2 — Land AASIST fix (blocked on agent sa-0-dd6e18fa report)
**Objective:** merge the principled frontend-seam fix once the agent proves ≥0.85 AUC w/ real score spread on 240 gold clips.
**Steps:** review diff against upstream evidence → run its proof script myself (240 clips, confusion@0.5, quantiles) → run full suite → commit with upstream file:line citations → restart webhook → update skill + FUSION_DATASETS doc.
**Accept:** AUC ≥0.85 non-degenerate; suite green; silence/sine/noise still distinct (test_34); operator audio path unchanged behaviorally except better separation.

### Task 3 — Re-extract ASVspoof 2021 eval pool + scale audio rows
**Objective:** grow training rows from 240 → thousands once Task 2 lands (only meaningful with a fixed AASIST).
**Steps:** extract 7.76 GB tar → parse 2021 keys+meta TSVs (asvspoof.org) → sample balanced 1000/1000 → reuse `/home/hermes/fusion_audio/scripts/{slice_asv19→adapt}_21.py` + extract_rows.py → train LR stack → compare OOF vs heuristic baseline (the Task-5 bar from the old plan).
**Accept:** if stack OOF AUC beats heuristic by ≥0.05 AND confusion non-degenerate → wire via load_trained(); else document and keep heuristic.

### Task 4 — Build the full-AI-video-with-audio corpus (the class that matters)
**Objective:** fix the n=1 AI-row problem; give av_risk/havic within-class variance to learn from.
**Steps:** curate ≥30 gen-3 AI videos WITH speech/audio (official Sora/Veo/Kling/Runway demo reels, yt-dlp where ToS-permitted) + ≥30 matched-domain real talking-heads with audio (CC interviews/Pexels) → same harness → extend protocol → three-head evaluation again.
**License note:** record per-clip source URL + license in a manifest CSV; demo-reel use = research/internal only.
**Accept:** with av_risk/havic live on ≥30 fakes + ≥30 reals, re-run GBDT vs heuristic. Wire only if bar met (OOF AUC > heuristic by ≥0.07, FP rate on reals ≤20% @ recall ≥80%).

### Task 5 — Investigate EFFORT inversion at C23
**Objective:** understand why EFFORT ranks FF++ fakes LOWER than reals (AUC 0.385) — calibration shift or preprocessing mismatch?
**Steps:** probe EFFORT on its own published benchmark subset if obtainable; check our frame extraction (8 frames, resize?) vs EFFORT's eval protocol (25fps sampling, face-crop pipeline?); test face-cropped vs full-frame inference; document findings in docs/research.
**Outcome:** either a preprocessing fix (restoring separation) or a documented domain-limitation with EFFORT deweighted for face-swap targets.

### Task 6 — Demamba decision
**Objective:** resolve demamba being a constant gap column (84/84).
**Steps:** check whether weights are actually provisioned anywhere (/opt listing says rawbmamba exists but demamba_general was 'unavailable'); either provision + wire (adds a 3rd video signal) or formally remove from WEIGHTS/_EXPECTED_PROB_DET with a loud startup log.
**Accept:** fusion no longer carries a dead signal; coverage counts updated; tests updated.

### Task 7 — Final docs + memory
**Steps:** consolidate gbdt_report.md + AASIST fix + this plan's outcomes into docs/research/FUSION_FINAL_2026-08.md; update verisafe-operations skill v1.2 (four-gate posture + new calibration + any wired stacks); memory update (short).

---

## Explicitly deferred / rejected
- **Wiring GBDT now**: rejected by evidence (both bars failed; would add sklearn dep to serving path for zero gain).
- **More FF++-style face-swap data**: EFFORT can't rank it; more rows won't help until Task 5 explains the inversion.
- **Trained-stack-for-url_phishing changes**: current stack_url_phishing.json untouched — out of scope this phase.

## Risks / tradeoffs
- Task 1 threshold changes could swing real-world verdicts broadly — mitigate by anchoring on the 84-row measured corpus + operator AI-video regression test.
- Task 2 depends on agent output quality; orchestrator must independently rerun the 240-clip proof before merging (agents' self-reports are not proof).
- Task 4 curation is manual/slow; cap at ~60 clips to stay within session budget.
- EFFORT C23 inversion may be unfixable without retraining — accept and document rather than force.

## Open questions
1. Should Task 1 prefer recalibrating effort's seed vs raising the clean-bonus epsilon — or both? (Decide empirically against the 84-row corpus.)
2. For Task 4, does the operator have preferred AI-video sources (their own past generations) to include as canonical fakes?
3. Keep `stack_deepfake_audio.json` artifact in-tree (untracked) or move provenance artifacts to /opt?
