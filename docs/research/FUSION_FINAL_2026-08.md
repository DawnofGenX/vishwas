# Fusion Layer — Final Consolidation (2026-08-26)

Closes out `.hermes/plans/2026-08-25_215218-fusion-next-phase.md` Task 7. All P0–P5 tasks landed as commits; this document is the durable summary.

## What shipped this phase
| Commit | Change |
|---|---|
| `6882a15` | fix(video): frame heuristics unit mismatch — [0,1] floats vs 0-255 thresholds |
| `98e83a6` | feat(govdoc): photo'd gov IDs reach QR verification via image capability |
| `03bb375` / `3d40116` | feat(ux): coverage-aware UNABLE replies + deepfake pattern explanations in WhatsApp replies |
| `86dd969` + `ca30446` | **Fusion v2** — calibrated per-signal fusion, pattern-aware deepfake agreement, reliability gate no longer aborts on coherent patterns |
| `7efa7fb` | fix(audio): prenorm frontend config — AASIST/XLSR were fully input-invariant before |
| `c2dfe14` | fix(xlsr): zero-pad short clips to trained window (restores XLSR separation) |
| `394aa08` + `f627e70` | **Real-video recalibration** — talking-heads no longer fuse to DO_NOT_USE; demamba slot retired |
| `35411b0` | docs: AASIST saturation post-mortem — checkpoint proven degenerate by 3 independent witnesses |

## Final measured operating point
- **84-row video corpus replay:** reals NOT-DNU 34/41 (82%, target ≥80%); fakes ≥CAUTION 100% (38 CAUTION / 5 DNU); the known full-AI row → DO_NOT_USE with fully_generated pattern. Tolerance ±1 row held.
- **Live WhatsApp proof:** operator AI video → do_not_use conf 0.655, reply delivered; photo → non-DNU reply delivered.
- **Suite:** 365 passed / 7 skipped.
- **Serving mechanism:** Fusion-v2 pattern classifier + calibrated per-signal fusion. GBDT wiring **rejected by evidence** (OOF AUC 0.499 ≈ LR 0.492 at n=84 — chance), documented in `/home/hermes/fusion_av/feat_vectors/gbdt_report.md`.

## Channel status (honest)
- **URL**: production-ready (heuristics + SSRF guard; VT gated on key/quota).
- **Video**: production-ready at the documented operating point — full-AI video → DNU reliably; FF++-style face-swaps → CAUTION (EFFORT C23 inversion is a model limitation, not a wiring bug).
- **Audio**: calibration-only. AASIST checkpoint degenerate (post-mortem in FUSION_DATASETS doc with checkpoint-swap steps); XLSR separates but variance high. Scaling path: ASVspoof2021 pool (verified intact, 181k clips) + Common Voice hi.
- **Gov documents**: QR verification live for photo'd IDs; DigitalLocker/SETU/GPG/RAG gated behind env vars as designed.

## Deferred / rejected (do not reopen without new evidence)
- Wiring any learned stack for deepfake_audio (bars failed twice).
- More FF++-style face-swap data until EFFORT inversion explained/retrained.
- DeMamba slot remains retired unless weights are sourced (`docs/research/MODEL_WEIGHTS_MANIFEST.json` has canonical sources).

## Verification artifacts
Full stage ladder + consolidated report: `/tmp/verification/` (REPORT.md + stage0..6b). Live posture: `systemctl --user status vishwas-webhook`, `curl localhost:2790/health`.
