# Video Fusion Seed (2026-08-25)

**Status: mechanism PROVEN, corpus pending.** The video feature pipeline runs
the PRODUCTION capabilities (DeepfakeVideoCapability + CrossModalCapability)
over clips and emits feature vectors byte-identical to
`FusionEngine.feature_vector("deepfake_video")` (verified exact match → a
trained stack maps 1:1 with zero width-drift).

## Seed result (4 AI + 2 synthetic-real controls)
| | effort | frameheur | av_risk | havic |
|---|---|---|---|---|
| AI original | 0.677 | 0.281 | 0.500 | 1.000 |
| AI crf45 | 0.593 | 0.272 | 0.500 | 1.000 |
| AI resize50 | 0.624 | 0.279 | 0.500 | 1.000 |
| AI fps12 | 0.672 | 0.280 | 0.500 | 1.000 |
| Control testsrc | 0.388 | 0.225 | 0.000 | gap |
| Control smptebars | 0.549 | 0.209 | 0.100 | 0.993 |

**effort>0.60 → 100% precision on AI (3/3, no false flags).** av_risk is a
strong hinge: AI pinned 0.50 (anti-correlated AV) vs controls 0.00/0.10.

## Key properties
- EFFORT and AV-sync are NOT saturated (contrast AASIST) — the AI "fully
  generated + AV-mismatch" signature survives heavy compression/resize/fps-drop.
- Controls are synthetic patterns, NOT true talking-head footage → the seed is
  the *minimum* proof; real FaceForensicsC23 pairs (17.9 GB, zero-cred, downloading)
  will supply genuine real/fake.

## Next
- When FFC23 + full-AI-video class (Sora/Veo/Kling demos) land, re-run the
  extractor, train `stack_deepfake_video.json`, cross-validate OOF vs the
  calibrated-heuristic baseline, then wire it in via `FusionEngine.load_trained()`.
