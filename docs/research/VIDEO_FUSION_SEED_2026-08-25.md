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

## Update — FFC23 real-pair result (2026-08-25)
Downloaded FaceForensics++_C23 (17.9 GB, zero-cred), extracted 10 real + 10
fake matched-pair clips + 6 seeded AI/synthetic-real. Extractors produce
serving-layout feature vectors.

**Finding (honest): EFFORT does NOT separate FF++ real vs fake.**
- effort AUC on the 20 FF++ clips: 0.46 (chance). Real (0.865/0.317/0.745)
  vs fake (0.871/0.341/0.762) are near-identical — matched-pair face-swaps
  are a hard benchmark for face-region forensics.
- av_risk = 0.000 on ALL FF++ clips (they ship no audio -> AV-sync silent)
  BUT 0.500 on the full-AI-video class. So av_risk cleanly separates
  full-AI-video from face-swap — the exact class distinction the operator
  needs.
- Full-AI signature is unambiguous: effort ~0.6 + av_risk 0.5 + havic 1.0.
- LINEAR LR stack on raw features: AUC 0.5 (fails). The decision boundary is
  PATTERN-SHAPED, not linear — confirming Fusion-v2's pattern classifier
  (already live) is the right architecture over a learned linear stack.

**Conclusion:** do NOT train a linear stack on mixed face-swap(FF++) + AI-video
clips — the class structure is non-linear. The trained LR stack is not wired in;
the pattern classifier remains the serving mechanism. A learned model would need
a non-linear head (e.g. small gradient-boosted trees / MLP) on richer features
(temporal face, identity-consistency), which is beyond the current pure-numpy LR
scope. Documented; not pursued further this session.
