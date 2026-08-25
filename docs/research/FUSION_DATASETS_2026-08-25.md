# Fusion Training — Dataset + First-Results (2026-08-25)

**Status: STAGE 1 (audio) complete — honest NEGATIVE result.** The audio fusion
stack at its current detector configuration does NOT separate real/fake on
ASVspoof 2019 LA (OOF AUC ≈ 0.51, baseline ≈ 0.50). This is a genuine finding
about the BASE detectors, not a training-pipeline failure. Downstream video/AV
work (Task 7) is expected to be far more informative because EFFORT already
showed strong real-vs-AI separation (0.677 vs 0.302) and is not saturated.

## Datasets acquired (Stage 1)
| Dataset | Size | License | Status |
|---|---|---|---|
| ASVspoof 2019 LA (HF mirror `Bisher/ASVspoof_2019_LA`) | 1.57 GB parquet | ODC-BY | Downloaded**
| — sliced to 240 labeled clips (120 real / 120 fake) | ~50 MB WAV | ODC-BY | Extracted → `/home/hermes/fusion_audio/asv19_la/clips/` |
| ASVspoof 2021 LA eval (Zenodo 4837263) | 7.6 GB | ODC-BY | Background download in flight → `/home/hermes/fusion_audio/asv21_la_eval/` |

**Provenance:** all detector figure values in this doc are from LIVE runs against
those clips on the RTX 5090 box (2026-08-25). Clip protocol: `asv19_la/protocol.csv`.

## Pipe built (reusable)
- `/home/hermes/fusion_audio/scripts/slice_asv19.py` — read parquet → balanced labeled WAV set
- `/home/hermes/fusion_audio/scripts/extract_rows.py` — run AASIST/XLSR/offline over clips → JSONL in the
  exact `FusionEngine.feature_vector("deepfake_audio")` serving layout (10 floats: value+gap per weighted signal)
- Train: `PYTHONPATH=... python3 -m vishwas.fusion_train --dataset rows_audio.jsonl --target deepfake_audio`
  → writes `fusion/training/stack_deepfake_audio.json` (NOT wired; see below)

## Result
| Scorer | 5-fold OOF AUC | Note |
|---|---|---|
| Current heuristic `FusionEngine.decide("deepfake_audio")` | **0.502** | effectively random on this corpus |
| Trained LR stack (OOF, no leakage) | **0.512** | no real gain over baseline |

Single-split training reported 0.634 (favorable split); the honest OOF number is ~0.51.
**Decision: do NOT wire `stack_deepfake_audio.json` in as an override** — it would replace a
0.502 baseline with a 0.512 one, adding surface for no benefit.

## Root-cause (why audio can't separate yet)
1. **AASIST is saturated**: posterior ≈ 0.9996–0.9999 on BOTH real and fake ASVspoof clips.
   Input-sensitivity test passes (it discriminates silence/noise/sine) but it is NOT a
   usable binary discriminator on this corpus — it needs real-speech-scale discrimination
   that the current vendored config/weights aren't delivering here.
2. **XLSR separation is weak**: mean real 0.079 vs fake 0.103 (gap 0.025) after the zero-pad
   fix; direction is correct but variance is too high relative to the gap.
3. **No length confound** (real/fake mean lengths differ by 0.03 s), so this is genuine.

## What the XLSR fix gave
Repeat-padding short clips to the 66800 window (previous behaviour) artificially
periodized the signal and DESTROYED the posterior; zero-padding (upstream fairseq
`pad_to_multiple`) restores it. Committed `c2dfe14`. Single-clip probe: XLSR fake
posterior 0.591 (repeat) → 0.041 (zero-pad); real 0.583 → 0.368 — a much larger and
usable separation than the corpus mean suggests, i.e. XLSR *can* separate; the corpus
variance is what the OOF AUC reflects.

## Actionable path forward (unchanged dependency order)
1. **AASIST saturation is the priority blocker.** Before scaling data, investigate whether
   the vendored AASIST weights/config can be made to emit a real-vs-real-speech posterior
   (or whether the corpus's bona-fide subset is being scored against a mismatched
   spoof-class prior). A single calibration/augmentation fix here would unlock the audio
   channel; without it more ASVspoof data only trains a ~0.51 stack.
2. **Then scale**: reuse the same pipe on the 2021 eval pool + Common Voice 'hi' real speech.
3. **Video/AV (Task 7)** is independent and expected to be the higher-value channel (EFFORT
   not saturated): FaceForensicsC23 (zero-cred, face-foot) + full-AI-video class + a
   gated AV set for the HAVIC/AV-sync foot.

## License boundary (for the repo, repeat in docs)
ASVspoof 2019/2021 = ODC-BY (attribution; redistribution allowed). Common Voice = CC0.
FakeAVCeleb / AV-Deepfake1M = research-only / CC BY-NC 4.0 — training/eval-internal only,
NOT for a commercial WhatsApp deployment. FaceForensicsC23 = research, non-commercial.