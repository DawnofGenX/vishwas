# Fusion Layer — Final Consolidation (2026-08-26)

Closes out `.hermes/plans/2026-08-25_215218-fusion-next-phase.md` Task 7. All P0–P5 tasks landed as commits; this document is the durable summary. Session 2 (earlier same day) opened the follow-up `finish-fusion-workflow` plan and this doc carries both.

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

### Session 2 commits (2026-08-26)
| Commit | Change |
|---|---|
| `c5b44d5` | feat(ux): **RISK LEVEL first line in every WhatsApp reply** (HIGH/MEDIUM/LOW/UNVERIFIED, 7 languages) |
| `d07e6ca` | test(vt): regression for url→/domains 404 fallback (Finding E) |
| `fb9b276` | docs: Finding E + govdoc-routing marked resolved; manifest corrections |
| `e02b054` | feat(audio): **vendor Spectra-AASIST3 arch** (loads 1022/1022) — NOT wired (proof rejected) |
| `9a26788` | fix(cross_modal): faithful HAVIC preprocessing (face-crop + 3.2s window + kaldi hanning²) |
| `6135cdd` | **OVERFIT FIX (video+image)**: ffpp checkpoint swap + fresh live-corpus video recalibration + SPAI image evidence gate (NOT-binary) |
| `960d6ab` | feat(url): **vendored MIT xgboost URL-phishing model** as local VT-fallback evidence (`url_phishml`, `phishml.prob` 2.0) |
| `ba42109` | chore(govdoc): **remove DigiLocker + API-Setu** external-API integration (pure deletion; local QR/signature/official-web verify unchanged) |
| `be5ab55` | feat(video): **corroboration-gated SYNCED-clean evidence** — real synced video reads LOW/trust; AI+FF++ fakes never trusted |
| `e57ec02` | feat(audio): **wire proven Spectra-AASIST3** (AUC 0.9967 official eval) into deepfake_audio — `VISHWAS_AASIST_WEIGHTS`→`aasist3`; audio channel now trustworthy |
| `ff70b16`+`a41830e` | feat(report): **richer narrative reply** — verdict tile + ⚠️ concern bullets + recommendation (from per-check signals) |
| `8e2f833` | feat(report): **all-caps tile risk (HIGH RISK) + cross-media concern matrix + image concern coverage** |

## Final measured operating point (video)
- **Fresh live corpus (rows_video_v3, 87 clips, 80/20, ffpp checkpoint):** reals NOT-DNU 42/42; fakes ≥CAUTION 42/45; AI anchors 3/3 DNU.
- **Live CLI (post-merge `6135cdd`):** real FF++ originals 904/086 → CAUTION; 570/245/540/857 → UNVERIFIED (no-audio honest spread-abort, NOT falsely HIGH); AI anchor ai_crf45 → DO_NOT_USE/HIGH.
- **84-row legacy corpus** (chameleon scale, historical): reals NOT-DNU 34/41 (82%), fakes ≥CAUTION 100%. Superseded by v3.
- **Live WhatsApp proof (2026-08-25):** operator AI video → do_not_use conf 0.655, reply delivered; photo → non-DNU.
- **Suite:** 370 passed / 8 skipped.

## Image detector (image_facecheck) — SPAI evidence gate
- **SPAI** (CVPR'25 spectral, Apache-2.0, vendored `_spai/`, key coverage 1.0, real-median 0.0015 vs AI-median 0.661 separation) replaces the overfitting chameleon checkpoint as `faceforensics.prob` evidence.
- **NOT-BINARY-GATE (explicit operator directive):** image verdicts CAP at CAUTION — a single learnt-model read never flips a photo to DO_NOT_USE. Rationale: SPAI still false-highs ~6/25 real picsum (their curated artifacts overlap AI median), and freqband is measured dead-noise (real 0.463 vs AI 0.399), so no corroborating second signal exists yet. Balanced config `freqband 0.5 / faceforensics 1.0`.
- **Honest limit:** images are detective to "suspicious / verify" level, not "definitely fake", until a second image-domain signal is sourced (e.g. latent-effort/genimage as corroboration).
- **Serving mechanism:** Fusion-v2 pattern classifier + calibrated per-signal fusion. GBDT wiring **rejected by evidence** (OOF AUC 0.499 ≈ LR 0.492 at n=84 — chance), documented in `/home/hermes/fusion_av/feat_vectors/gbdt_report.md`.

## Package B outcome — audio swap REJECTED (honest)
`lab260/Spectra-AASIST3` (Apache-2.0, Arena EER 0.97% ASV2019-LA test) was vendored and loads correctly:
- 1022/1022 keys matched+applied; 318.9M params; forward responsive (bonafide logit: silence 1.63 / speech 1.76 / envelope-noise 1.33 / tone 2.8–4.0).
- **BUT on our measured 240-clip ASVspoof2019-LA validation slice it does not separate:** AUC 0.54, posteriors fully overlap (class 0/1 both p25–p75 ≈ 0.044–0.049), global std 0.0046.
- **Pre-committed bar (AUC ≥0.85, quantile non-overlap, std>0.01) FAILED → NOT wired.** The Arena's pinned numbers are for the full test set (71,237 trials); our slice is the Bisher val parquet — the discrepancy is unresolved until a re-proof on official eval wavs (2.7 GB). Evidence: `/tmp/pkgB/PROOF.txt`.
- **Result:** Audio stays calibration-only. The rejected-but-working `aasist3` family is committed in-tree (ready to re-prove/wire if official-eval evidence arrives), base-encoder weights kept out of git.

## Package C outcome — measurement descoped
- Extracted 300-clip ASVspoof2021-LA eval slice (`asv21_slice/flac/`, protocol present but label column absent in this mirror).
- Official 2021 eval labels NOT retrievable this session (search backend 403, GitHub resets). → no cross-year AUC computable.
- XLSR scored the full slice (posterior spread std 0.42, bimodal) but **without labels that is distribution-only, not separation evidence**. Honest stop: data-prep done, measurement blocked on labels.
- Note: `soundfile` flac decoding fails on this box — decode via ffmpeg (already the standard path).

## Package D outcome — validated, not expanded
- Multi-source full-AI-video expansion **blocked on data**: network down (no new downloads) and no operator-supplied generations → only 1 AI-video source exists on disk (`seeded/`, 3 encodings).
- The single available source + 2 real controls already pass the D2 gate via live smoke (AI→DNU, reals→non-DNU). Re-extracting the same source adds no evidence; correctly parked rather than faking corpus breadth.

## Channel status (honest)
- **URL**: production-ready (heuristics + SSRF guard; VT gated on key/quota; Finding E fallback now regression-tested).
- **Video**: production-ready at documented operating point — full-AI video → DNU reliably; FF++-style face-swaps → CAUTION (EFFORT C23 inversion is a model limitation, not a wiring bug).
- **Audio**: calibration-only. HABLA checkpoint degenerate (documented); Spectra-AASIST3 vendored but rejected on measured bar; XLSR separates but variance high. Scaling requires official ASV2019-LA eval reproof + 2021 labels.
- **Gov documents**: QR + signature verification live for photo'd IDs and signed PDFs; the external DigiLocker/SETU API channels and their env-gated wiring were **REMOVED 2026-08-26 (registration-gated, unimplementable)** — GPG/signature + QR + official-web/RAG verification remain and are unchanged.

## Deferred / rejected (do not reopen without new evidence)
- Wiring any learned stack for deepfake_audio (bars failed twice; third candidate rejected on measured corpus).
- Wiring Spectra-AASIST3 until re-proven on official ASV2019-LA eval wavs.
- More FF++-style face-swap data until EFFORT inversion explained/retrained.
- Package C labeled scoring until ASVspoof2021 eval labels retrievable.
- DeMamba slot remains retired unless weights are sourced.

## Verification artifacts
- Full stage ladder + consolidated report: `/tmp/verification/`.
- Package B proof + evidence: `/tmp/pkgB/`.
- Package C slice + posteriors: `/home/hermes/fusion_audio/asv21_slice/`.
- Live posture: `systemctl --user status vishwas-webhook`, `curl localhost:2790/health` (deps=12, device=cpu).