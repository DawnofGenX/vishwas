# Deepfake Detection SOTA Survey — Vishwas (2026-08)

Note: primary web_search backend (Firecrawl keyless) returned 403; research done via arXiv API, HuggingFace Hub API, GitHub API. All HF facts below verified live via the Hub API on 2026-08.

## Slot A — Audio anti-spoofing (current: HABLA WavLM-AASIST, 3.79 GB ckpt)
- HABLA_WavLM_AASIST_NoAug_Seed42 still on HF (DeepFense), license field unset on card (paper/code Apache-2.0 lineage). ASVspoof2019 LA ~0.71% EER class; strong but heavy (WavLM-Large frontend, ~1.2 GB + graph backend) and fairseq-dependent.
- Field trend 2025–26: SSL-front-end ensembles dominate challenges (ImageCLEF 2026 ADD winner used 4-backbone SSL ensemble; SAFE 2025 systems use SSL frontends). Single-model gains are incremental (RAPTOR compact-SSL study, Spoof-SUPERB benchmark, noise-robustness surveys show all clean-lab SOTA degrades under real channel/noise — relevant to WhatsApp audio).
- No clearly better single open checkpoint that is also lighter. Verdict: KEEP as primary.

## Slot B — Video/face deepfake (current: EFFORT effort_chameleon.pth, NOT provisioned, CC BY-NC)
- EFFORT (CVPR 2025, "Efficient Orthogonal Modeling for Generalizable Face Forgery Detection") remains a standard generalizable baseline; CLIP/Chameleon-backbone cross-domain ACERs are competitive but 2025-26 literature moved to CLIP-based low-rank/subspace intervention (arXiv 2601.11915), VLM-reconstruction methods (MFVLR), MoE fine-grained alignment — mostly code-only, no public checkpoints yet.
- No dominant new open-weights video detector with a downloadable checkpoint was found on HF (searches for EFFORT/chameleon checkpoints returned nothing).
- Verdict: PROVISION EFFORT as planned (it's still near-SOTA and the checkpoint exists); flag CC BY-NC. Revisit once any of the 2026 CLIP-subspace/VLM methods release weights.

## Slot C — Audio-visual crossmodal (current: HAVIC best_ft_model.pth, MIT, 859 MB)
- HAVIC (JielunPeng/HAVIC) confirmed on HF, MIT license, updated 2026-07.
- Newer work: AV-Deepfake1M++ benchmark (2025) with real-world perturbations; KLASSify (SSL audio + handcrafted visual, ACM MM 2025); AuViRe (temporal localization, ckpt public: ckoutlis/auvire-avdeepfake1m, Apache-2.0, only 36 MB safetensors — localization not whole-video classification); modality-decoupled general AIGC detectors (2026, no ckpts found).
- Verdict: KEEP HAVIC. Optionally ADD AuViRe as a cheap localization side-channel (Apache-2.0, tiny) if segment-level evidence is ever useful.

## Slot D — Waveform/Mamba second opinion (current: RawBMamba, no license)
- RawBMamba: paper arXiv 2406.06086. Community mirror now on HF (SpeechAntiSpoofingBenchmarks/RawBMamba): rawbmamba_best.pt 2.97 MB + ONNX 12.7 MB, NO license, gated:false. Verified third-party eval numbers from that repo's harness: ASVspoof2019 LA EER 1.18%, but ASVspoof5 test EER 37.9% — very weak on newer generators. Original GitHub cyjie429/RawBMamba has no license either.
- XLSR-Mamba (arXiv 2411.10027, Xiao & Das): official checkpoints on HF AustinXiao/XLSR-Mamba-LA and -DF, **MIT license**, safetensors 1.28 GB each (includes wav2vec2-XLSR-300M frontend), PytorchModelHubMixin-loadable. Reported EERs: 21LA 0.93 / 21DF 1.88 / In-the-Wild 6.71%. Repo swagshaw/XLSR-Mamba MIT, needs pinned fairseq commit a54021305d6b3c.
- Fake-Mamba (2508.09294): XLSR frontend + bi-Mamba backend, real-time claims; repos exist but no official checkpoint found on HF.
- BiCrossMamba-ST (2505.13930): no public ckpt.
- Verdict: REPLACE RawBMamba with XLSR-Mamba-LA (MIT, public safetensors, better in-the-wild numbers) as the Mamba-family second opinion; or keep RawBMamba ONNX as evaluation-grade fallback. XLSR-Mamba shares the fairseq/WavLM-style stack with HABLA so integration cost is low.

## Bonus gates
- Phishing/URL: no standout new open-weights classifier worth displacing VT API + heuristics; keep as-is.
- Text-spam/scam: modern approach would be an LLM/NLI prompt gate rather than a dedicated model — out of scope here.

## Bottom line
| Slot | Action |
|---|---|
| Audio primary | Keep HABLA WvLM-AASIST |
| Video | Provision EFFORT (CC BY-NC flagged) |
| AV crossmodal | Keep HAVIC; optional AuViRe add-on |
| Mamba slot | Replace RawBMamba → XLSR-Mamba-LA (MIT) |

Caveats: web search engine unavailable; EFFORT official repo/checkpoint location not re-verified this session (GitHub search rate limits). RawBMamba ASVspoof5 37.9% EER figure comes from the community benchmark mirror's own harness, not the original paper.
