# Weight Hunt — 2026-08-25

Verified downloadable model weights for Vishwas' missing/failing detector slots.
Every "verified" row below had its download URL probed live on 2026-08-25 (HTTP status +
file size via HEAD/x-linked-size or API blob listing). No fabricated links.

Legend: ✅ = download URL verified live this session · ⚠️ = link exists but gated/unverified
bytes · 🚫 = confirmed absent. Sizes are exact bytes where the host reported them.

---

## 1. AASIST speech checkpoint (CRITICAL)

Context: current `DeepFense/HABLA_WavLM_AASIST_NoAug_Seed42` loads but saturates
(~0.9997 spoof on bonafide AND spoof; AUC≈0.49). Official clovaai/aasist repo was checked
first: **MIT code, 0 releases, no weight files anywhere in the tree** 🚫 — the original
AASIST authors never published checkpoints. All viable options are third-party uploads.

| Name | Source | Direct download | Size | License | Arch | Quality evidence | Difficulty |
|---|---|---|---|---|---|---|---|
| **lab260/Spectra-AASIST3** ⭐ | https://huggingface.co/lab260/Spectra-AASIST3 | ✅ `/resolve/main/model.safetensors` (1,276,330,956 B) · ✅ `spectra-aasist3.onnx` (1,279,022,864 B) | ~1.19 GiB | **Apache-2.0** | wav2vec2-XLS-R-300m frontend + KAN-enhanced AASIST backend, ~319M params, self-contained (SSL encoder bundled) | **Independently re-scored EER 0.97 % on ASVspoof2019-LA (71,237 trials)**, 4.38 % on 2021-LA, 4.30 % on 2021-DF, 1.20 % In-The-Wild, 0.03 % on HABLA — pinned `scores.txt`+`result.yaml` in-repo (Speech Anti-Spoofing Arena) | LOW-MED |
| lab260/AASIST3 (deprecated predecessor) | https://huggingface.co/lab260/AASIST3 | ✅ `model.safetensors` (1,287,120,456 B) · ✅ `aasist3.onnx` (1,290,270,071 B) | ~1.2 GiB | CC BY-NC 4.0 | wav2vec2 + AASIST + KAN bridge | Arena EER 9.44 % LA (pinned scores); author-deprecated in favor of Spectra | MED |
| ash56/ssl-aasist | https://huggingface.co/ash56/ssl-aasist | ✅ `pytorch_model.bin` (1,271,613,458 B) | ~1.18 GiB | Apache-2.0 | SSL-AASIST (XLSR-300m frontend), trust_remote_code `AutoModel` | Official ckpt of arXiv 2502.05674, but **trained on WaveFake/HiFiGAN (LJSpeech+Aishell)** — no published LA-eval number; fairseq-pin install pain (`pip<24.1`, `omegaconf==2.0.6`, fairseq@920a548) | MED-HIGH |
| arnabdas8901/aasist-trained-asvspoof2024 | https://huggingface.co/arnabdas8901/aasist-trained-asvspoof2024 | ✅ `orig_aasist_epoch_1.pth` (1,276,136 B — classic-size LFCC-AASIST) · ✅ `ssl_aasist_epoch_7.pth` (1,271,618,222 B) | 1.3 MB / 1.18 GiB | MIT | classic AASIST (LFCC) + SSL-AASIST | Community upload (author is an ASVspoof researcher), **no eval evidence attached** — provenance risk, test before trusting | MED |
| DeepFense HABLA_{EAT,Hubert,Wav2Vec2}_AASIST_* | https://huggingface.co/DeepFense/ | exist (listed on HF) | — | unclear | same wrapper family as our saturated ckpt | Same lab/wrapper as the broken one — same saturation risk; low priority | — |

### TOP PICK: `lab260/Spectra-AASIST3`
Why: only candidate with *independently re-scored, pinned* near-SOTA LA evidence (0.97 %
EER ≫ our target <10 %), permissive Apache-2.0, self-contained safetensors (no fairseq,
no separate frontend ckpt), plus an ONNX twin for fast serving.

**Drop-in plan**
1. Download:
   `wget https://huggingface.co/lab260/Spectra-AASIST3/resolve/main/model.safetensors -O /opt/verisafe/models/aasist3/model.safetensors`
   plus vendor `model.py` (32 KB, the network def) from the same repo into
   `src/vishwas/model_archs/_spectra/`.
2. New arch module `src/vishwas/model_archs/aasist3.py`: build net from vendored
   `model.py`; state_dict from safetensors; expose `.score(wav)` like the existing spec.
3. Register: `_ARCH_FAMILIES["VISHWAS_AASIST_WEIGHTS"] = "aasist3"` (or a new
   `VISHWAS_SPEECH_CM_WEIGHTS` slot) in `src/vishwas/model_adapters.py`, and point
   `deploy/vishwas-secrets.env: VISHWAS_AASIST_WEIGHTS=/opt/verisafe/models/aasist3/model.safetensors`.
   Keep the old HABLA slot value commented out for rollback.
4. Preprocessing to replicate EXACTLY (arena wrapper, documented in the model card):
   resample to 16 kHz mono PCM → **preemphasis 0.97** (`y[n]=x[n]-0.97x[n-1]`) →
   **first 64,600 samples** (tile-pad if shorter) → forward → score = logit of class 1 =
   **bonafide** (higher = more bona fide — OPPOSITE of our current spoof-posterior
   convention; invert: `p_spoof = 1 - sigmoid(logit_bonafide)`).
5. Runtime: ~319 M params, bf16 ≈ 0.7 GB VRAM; a 10 s clip is one 4 s window + stride;
   ONNX path available if we want CPU viability.

---

## 2. DeMamba-General video weights (MEDIUM)

| Name | Source | Weights | License | Notes |
|---|---|---|---|---|
| chenhaoxing/DeMamba (OFFICIAL, arXiv 2405.19707) | https://github.com/chenhaoxing/DeMamba | 🚫 **No releases, no .pth/.ckpt anywhere in tree** (root listing verified: configs=`XCLIP_DeMamba.yaml` only). Code Apache-2.0, 195★, pushed 2025-09-24. Dataset (GenVideo/GenVideo-100K) IS released on ModelScope. | Apache-2.0 (code) | Training-only repo. Our dead weight slot is dead at the source — **correct `MODEL_WEIGHTS_MANIFEST.json` gate `demamba` from `real:true` to weights-never-published**. |
| HF mirrors / community re-uploads | HF search `demamba` | 🚫 none | — | No mirror exists as of today. |

**Recommendation:** do not wait on DeMamba weights. Either (a) train the XCLIP+Mamba
head ourselves on GenVideo-100K (10k clips/category, ModelScope, feasible on the 5090),
or (b) reallocate the `VISHWAS_DEMAMBA_WEIGHTS` slot to a detector that actually ships
weights — top candidates are in §3.

---

## 3. Full-AI-video (Sora/Veo/Kling-class) detectors (HIGH value)

| Name | Source | Direct download | Size | License | Arch | Evidence | Difficulty |
|---|---|---|---|---|---|---|---|
| **D3 (ICCV 2025)** — training-free | https://github.com/Zig-HS/D3 | ✅ no weights needed at all (second-order features over frozen DINOv2 ViT-S/14 via torch.hub); MIT code | 0 | MIT | Training-free statistical detector | Paper reports strong cross-generator results on GenVideo/Pika/Sora-class; `eval.py` provided | **LOW** (best effort/reward) |
| **Skyra-RL / Skyra-SFT (CVPR 2026)** | https://huggingface.co/JoeLeelyf/Skyra-RL , …/Skyra-SFT (collection `JoeLeelyf/skyra`) | ✅ 4-shard safetensors each: RL shards sum ≈16.59 GB, SFT ≈16.58 GB (Qwen2.5-VL-7B finetune) | ~16.6 GB | **CC BY 4.0** (README: "Skyra model weights are released under CC BY 4.0"; adhere to Kinetics/Panda/HD-VILA terms) | Qwen2.5-VL-7B grounded-artifact-reasoning detector; vLLM-friendly | CVPR 2026; ViF-Bench balanced-acc eval scripts in repo | MED-HIGH (VLM prompting loop instead of prob head; fits 5090 32 GB in bf16) |
| **NSG-VD (NeurIPS 2025 Spotlight)** | https://github.com/ZSHsh98/NSG-VD | ✅ ckpts committed in-repo, e.g. `https://raw.githubusercontent.com/ZSHsh98/NSG-VD/main/ckpts/standard-Pika-mp.pth` → HTTP 200, etag-tagged; 6 files × 1,709,120 B (Pika/SEINE × mp/d settings) | 1.7 MB each | Apache-2.0 | Small MMD-head over physics-driven NSG features — **requires** guided-diffusion `256x256_diffusion_uncond.pt` prior + reference-set MMD | NeurIPS'25 spotlight; +16 % Recall / +10.75 % F1 over SOTA claims | HIGH (output is an MMD statistic vs a reference set, not a calibrated P(fake); ImageNet-256 diffusion prior is a domain stretch for WhatsApp clips) |
| VideoVeritas (ICML 2026) | https://github.com/EricTan7/VideoVeritas | ⚠️ weights on ModelScope (`EricTanh/VideoVeritas`) — reachable but not byte-verified | n/a | Apache-2.0 (code) | perception-pretext-reward MLLM, vLLM deploy | ICML 2026 | MED |
| ReStraV (NeurIPS 2025) | https://github.com/ChristianInterno/ReStraV | ⚠️ no trained MLP released; ships `demo.py` + feature extractor; you train a 21-D logistic head yourself | tiny | Apache-2.0 | DINOv2 trajectory geometry (straightening) | NeurIPS 2025 | LOW-MED |
| AIGVDet (PRCV 2024) | https://github.com/multimediaFor/AIGVDet | ⚠️ weights on Google Drive folder (page HTTP 200) + RAFT flow model also GDrive | n/a | research-purpose-only clause | two-stream RGB+optical-flow ResNet | PRCV 2024 | MED (flow preprocessing cost per clip) |
| GenVidBench (AAAI 2026) | https://github.com/genvidbench/GenVidBench | 🚫 no ckpt assets in release/listing | — | — | mmaction-style benchmark framework | useful as *training* infra, not weights | — |

Excluded: `APRIL-AIGC/T3-Video` (HF, Apache-2.0) is a Wan2.1/2.2 video **generator**
accelerator, not a detector — do not wire into the detector slot.

### TOP PICK: **D3** for immediate capability + **Skyra-RL** as the quality ceiling
- D3: zero checkpoint risk (uses public DINOv2), MIT, deterministic; plan = extract N=16
  frames/clip → DINOv2 ViT-S/14 features → second-order statistics → threshold from
  `eval.py`. Slot: reuse `VISHWAS_DEMAMBA_WEIGHTS` env (rename display) or add
  `VISHWAS_AIGCVIDEO_WEIGHTS` with `family="video"`.
- Skyra-RL when we want a reasoning-grade verdict: serve via vLLM on the 5090, prompt
  template from repo `eval/`, parse Yes/No artifact-grounded answer into a probability.
- NSG-VD only if we later want its physics-prior ensemble member (its 1.7 MB heads are
  trivially downloadable; the burden is the diffusion-prior runtime).

---

## 4. AV-sync / cross-modal alternatives (MEDIUM-HIGH)

Key finding first: **our HAVIC weights are the official ones.** `JielunPeng/HAVIC` (HF,
MIT) hosts `best_ft_model.pth` (858,837,738 B — byte-identical role to our copy at
`/opt/verisafe/models/havic/best_ft/best_ft_model.pth`) and `pt_model.200.pth`
(972,770,538 B). Official repo: https://github.com/tuffy-studio/HAVIC (MIT, CVPR'26
Findings). Saturation at ~1.0 therefore points to a **preprocessing/integration bug**
(their pipeline feeds AudioMAE + MARLIN specific 25 fps face-crops + 16 kHz audio crops
via `video_data_engine/preprocess_pt_dataset.py`) — fix before swapping models.

| Name | Source | Direct download | Size | License | Arch | Evidence | Difficulty |
|---|---|---|---|---|---|---|---|
| **SyncNet v2 (joonson/syncnet_python)** | https://github.com/joonson/syncnet_python | ✅ `http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model` → HTTP 200, 54,573,114 B (+ `sfd_face.pth` face det, same host) | 54.6 MB | MIT | lip-sync AV offset/confidence CNN | canonical Oxford lip-sync model (898★ repo); mature `demo_syncnet.py` pipeline | **LOW** |
| **BATFD (LAV-DF authors, CVPR'23 line)** | https://huggingface.co/ControlNet/LAV-DF | ✅ `batfd_default.ckpt` (66,095,063 B) · ✅ `batfd_plus_default.ckpt` (1,716,289,320 B) | 66 MB / 1.6 GB | **CC BY-NC 4.0** | audio-visual temporal forgery localizer/detector | established AVFD benchmark family (arXiv 2204.06228 / 2305.01979) | MED (needs their frame/audio windowing) |
| ics-av-deepfake (ICCV 2025) — AVFF-lineage sync-consistency | https://github.com/AshutoshAnshul/ics-av-deepfake | ⚠️ checkpoints on NTU OneDrive share (gated-ish, not curl-verifiable) | n/a | none stated | intra+cross-modal synchronization pretraining → classifier | ICCV 2025 | HIGH |
| X-AVDT (CVPR 2026) | https://github.com/youngseo0526/X-AVDT | ⚠️ no standalone detector ckpt yet; requires Hallo bundle (`fudan-generative-ai/hallo` on HF) for diffusion-inversion features | heavy | TBD | AV cross-attention over Hallo features | CVPR 2026; MMDF dataset on HF | HIGH |
| MRDF (Apache-2.0) | https://github.com/Vincent-ZHQ/MRDF | ⚠️ example FakeAVCeleb ckpt on Google Drive folder (reachable) | n/a | Apache-2.0 | AVHuBERT-based margin/CE regularized detector | ACM MM'23 | MED (fairseq dep) |
| OpenAVFF | https://github.com/JoeLeelyf/OpenAVFF | ⚠️ unofficial AVFF clone | n/a | **no license file** — avoid | AVFF reimpl | unofficial | — |

### TOP PICK
1. **Fix HAVIC preprocessing first** (weights already official & present) — highest EV.
2. Add **SyncNet v2** as an always-on AV-coherence evidence channel: MIT, 54 MB,
   battle-tested; emit `|offset|` + sync-confidence distance as features/fusion inputs
   (large offset/high distance ⇒ tamper suspicion). New slot
   `VISHWAS_AVSYNC_WEIGHTS=/opt/verisafe/models/syncnet/syncnet_v2.model`.
3. If a learned cross-modal posterior is still wanted: **BATFD `batfd_default.ckpt`**
   (66 MB direct wget) — flag CC BY-NC for the commercial-later caveat.

---

## 5. Face-swap detectors better than EFFORT on FF++ (LOW-MEDIUM)

Root-cause note before shopping: `VISHWAS_EFFORT_WEIGHTS` currently points at
`effort/chameleon/effort_chameleon.pth` (deploy/vishwas-secrets.env) — an **AIGC-images
(Chameleon-trained)** checkpoint applied to FF++ C23 face swaps. That alone plausibly
explains the inverted 0.385 AUC. The FF++-trained Effort ckpt (`effort_ffpp.pth`,
sha256-prefix `8d86711f098d…`, 1,213,769,519 B) is already recorded in
`MODEL_WEIGHTS_MANIFEST.json` — **try swapping to it before adding any new model.**

| Name | Source | Direct download | Size | License | FF++ c23 evidence | Difficulty |
|---|---|---|---|---|---|---|
| **DeepfakeBench v1.0.1 ckpt pack (13 detectors)** | https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1 | ✅ all assets direct: `ucf_best.pth` 188,006,161 B (HEAD 302→200 verified), `spsl_best.pth` 87,764,683, `f3net_best.pth` 90,398,839, `recce_best.pth` 191,466,941, `core_best.pth` 87,766,765, `srm_best.pth` 222,088,575, `ffd_best.pth` 87,796,541, `xception_best.pth` 87,763,531, `effnb4_best.pth` 70,979,261, `cnnaug_best.pth` 85,285,901, `capsule_best.pth` 15,694,749, `meso4_best.pth` 117,943, `meso4Incep_best.pth` 126,419 | 0.1–222 MB | **CC BY-NC 4.0** (CUHK-SZ custom header) | every ckpt trained+reported on the bench's FF++ c23 protocol (paper tables give per-method AUC; UCF/CORE/SPSL/F3Net are the strong ones) | MED (replicate DeepfakeBench transform: crop→299²/224² per detector yaml in `training/config/detector/`) |
| **SBI official (CVPR 2022 Oral)** | https://github.com/mapooon/SelfBlendedImages | ⚠️ Google Drive: c23 ckpt `1X0-NYT8KPursLZZdxduRQju6E52hauV0`, raw ckpt `12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV` (pages reachable; Drive not byte-verifiable) | ~70–80 MB | custom **non-commercial research-only** (Yamasaki Lab; commercial needs license) | EfficientNet-B4 trained on SBIs from FF-c23 explicitly (changelog 10.9.2022); paper AUC ≈ 93 % c23 | LOW-MED (simple EffNet-B4 inference; Xception-style preprocessing) |
| LSDA/CADDM (ICCV 2023) | https://github.com/pandalandala/CADDM-Implicit_Identity_Leakage | ⚠️ single GDrive file `1JNMI4RGssgCOl9t05jkUa6imnw5XR5id` (README §Pretrained weights); Apache-2.0 code; needs 81-pt landmark detector | n/a | Apache-2.0 (code) | ICCV'23 paper reports FF++ c23 | MED-HIGH (landmark pipeline) |
| UIA-ViT (IJCAI 2023) | https://github.com/wany0824/UIA-ViT | ⚠️ GDrive folder `1zPx4TLEfLnJDZYpSV0LhFvrMEEDzroB0`; **no license file** | n/a | unspecified — treat as restricted | DD-VFF/FakeAVCeleb focus, FF++ secondary | MED |
| Face X-Ray | official code never fully released with weights; only unofficial repos (neverUseThisName/Face-X-Ray etc., no maintained ckpts) 🚫; DeepfakeBench v1.0.2 ships its landmark support pkls only | — | — | — | — | — |

### TOP PICK
1. **Zero-cost first move:** swap `VISHWAS_EFFORT_WEIGHTS` to `effort_ffpp.pth`
   (already on disk / re-fetchable from YZY-stack/Effort-AIGI-Detection release, CC BY-NC
   — same terms as today) and re-run the FF++ C23 probe.
2. **New model:** pull `ucf_best.pth` + `spsl_best.pth` + `f3net_best.pth` from
   DeepfakeBench v1.0.1 (direct URLs above), stand them up behind ONE adapter
   ("deepfakebench" family) that reads the matching detector yaml for preprocessing.
   UCF (ICCV'21) is the strongest single choice; all three report solid FF++ c23 AUC.
   License CC BY-NC 4.0 = fine for local research; renegotiate before any commercial use.

---

## Cross-cutting notes

- **Runtime envelope (RTX 5090, WSL, Python 3.12):** Spectra-AASIST3 (~0.7 GB bf16),
  SyncNet (54 MB), UCF/SPSL/F3Net (<250 MB each) are all light; Skyra-RL needs the VLM
  serving lane (~17 GB bf16, fits 32 GB card); NSG-VD/BATFD-plus need diffusion or large
  ckpt lanes. Avoid anything requiring `fairseq` on Python 3.12 (ash56/ssl-aasist, MRDF)
  unless containerized separately.
- **License gates (non-commercial flags):** DeepfakeBench ckpts + AASIST3 + LAV-DF +
  EFFORT (CC BY-NC family), SBI + UIA-ViT (custom NC / unspecified). Permissive: Spectra-
  AASIST3 + ash56 + arnabdas (Apache/MIT), SyncNet (MIT), NSG-VD/D3/ReStraV/VideoVeritas/
  MRDF (Apache/MIT), Skyra (CC BY 4.0).
- **Manifest corrections to make:** gate `demamba`: weights never released (keep code
  link); consider adding gates `spectra-aasist3`, `syncnet-v2`, `batfd`, `ns-gvd`,
  `skyra`, `deepfakebench-pack` with the verified URLs above.
