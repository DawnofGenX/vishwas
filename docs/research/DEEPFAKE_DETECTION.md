# Deepfake Detection — Research & Evidence Note (P1)

**Vishwas** | generated 2026-08-19 by orchestrator direct verification (subagent batch deleg_02086fd4 died on provider rate-caps before writing; findings below re-verified against primary sources). Every arXiv ID verified via `export.arxiv.org/api/query`; venue taken from each record's own `arxiv:comment` field.

## 0. Headline finding (read first)

**Status after second-pass verification (2026-08-19, deleg_78bafee3 salvage + direct re-query):**
- **EFFORT — RESOLVED.** “Effort” = *“Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection”*, arXiv **2411.15633** (v4), **ICML 2025 Oral**. Official implementation: `github.com/YZY-stack/Effort-AIGI-Detection` (self-described “Official implementation of ICML 2025 Oral 🏆”, ~228★, no SPDX license declared in GitHub API metadata → treat weights licensing as unconfirmed). Role fits our spatial/generic-AIGI slot; note it is an image-level detector (no temporal stream).
- **VB+StA — NOT FOUND.** Zero hits on arXiv (title/abstract), CVPR2025 open-access listing scan, and web search. Likely an internal codename or mis-attribution in the original spec. Keep AV-local-temporal (**2501.08137**, ICASSP 2025) as the standing substitute for the temporal slot.
- **Fake-Mamba — RESOLVED.** *Real*: “Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention’s Alternative”, arXiv **2508.09294** (Xuan, Zhu, Zhang, Lin, Kinnunen — NTU/Toki group), XLSR front-end + bi-Mamba. The earlier “zero hits” note was a query miss, not a non-existence.
- **AASIST base-paper correction:** AASIST proper is **“Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks”**, arXiv **2110.01200** (Jung, Heo, Tak, Shim, Chung — NCSOFT/KAIST line, submitted ICASSP 2022). The previously cited 2006.03214 (Wu/Liu/Lee black-box defense) is a *related* SSL anti-spoofing paper, not AASIST.

Per project rules these are flagged **UNVERIFIED, not fabricated**: the capability modules keep their weight-file gates (`VISHWAS_EFFORT_WEIGHTS`, etc.) so they activate only when someone drops a real weights path in. The pipeline runs fully without them (T0/T1 heuristics + any gated model that exists). If the user has PDFs/links to those exact papers, point Vishwas at them and the gates light up.

Everything else below is verified.

## 1. Video detection — verified candidates

| Name | arXiv (verified) | Venue (from arXiv comment) | Role in our pipeline | CPU feasibility |
|---|---|---|---|---|
| DeMamba | 2405.19707 | AI-Generated Video Detection on Million-Scale GenVideo Benchmark (2024-05) | general/degraded-video specialist; million-scale pretraining targets exactly the transcoding-drift problem | RUNNABLE-SLOW on i5-8250U if weights available (Mamba backbone ~100–400ms/frame @256² est.) |
| UMCL | 2511.18983 | accepted to **IJCV** (24-page manuscript) | cross-compression-rate multimodal contrastive learning — direct evidence-based design input for our robustness transforms | research reference; not shipped directly |
| EFFORT (OrthAlign family) | **2411.15633** | **ICML 2025 Oral** (per official repo) | orthogonal-subspace decomposition for generalizable AIGI images — the actual paper behind the spec's “Effort” | RUNNABLE-SLOW if weights obtainable (repo license unconfirmed) |
| FakeFormer | 2410.21964 | efficient vulnerability-driven transformers, generalisable | generalist alternative when Effort weights unavailable | RUNNABLE-SLOW |
| AV-temporal (local temporal inconsistencies) | 2501.08137 | **ICASSP 2025** | temporal/AV specialist — best verified substitute for the missing VB+StA slot | RUNNABLE-SLOW |
| HOLA (hierarchical contextual aggregation) | 2507.22781 | 2025-07 | hierarchical AV fusion | RUNNABLE-SLOW |

**Design consequence:** the “spatial face / temporal / degraded” three-slot layout stands; slots fill from (a) whatever gated weights appear, (b) offline heuristics already implemented (temporal flicker, face-region artifact, compression-blockiness), and (c) the above verified papers as the integration roadmap for later weight drops.

## 2. Audio deepfake — verified

| Model | arXiv (verified) | Venue | Notes |
|---|---|---|---|
| AASIST (original) | **2110.01200** | submitted ICASSP 2022 (per arXiv comment) | the classic single-model spectro-temporal graph-attention anti-spoofing backend (Jung et al.); ASVspoof-lineage baseline |
| SpAArSIST | 2606.11674 | 2026-06 | deployment-oriented sparsified AASIST (replaces learned pooling + stack-node attention) — best candidate for CPU-constrained serving |
| AASIST3 | 2408.17352 | ASVspoof 2024 (KAN-enhanced, SSL features) | challenge SOTA of 2024 |
| SSL front-ends (BEATs/HuBERT/Wav2Vec2 fine-tuned) | 2111.07725 (v3, ISCA Odyssey2022) | sub-band analysis added in v3 — relevant to bandwidth-limited phone audio | direct evidence SSL backbones transfer to spoofing CM |
| Towards Scalable AASIST (graph attention) | 2507.11777 | 2025-07 | current AASIST-lineage refresh; our `VISHWAS_AASIST_WEIGHTS` gate target |
| **Fake-Mamba** | **2508.09294** | 2025-08 (real-time speech deepfake, bi-Mamba replaces self-attention) | the paper the spec meant by “Fake-Mamba”; XLSR front-end, latency-focused |
| XLSR-Mamba | 2411.10027 | IEEE SPL 2025 (accepted) | Mamba dual-column bidirectional for spoofing audit — sibling line |
| RawBMamba | 2406.06086 | Interspeech 2024 | end-to-end bi-Mamba on raw waveforms; good fit for variable-bitrate voice memos |
| XMU ASVspoof 5 systems | 2509.18102 | ASVspoof 5 challenge | state of the art on latest challenge incl. TTA track (real-world channel) — useful eval template |
| SIGNL (label-efficient spectral-temporal graph) | 2501.04942 | 2025 | label-efficient = relevant given tiny Indian-language labeled sets |
| Wav2DF-TSL | 2509.04161 | 2025 | two-stage pretraining, efficient experts — CPU-friendly architecture story |

## 3. Cross-modal (audio-video coherence)

| Item | arXiv (verified) | Venue | Use |
|---|---|---|---|
| Holistic Audio-Visual Intrinsic Coherence (“Leave No Stone Unturned”) | 2603.23960 | 2026-03 | **best current HAVIC-class method found**; holistic AV intrinsic-coherence for manipulation localization |
| From Talking to Singing | 2605.27944 | **ICML 2026** (per comment) | new challenge set: singing AV deepfakes — future threat-surface note |
| AV local temporal inconsistencies | 2501.08137 | ICASSP 2025 | shared w/ section 1 |

Our `cross_modal.py` implements modality-localized correlation (lip-sync proxy, AV correlation sign/strength) behind `VISHWAS_HAVIC_WEIGHTS`; without weights it degrades to ffmpeg-measured A/V offset + pitch-contour correlation — honest heuristic tier.

## 4. Robustness to transcoding (the actual failure mode on WhatsApp)

| Paper | arXiv | Venue | Quantified claim |
|---|---|---|---|
| UMCL | 2511.18983 | IJCV | detector performance tracked across compression rates; unimodal-generated multimodal CL keeps scores aligned across bitrate regimes |
| Pay Less Attention to Deceptive Artifacts | 2506.20548 | 2025 (20pp) | compressed deepfakes: naive detectors over-rely on compression artifacts themselves → misfires on legit compressed video |
| Measuring Robustness of Audio DFD under Real-World Corruption | 2503.17577 | 2025 | corruption battery incl. resampling/noise/bandwidth — maps to our Opus/AAC/MP3 matrix |
| Benchmarking Audio DFD Robustness in Real-World Communication Scenarios | 2504.12423 (v4) | **EUSIPCO 2025** | communication-channel degradation (packet loss, codec) benchmark — the most WhatsApp-realistic public study found |
| Proteus | 2606.29544 | 2026 | automated adversarial robustness testing of audio DFDs |

**Direct implication for Vishwas** (and why we never average raw scores): after WhatsApp's Opus+AAC re-encode, detector AUROC drops enough that a single-model threshold flips verdicts. Our answer is the OOF-stacked LR with calibration + selective prediction (abstain when signal conflict), plus the transform-matrix stress test in P7 that replays every detector through the same codec ladder and records per-score drift.

### 4.1 Learned-stage adversarial sensitivity — small-N real-weights check (2026-08-21)

Phase 5 Task A red-team re-run, now that all three learned families are live.
**This is a small-N sanity check on ONE synthetic fixture (N=7 runs), not a
benchmark** — it answers "do realistic channel transforms flip the learned
score today", nothing more.

**Method.** Base clip: 4 s `testsrc`+`sine`, 640×360 @ 8 fps, h264+AAC
(same recipe as the P1 proof fixture; original purged by zero-retention).
Six transform variants built with ffmpeg (`VISHWAS_FFMPEG_THREADS=1`),
audio-affecting transforms keeping the video stream and video-affecting ones
keeping the audio stream so all three families score on every variant:

| Variant | Transform |
|---|---|
| v_aac32k | audio → AAC 32 kbps low-bitrate (video copied) |
| v_mp3_64k | audio → MP3 64 kbps lossy (video copied) |
| v_resample8k | audio resampled to 8 kHz (video copied) |
| v_h264crf40 | video h264 re-encode CRF 40 (audio copied) |
| v_scale50 | 50% luma/frame scale → 320×180 (audio copied) |
| v_fps6 | frame drop 8 fps → 6 fps (audio copied) |

Executed via **direct capability calls** (not 7 CLI runs): each model loaded
once into the production seam (`model_adapters.resolve()` → arch wrapper),
then per variant the exact capability code paths ran — AASIST 3-crop median
(`_multi_crop` seam), EFFORT 8-frame median (`_effort` seam), HAVIC
(`_havic_check`). Chosen to avoid reloading ~5 GB of checkpoints per variant;
cross-validated: driver baseline scores match the same-day CLI E2E run
exactly (0.997 / 0.378 / 0.993), confirming seam equivalence.

**Results** (`prob_deepfake` / `prob_inconsistent`, baseline vs variants):

| Clip | AASIST (audio) | EFFORT (video) | HAVIC (cross-modal) |
|---|---|---|---|
| clip_av (baseline) | 0.997 | 0.378 | 0.993 |
| v_aac32k | 0.997 | 0.375 | 0.993 |
| v_mp3_64k | 0.997 | 0.375 | 0.993 |
| v_resample8k | 0.997 | 0.375 | 0.996 |
| v_h264crf40 | 0.997 | 0.395 | 0.992 |
| v_scale50 | 0.997 | 0.388 | 0.993 |
| v_fps6 | 0.997 | 0.367 | 0.994 |
| **max \|Δ\| vs baseline** | **0.000** | **0.017** | **0.003** |

**Verdict per family** (stable = max score delta < 0.15, no flips):
- **AASIST: STABLE** (Δ = 0.000 across all six transforms).
- **EFFORT: STABLE** (spread 0.367–0.395; worst Δ 0.017 under CRF-40 re-encode).
- **HAVIC: STABLE** (spread 0.992–0.996; worst Δ 0.003 under 8 kHz resample).

No verdict flips anywhere in the matrix.

**Honest caveats (read before citing this as robustness evidence):**
1. **Ceiling saturation**: the fixture is a synthetic tone/testsrc pattern, and
   both AASIST (0.997) and HAVIC (~0.99) sit at their score ceilings. A pinned
   score cannot flip by construction, so "stable" here partly reflects
   saturation, not measured margin. EFFORT (mid-range 0.37–0.40) is the only
   family whose stability is genuinely informative.
2. N=7 runs, one fixture, one transform severity per family — far below the
   codec-ladder coverage of §4's cited benchmarks (UMCL, 2504.12423).
3. Real deepfake fixtures (FaceForensics++/CelebV-class) must replace the
   synthetic clip before any deployment claim; eval-only dataset use per §5.
4. The heuristic `cross_modal_av` probe did shift class on this fixture
   (weakly_synced, r=0.195) — heuristics remain the more fragile tier, which
   is exactly why fusion never trusts a single family.


## 5. Labeled data & minimal eval design
- Public datasets referenced by the papers above: FaceForensics++/CelebV-Flip/FoFo (video); ASVspoof 2015/2019/2021(+TTA)/5 (audio); GenVideo (million-scale, DeMamba). Sizes span 10GB–2TB — full training is out of scope for this box; we use them for **eval-only** downloads when a specific gated model lands.
- Minimal selective-prediction calibration design (implemented in `fusion_train.py`): 5-fold OOF → per-capability logistic stacking on [value, gap-flag] features → temperature-scaling grid minimizing NLL → coverage/risk curve; demo dataset generator is seeded-reproducible synthetic (labeled as such in outputs).

## 6. Recommendation table (current state of the codebase)

| Component | Slot | Gate env var | Weights status 2026-08-19 | CPU verdict | Integration effort |
|---|---|---|---|---|---|
| Effort (ICML'25 Oral) | spatial/AIGI | VISHWAS_EFFORT_WEIGHTS | **RESOLVED 2411.15633**; official repo YZY-stack/Effort-AIGI-Detection (license unconfirmed) | gated-off until weights downloaded | 1–2 h to load into gate once weights exist |
| VB+StA | temporal | VISHWAS_VBSTA_WEIGHTS | **NOT FOUND — kept as placeholder only** (substitute: 2501.08137 AV-local-temporal) | gated-off | 1–2 h |
| DeMamba | degraded/general | VISHWAS_DEMAMBA_WEIGHTS | **verified** 2405.19707 | RUNNABLE-SLOW if weights obtainable | 2–4 h (weight-format adapter) |
| AASIST(-lineage) | audio | VISHWAS_AASIST_WEIGHTS | **verified** 2006.03214 / 2507.11777 | RUNNABLE-SLOW | 2–4 h |
| XLSR-Mamba / RawBMamba | audio-alt | (fold into AASIST gate or VISHWAS_SSL_AUDIO_WEIGHTS) | **verified** 2411.10027 / 2406.06086 | RUNNABLE-SLOW | 2–4 h |
| HAVIC-class (Holistic AV) | cross-modal | VISHWAS_HAVIC_WEIGHTS | **verified** 2603.23960 | RUNNABLE-SLOW | 4–6 h (needs both modalities loaded) |

## Gaps and risks
1. **VB+StA unverifiable** — no public record found on arXiv/CVPR2025/web despite multiple distinct query families; treat as internal codename/mis-attribution. Do **not** cite it in user-facing docs. **EFFORT is resolved** (2411.15633, ICML 2025 Oral, official repo available); its repo declares no SPDX license → verify weights licensing before shipping.
   *Provenance:* IDs recovered from deleg_78bafee3 partial output (`/home/hermes/deepfake-factcheck/arxiv_raw.json` — subagent finished research but died on 429 before delivering the write-up; raw data salvaged).
2. No public weights were located for any of the five named models in this pass — HF Hub lookup was limited by rate caps; treat “weights available” as TODO-per-model, not assumed.
3. Bandwidth-limited-audio evidence is strong for *corruption measurement* (2503.17577, 2504.12423) but few papers publish detector scores specifically at 8 kHz/Opus; our transform-matrix battery fills that gap empirically on this box.
4. Subagent infra flakiness (HTTP 429 concurrency cap, 504 Cloudflare) burned one delegation batch — subsequent research done directly by orchestrator with serial curl (worked reliably: arXiv API, GitHub API/search, PyPI JSON, gov sites).
