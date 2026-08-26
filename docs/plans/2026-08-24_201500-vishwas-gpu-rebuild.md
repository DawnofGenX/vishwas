# Vishwas GPU-Era Rebuild Plan: Fresh Weight Downloads + Better Models + CUDA

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Rebuild Vishwas's learned-model tier for the new hardware reality — RTX 5090 Laptop (24 GB, sm_120, CUDA 13.0) replaces the thermal-capped i5-8250U CPU assumption. Download all weights fresh, enable the previously-skipped EFFORT tier, move inference to CUDA, and re-baseline performance.

**Architecture:** Three tracks. (A) CUDA enablement: torch 2.13.0+cu130 already sees the GPU in `~/vllm-env`; vishwas currently hardcodes `map_location="cpu"` and runs on CPU — add a device-resolution seam, defaulting to cuda when available with CPU fallback. (B) Weights: fresh downloads of AASIST + HAVIC + RawBMamba via the project fetcher, PLUS EFFORT which was skipped only for thermal reasons (now viable on GPU). (C) Verification: hermetic suite, gate probes, E2E matrix, and a dated PERFORMANCE.md re-baseline (old numbers are CPU-era and obsolete).

**Tech Stack:** torch 2.13 cu130 (`~/docling-python` flat site-packages AND `~/vllm-env` venv both have it), `scripts/fetch_model_weights.sh` (DRY_RUN=0), HuggingFace/GitHub direct downloads, pytest, systemd webhook unit.

---

## Current context (verified 2026-08-24 20:00)

- **GPU**: RTX 5090 Laptop, 24 GB VRAM, sm_120 (Blackwell), driver CUDA 13.3, 42 °C idle
- **torch**: `~/vllm-env/bin/python` → torch 2.11.0+cu130, `cuda.is_available()=True`, device recognized; `~/docling-python` (the PYTHONPATH tree run_vishwas.sh uses) has torch 2.13.0+cu130
- Repo `/home/hermes/vishwas` clean at `b4a8c0d`; suite collects 336 tests
- `/opt/vishwas/models/` absent — ALL gates down; clamscan not installed
- Code facts driving Track A:
  - `model_adapters.py:187` and all four capability loaders hardcode `map_location="cpu"`
  - `model_archs/aasist.py score()` builds tensors on CPU implicitly (no `.to(device)`)
  - `model_archs/fakemamba.py:46` hardcodes `device="cpu"` (CPU-shim vendoring)
  - No `cuda` references anywhere in model_archs except fakemamba's comment
- Old plan constraint OBSOLETE: "EFFORT not viable under Vishwas thermal caps" (fetch_model_weights.sh:68) — GPU removes this
- RawBMamba caveat: vendored CPU reference-scan shims work but were built because upstream was CUDA-hard; on real CUDA the shims still function (pure PyTorch), just slower than fused kernels. Keep shims; optionally revisit later.
- WavLM-Large frontend inside AASIST checkpoint is the heavy part (~1.2 GB params); 24 GB VRAM fits all four gates simultaneously with room to spare.

**Model upgrade decisions (better models, per operator):**
| Gate | Action | Rationale |
|---|---|---|
| AASIST (audio) | Download same checkpoint (HABLA WavLM-AASIST) | SOTA survey 2026-08: still best open single model; 2025-26 winners are SSL ensembles, not better single checkpoints. GPU makes its stage ~sub-second |
| EFFORT (video/face) | **NEW: download** (was thermally skipped) | Still near-SOTA generalizable detector (CVPR 2025); 2026 successors have code but NO public checkpoints. CC BY-NC flagged |
| HAVIC (av-crossmodal) | Download same checkpoint | MIT confirmed live on HF (updated Jul 2026). Companion `pt_model.200.pth` fetch if arch requires |
| Mamba slot (audio 2nd opinion) | **REPLACE RawBMamba → XLSR-Mamba-LA** (`AustinXiao/XLSR-Mamba-LA`, verified HF: MIT license, model.safetensors ~1.28 GB incl. XLSR frontend) | RawBMamba third-party eval: ASVspoof5 EER ~37.9% (weak vs modern generators) + no license. XLSR-Mamba: MIT, public safetensors, ASVspoof21 LA EER 0.93% / In-the-Wild 6.71%. Shares fairseq/WavLM stack with HABLA so vendoring cost is low |
| NEW optional: AuViRe (AV localization) | ADD later as separate task if segment-level evidence wanted (`ckoutlis/auvire-avdeepfake1m`, Apache-2.0, 36 MB safetensors — verified live on HF) | Localization not whole-video classification; cheap add-on, defer |

**Research provenance:** full survey at `research/deepfake-sota-survey-2026-08.md` (arXiv/HF/GitHub API research, 2026-08-24); all load-bearing HF repo facts (licenses, files) independently re-verified via Hub API the same day.

---

## Task 1: Add CUDA device seam to model loading (TDD)

**Objective:** All learned stages load and run on GPU when available, CPU otherwise. One seam, no per-callsite edits.

**Files:**
- Create: `src/vishwas/device.py`
- Modify: `src/vishwas/model_adapters.py:182-188` (torch.load call)
- Modify: `src/vishwas/model_archs/aasist.py` (score(): move wav + model to resolved device)
- Modify: `src/vishwas/model_archs/havic.py`, `effort.py` (same pattern where tensors/materialized modules exist)
- Test: `tests/test_27_device_seam.py`

**Step 1: Write failing test**

```python
"""test_27_device_seam.py — device resolution seam (hermetic)."""
import os
from unittest import mock


def test_resolve_device_prefers_cuda_when_available(monkeypatch):
    import fake_torch_guard  # noqa: F401  -- see conftest pattern in test_26
    from vishwas.device import resolve_device
    monkeypatch.setenv("VISHWAS_DEVICE", "")
    fake_torch = mock.MagicMock()
    fake_torch.cuda.is_available.return_value = True
    with mock.patch.dict("sys.modules", {"torch": fake_torch}):
        assert resolve_device() == "cuda"


def test_resolve_env_override_wins():
    from vishwas.device import resolve_device
    os.environ["VISHWAS_DEVICE"] = "cpu"
    try:
        assert resolve_device() == "cpu"   # explicit override beats autodetect
    finally:
        del os.environ["VISHWAS_DEVICE"]


def test_resolve_falls_back_to_cpu_without_torch():
    from vishwas.device import resolve_device
    with mock.patch.dict("sys.modules", {"torch": None}):
        assert resolve_device() == "cpu"
```

(Adjust the torch-mock mechanics to match whatever stubbing pattern `tests/test_26_clamav_guard.py` uses for optional binaries — read that file first and mirror it.)

**Step 2:** Run `PYTHONPATH=src python3 -m pytest tests/test_27_device_seam.py -q` — expect FAIL (module missing).

**Step 3: Implement minimal seam**

```python
# src/vishwas/device.py
"""Single device-resolution seam for learned-model inference.

Resolution order:
  1. VISHWAS_DEVICE env ("cpu"/"cuda") — explicit operator override
  2. cuda when torch reports it available
  3. cpu fallback (never raises)
"""
from __future__ import annotations

import os


def resolve_device() -> str:
    override = os.environ.get("VISHWAS_DEVICE", "").strip().lower()
    if override in ("cpu", "cuda"):
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
```

Then:
- In `model_adapters.py`: replace `map_location="cpu"` with `map_location=resolve_device()` (import at top). After load, if the returned object is an `nn.Module`, leave placement to the ArchSpec `score()` paths (loading to meta/cpu then moving is safest: keep `map_location` as resolved device directly).
- In `aasist.py score()`: after building `wav`, do `dev = resolve_device(); model_dev = next(model.parameters()).device` — if they differ, `model.to(dev)` once (guard with a cached flag or move at build()) and `wav.to(dev)`; final `.item()` works regardless of device.
- Same pattern for havic/effort score paths (read each module first — follow their existing structure).
- Leave `fakemamba.py` CPU-hardcoded for now (Task 5 covers it).

**Step 4:** Run test — expect PASS. Then full suite: `PYTHONPATH=src python3 -m pytest tests/ -q` — expect 333+ passed (new tests add to count), zero regressions.

**Step 5:** Commit: `feat(device): single device-resolution seam, cuda-preferred`

---

## Task 2: Provision /opt/vishwas/models + fetch AASIST & HAVIC

**Objective:** Fresh-download the big checkpoints straight onto the new disk.

**Files:**
- Create: `/opt/vishwas/models/{aasist,havic/best_ft,rawbmamba}/`

**Step 1:** `sudo mkdir -p /opt/vishwas/models/{aasist,havic/best_ft,rawbmamba} && sudo chown -R hermes:hermes /opt/vishwas`

**Step 2 (background=true, notify_on_complete=true):**
```bash
export VISHWAS_MODEL_DIR=/opt/vishwas/models VISHWAS_FETCH_DRY_RUN=0
bash ~/vishwas/scripts/fetch_model_weights.sh
```
Expected: aasist/best_model.pth (3791.7 MB) + havic/best_ft/best_ft_model.pth (858.8 MB). Network-bound, thermal-safe. Expect 10–30 min depending on bandwidth.

**Step 3:** Verify sizes match the fetcher's manifest (`stat -c%s`), and each loads:
```bash
~/vllm-env/bin/python -c "import torch; torch.load('/opt/verisave/models/aasist/best_model.pth', map_location='cpu', weights_only=False)" 
```
(correct path prefix `/opt/vishwas/`.)

---

## Task 3: Enable EFFORT gate (previously thermally skipped)

**Objective:** Download the video-tier EFFORT chameleon checkpoint and wire its env var.

**Files:**
- Create: `/opt/vishwas/models/effort/chameleon/effort_chameleon.pth`
- Modify: `deploy/vishwas-secrets.env` (append VISHWAS_EFFORT_WEIGHTS)

**Step 1:** Read `src/vishwas/model_archs/effort.py` header + `provision_weight_env.sh` to get the exact expected relative path and filename (`effort/chameleon/effort_chameleon.pth` per provision script output).

**Step 2:** Locate the public checkpoint URL: consult `docs/research/` and the effort.py provenance header (the vendored code records the source repo). If no direct URL is recorded, search HF for the EFFORT repo cited there. If the checkpoint requires auth/non-public → STOP, report honestly, leave gate absent (graceful by design).

**Step 3 (background if >500 MB):**
```bash
curl -L --fail -o /opt/vishwas/models/effort/chameleon/effort_chameleon.pth <URL>
```

**Step 4:** Append to `deploy/vishwas-secrets.env`:
```
# EFFORT video/face deepfake tier — enabled 2026-08-24 (GPU box; thermal skip obsolete)
VISHWAS_EFFORT_WEIGHTS=/opt/vishwas/models/effort/chameleon/effort_chameleon.pth
```

**Step 5:** License check: confirm the source license permits this use (CC BY-NC per fetcher note); record verdict in the file comment. If NC-only is unacceptable to the operator, stop and ask.

---

## Task 4: Fetch RawBMamba checkpoint

**Objective:** REPLACE the fakemamba-slot model: download XLSR-Mamba-LA (MIT, better in-the-wild numbers) instead of RawBMamba. New env gate `VISHWAS_XLSRMAMBA_WEIGHTS`; RawBMamba kept as evaluation-grade ONNX fallback only.

**Files:**
- Create: `/opt/vishwas/models/xlsr-mamba/model.safetensors` (from HF `AustinXiao/XLSR-Mamba-LA`, ~1.28 GB incl. XLSR frontend)
- Create: `/opt/vishwas/models/xlsr-mamba/PROVENANCE.md`
- Modify: `deploy/vishwas-secrets.env` (add `VISHWAS_XLSRMAMBA_WEIGHTS=/opt/vishwas/models/xlsr-mamba/model.safetensors`; comment out FAKEMAMBA line with deprecation note)

**Step 1:**
```bash
mkdir -p /opt/vishwas/models/xlsr-mamba
curl -L --fail -o /opt/vishwas/models/xlsr-mamba/model.safetensors \
  https://huggingface.co/AustinXiao/XLSR-Mamba-LA/resolve/main/model.safetensors
```

**Step 2:** PROVENANCE.md: source HF repo, MIT license, paper arXiv 2411.10027, reported EERs (ASVspoof2021 LA 0.93% / DF 1.88% / In-the-Wild 6.71%). NOTE honestly: arch vendoring + strict-load wiring is a FOLLOW-UP commit cycle (needs fairseq pin like HABLA's `_wavlm`); until then this gate reports `unavailable` gracefully.

**Step 3:** Also fetch RawBMamba ONNX fallback (12.7 MB, HF mirror `SpeechAntiSpoofingBenchmarks/RawBMamba`) into `/opt/vishwas/models/rawbmamba/` with its own PROVENANCE.md: NO license, third-party-harness ASVspoof5 EER ~37.9% — eval-grade fallback only. Keep `scripts/verify_rawbmamba.py` green for the vendored CPU shim code (it stays in-tree for that fallback).

---

## Task 5: Install ClamAV

**Objective:** Restore malicious-file signature scanning (dep count 7→8).

**Step 1 (approval):** `sudo apt-get install -y clamav clamav-daemon`

**Step 2:** `sudo freshclam` (signature DB download, several minutes).

**Step 3:** Smoke: EICAR fixture through CLI (MZ-wrapped per `references/yara-rules-bundle.md` magic-byte routing notes) — expect clamscan FOUND evidence.

**Step 4:** `systemctl --user restart vishwas-webhook` then `/health` shows the new dep.

---

## Task 6: GPU smoke — every live gate loads AND scores on CUDA

**Objective:** Prove the silent-None pitfall is dead for all four gates on the new device seam.

**Step 1:** Direct adapter probe (never trust absence-of-error):
```bash
cd ~/vishwas && PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:src \
~/vllm-env/bin/python -c "
from vishwas.model_adapters import resolve
from vishwas.device import resolve_device
print('device:', resolve_device())
for g in ['VISHWAS_AASIST_WEIGHTS','VISHWAS_EFFORT_WEIGHTS','VISHWAS_HAVIC_WEIGHTS','VISHWAS_FAKEMAMBA_WEIGHTS']:
    r = resolve(g)
    try:
        obj = r.load(r.path)
        print(g, type(obj).__name__, 'loaded')
    except Exception as e:
        print(g, 'ERROR:', e)
"
```
IMPORTANT interpreter choice: probe with BOTH `~/vllm-env/bin/python` (known-good CUDA torch) AND system `python3` (which resolves the docling-python tree via PYTHONPATH). If the docling-python torch 2.13.0+cu130 wheel doesn't actually init CUDA on sm_120 (Blackwell needs cu128+; cu130 should be fine but VERIFY — `torch.cuda.get_device_name(0)` inside that interpreter), decide: either standardize run_vishwas.sh on vllm-env's interpreter or accept CPU for the webhook path. Record the decision.

**Step 2:** Score one real fixture per gate (smoke_5s.wav for audio gates; a short mp4 for EFFORT/HAVIC if fixtures exist under tests/fixtures/) and print wall time. Expect audio stage to drop from ~23 s (CPU) to low single-digit seconds.

**Thermal note:** GPU at 42 °C, 16 W — no thermal protocol constraints anymore for inference, but keep sequential runs during verification for clean timings.

---

## Task 7: Suite + E2E regression proof

**Objective:** Nothing broke; new posture measured.

**Step 1:** `cd ~/vishwas && PYTHONPATH=src python3 -m pytest tests/ -q` — expect 333+ passed, ≤3 skipped, plus new device-seam tests.

**Step 2:** E2E verdict matrix via `scripts/run_vishwas.sh cli`:
- text URL light path (<1 s, unchanged)
- smoke_5s.wav audio — AASIST stage LIVE on GPU
- MZ+EICAR malicious-file probe — clamscan+yara+VT fused

**Step 3:** Webhook round trip: restart unit, `/health` status ok, deps include everything.

---

## Task 8: Performance re-baseline + docs/skill update

**Objective:** The CPU-era numbers in PERFORMANCE.md are now wrong; replace with dated GPU numbers.

**Files:**
- Modify: `docs/PERFORMANCE.md` §7 — new dated table "2026-08-24 RTX 5090 baseline", keep old table labeled as legacy-CPU
- Modify: `docs/GAPS_AND_ENABLEMENT.md` — EFFORT slot now filled; posture delta
- Patch: skill `vishwas-operations` SKILL.md — rewrite "Weight gates", "Thermal protocol" (GPU inference changes it; ffmpeg threads note may persist for decode), "Measured stage costs" sections
- Commit: `perf(gpu): RTX 5090 re-baseline, EFFORT gate live, docs`

---

## Risks / tradeoffs / open questions

1. **Blackwell sm_120 compatibility**: cu130 wheels should support sm_120, but the docling-python torch must be verified in-interpreter (Task 6 Step 1 decision point). If broken, options: point run_vishwas.sh PYTHONPATH at vllm-env's site-packages instead, or pip-install matching torch into a dedicated vishwas venv.
2. **EFFORT checkpoint availability**: may need HF auth or be unlisted — Task 3 has an honest STOP clause.
3. **HAVIC companion file**: `pt_model.200.pth` (~973 MB) was deliberately skipped in CPU era; check `_havic` vendor code whether the ft model needs it at load. If yes, fetch it too.
4. **RawBMamba shims**: pure-PyTorch reference scan works on CUDA but won't use fused mamba kernels — correctness unaffected, speed mediocre. Fine for a second-opinion tier.
5. **VRAM co-existence** — see risk 6 (decided): webhook defaults to `VISHWAS_DEVICE=cpu` in secrets env; GPU is opt-in per invocation for heavy runs. `/health` gains a `device` field (Task 6).
6. **VRAM co-existence with local LLM serving — DECIDED (operator confirmed)**: the operator WILL serve local LLMs on the same RTX 5090. Therefore:
   - The device seam (Task 1) MUST support per-run override via `VISHWAS_DEVICE` (already in design) AND the webhook systemd unit must NOT hardcode cuda.
   - Default posture: webhook runs `VISHWAS_DEVICE=cpu` in `deploy/vishwas-secrets.env` so background WhatsApp traffic never competes with the LLM for VRAM; heavy/batch runs are invoked ad-hoc with `VISHWAS_DEVICE=cuda`.
   - Add a Task 6 sub-step: document the "GPU mode" invocation (`VISHWAS_DEVICE=cuda bash scripts/run_vishwas.sh ...`) and add a `/health` field `device` reporting the resolved device so the operator can confirm which mode the webhook is in.
   - VRM/heat note: GPU inference + LLM serving simultaneously is fine thermally, but serialize wall-clock if both hit at once.
6. **ClamAV apt install** needs approval; freshclam DB is a network download.

## Execution notes for multi-agent workflow

- Tasks 1, 5, 8 are code/docs tasks (delegate normally). Tasks 2–4 are long downloads — run as background terminal processes from the orchestrator, NOT inside subagents (subagent context dies before multi-GB downloads finish; poll via process tool).
- Task 6 depends on ALL of 1–4. Task 7 depends on 5–6. Serialize: 1 → (2 ∥ 3 ∥ 4 ∥ 5) → 6 → 7 → 8.
