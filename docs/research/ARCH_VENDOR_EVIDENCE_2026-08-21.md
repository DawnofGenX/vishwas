# Architecture-Vendor Evidence — 2026-08-21

Ad-hoc verification outputs backing the Phase 1 "learned tier" claims. Every
number below was produced by a throwaway script in a tempdir (deleted after
run) against the REAL provisioned weights under `/opt/verisafe/models/` — not
by the hermetic test suite. Re-run any check by re-creating the script; the
commands are recorded per family.

Environment: i5-8250U (15 W), CPU-only, torch 2.13.0+cu130 (CPU path) from
`/home/hermes/docling-python`, `PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:/home/hermes/verisafe/src`.
Thermal precheck: max zone 39 °C before work; no undervolt needed.

---

## AASIST — HABLA_WavLM_AASIST (deepfake_audio T2) — VENDORED ✅

**Checkpoint:** `/opt/verisafe/models/aasist/best_model.pth`
**Provenance:** DeepFense/HABLA_WavLM_AASIST_NoAug_Seed42 (arXiv:2110.01200 lineage)
**Arch class:** `src/verisafe/model_archs/aasist.py` (`HABLA_WavLM_AASIST`:
WavLM-Large 315 M front-end + HtrgGAT trunk), vendored backend in
`_aasist_backend.py` + `_wavlm/`.

### Fresh verify run (this file's date)

Command:
```
PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:/home/hermes/verisafe/src \
  python3 /tmp/verisafe-t12-verify/verify_aasist.py
```

| Check | Result |
|-------|--------|
| sha256 of on-disk weight | `febfc126a079716bf6957d776da937ba4fb6fc711fd9bb9ce640ded9c74d8d0f` |
| matches `.sha256` sidecar | ✅ true |
| matches manifest first-12 (`febfc126a079`) | ✅ true |
| `adapter.load()` through real seam (no monkeypatch) | `ArchModelWrapper` in **15.4 s** |
| `is_usable_model(model)` | **True** |
| full `adapter.run()` on `tests/fixtures/audio/smoke_5s.wav` | status `ok`, notes "model inference succeeded" |
| inference wall-time | **13.7 s** (cap 180 s; T2 stage budget 120 s → NOT SLOW) |
| spoof posterior | 0.997 (synthetic tone+noise fixture — value sanity only, not a verdict claim) |
| max thermal zone during run | 49 °C (normal tier; no SLOW marking per Decision #3) |
| overall | **PASS** |

Key-map provenance: 733/733 checkpoint keys matched the vendored skeleton
(`model_state` dict; first key `wavlm.encoder.pos_conv1d.weight`, last
`htrg_gat.classifier.bias`). Recorded in the module docstring of
`model_archs/aasist.py`.

Prior smoke (same day, earlier): load 15.7 s, single-crop inference 15.3 s,
full 3-crop T2 stage ~23 s inside the default 300 s job budget (~8 % share).
Both runs agree within noise.

**Status:** deepfake_audio T2 now has a live learned scoring path behind the
existing availability gate; heuristic fallback untouched and still active when
the env var is unset or the arch is unavailable.

---

## EFFORT — spatial face/AIGI detector (deepfake_video T2) — VENDORED ✅

**Checkpoint (primary):** `/opt/verisafe/models/effort/chameleon/effort_chameleon.pth`
**Provenance:** YZY-stack/Effort-AIGI-Detection, chameleon checkpoint (Decision #2 primary)
**Arch class:** `src/verisafe/model_archs/effort.py` (`EffortSpec` / `_EffortNet`:
CLIP-style ViT-L/14, 303.4 M params, 24 encoder layers, OrthAlign rank-1
residuals on every self-attn projection). All three checkpoints probed to be
architecturally identical (681 keys each; same layer-0 and non-layer counts),
so one spec covers chameleon/ffpp/genimage.

### Fresh verify run (this file's date)

Command:
```
PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:/home/hermes/verisafe/src \
  python3 /tmp/verisafe-t13/verify_effort.py
```

| Check | Result |
|-------|--------|
| sha256 of on-disk weight | `2fc1b97014b456d5…` |
| matches `.sha256` sidecar | ✅ true |
| skeleton build (random init) | ok, **303.4 M params**, 2.55 s |
| torch.load (1.21 GB) | 1.1 s, 681 keys |
| `apply_state` key coverage | **ok=True, frac_missing=0.0, frac_unexpected=0.0** (681/681) |
| forward pass (synthetic 224×224) | fake posterior 0.48–0.58 (value sanity only) |
| inference wall-time | **~1.7 s** per frame (cap 180 s → NOT SLOW) |
| max thermal zone during run | 55 °C peak (normal tier; no SLOW marking per Decision #3) |
| overall | **PASS** |

### Label-order confirmation (assumption E4)

Known-REAL photos through the full cv2→resize→score path:

| Input | fake posterior |
|-------|---------------|
| `/usr/share/backgrounds/Fuji_san_by_amaral.png` (real photo) | **0.2858** |
| `/usr/share/backgrounds/Clouds_by_Tibor_Mokanszki.jpg` (real photo) | **0.3509** |

Both well below 0.5 → head order `[real, fake]` confirmed; `score()` returns
the fake-class probability as documented.

**Full-seam smoke (2026-08-21):** `Adapter.load()` on the real checkpoint →
`ArchModelWrapper` (8.0 s), `is_usable_model` True, then `adapter.run()` on a
real photo through the exact `deepfake_video._effort` chain
(`_load_model` + `_infer`: preprocess → `_call_model` → extract_prob):
status `ok`, `prob_deepfake=0.286`, wrapper predict 1.36 s/frame — identical
to the direct label-order number (0.2858), confirming the capability wiring
is lossless.

**Test_20 hardening pass (2026-08-21, real-torch run):** running
`tests/test_20_effort_arch.py` under real torch (not just the hermetic stub
tree) exposed three latent bugs, all fixed:
1. `_OrthAlignLinear` init used `torch.empty` for bias/residuals → NaN in
   random-init forward. Now `torch.zeros` + `nn.init.normal_(weight_main)`.
2. `EffortSpec.apply_state` called `sd.items()` before the base None/empty
   guard → AttributeError on empty payload. Now delegates to base first.
3. `score()` accepted str/bytes and crashed inside `np.asarray`. Now raises
   TypeError early. Also `_EffortNet.class_embedding` → `torch.zeros(dim)`
   (was uninitialised memory).
Result: **12/12 passed** under real torch; hermetic suite still 241 passed /
2 skipped (test_20 skips there by design — needs real torch).

Key-map provenance: verbatim 681-key map recorded in the module docstring of
`model_archs/effort.py`, including the two reconstruction details that a naive
CLIP-ViT-L would get wrong: `patch_embedding` has **no bias** and
`position_embedding` is an `nn.Embedding(257,1024)` (key
`position_embedding.weight`); the `pre_layrnorm` typo is preserved verbatim;
the `module.` DataParallel prefix is stripped in `apply_state`.

License: CC BY-NC 4.0 — operator opt-in via `VERISAFE_EFFORT_WEIGHTS`;
evidence record carries `"license": "CC-BY-NC-4.0"`.

**Status:** deepfake_video T2 now has a live learned scoring path behind the
existing availability gate; heuristic fallback untouched and still active when
the env var is unset or the arch is unavailable.

## HAVIC — holistic AV coherence (cross_modal) — VENDORED ✅

**Checkpoint (primary):** `/opt/verisafe/models/havic/best_ft/best_ft_model.pth`
**Provenance:** JielunPeng/HAVIC, arXiv:2603.23960 — **MIT licensed** (no opt-in
gate needed). `best_ft` = HAVIC_FT finetune payload (456 tensors); `pt200`
secondary entry in `CHECKPOINT_CHAIN` is a pretrain payload WITHOUT classifier
heads — apply_state honestly returns False for it (documented limitation; not
a working scoring fallback).
**Arch class:** `src/verisafe/model_archs/havic.py` (`HavicArch`, implemented)
over vendored backend package `model_archs/_havic/` (copy-adapted reference
modules + minimal `_timm_shim.py`; timm absent from this tree). One upstream
bug fixed in the vendored copy: forward passed `use_mask=False` to encoders
whose signature says `use_hierarchical`.
**Wiring:** `cross_modal._load_havic()` routes through
`resolve("VERISAFE_HAVIC_WEIGHTS") -> adapter.load() -> is_usable_model()`;
preprocessing = numpy kaldi-fbank port (validated vs real torchaudio: max abs
diff ≤5e-4 across sine/speech/noise/short inputs) + 16-frame [0,1] tensor.

### Fresh verify run (this file's date)

Command:
```
VERISAFE_HAVIC_WEIGHTS=/opt/verisafe/models/havic/best_ft/best_ft_model.pth \
PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:/home/hermes/verisafe/src \
  timeout 900 python3 /tmp/verisafe-t14-smoke/verify_havic.py <clip> <workdir>
```

| Check | Result |
|-------|--------|
| sha256 of on-disk weight | `7a0e3ddc6effd6813c81d915eb2ea57e7dc7d4711cb39798680ff477de2ca19d` |
| matches manifest first-12 (`7a0e3ddc…`) | ✅ true |
| `adapter.load()` through real seam | `ArchModelWrapper` in **8.6 s** |
| `is_usable_model(model)` | **True** |
| preprocess (production helpers) | 0.3 s → audio `(1024,128)` fbank + visual `(3,16,224,224)` |
| inference wall-time (4 s clip, 16 frames) | **5.1 s** (cap 900 s → **NOT SLOW**) |
| inconsistency posterior | 0.9986 (synthetic testsrc+sine fixture — value sanity only, not a verdict claim) |
| max thermal zone during run | 48 °C (normal tier; no SLOW marking per Decision #3) |
| overall | **PASS** |

Preprocessing deviations (documented honestly): whole frames instead of face
crops (no face cropper vendored); whole-clip audio mean-removal rather than
per-window alignment; last-frame repeat when <16 frames extract; polarity
(assumption H5) unverified — smoke value is sanity-only.

**Status:** cross_modal now has a live learned consistency path behind the
existing availability gate; heuristic AV probe untouched and still active when
the env var is unset or the arch is unavailable.

---

## Test-suite state at this date

`PYTHONPATH=src python3 -m pytest tests/ -q` → **252 passed, 3 skipped, 0 failed**
(hermetic tree; skips are the pre-existing GPU/optional-dep cases). The one
failure that existed before this file was written —
`test_14_model_adapters.py::test_get_arch_lazy_registry` asserting the old
stub contract for aasist — was updated to reflect that aasist is now vendored
(environment-aware: real-torch tree expects `implemented=True`; hermetic
stub-torch tree expects honest `None`). After EFFORT vendoring (Task 1.3) the
same test moved effort into the vendored group, and after HAVIC vendoring
(Task 1.4) havic joined it; `tests/test_19_aasist_arch.py`,
`tests/test_20_effort_arch.py` and `tests/test_21_havic_arch.py` pin each
spec's contract hermetically (same stub-torch pattern). Real-torch spot:
test_14 + test_21 → 50 passed.

---

## Phase-1 proof bar — E2E CLI evidence (2026-08-21)

Per the roadmap proof bar ("suite green is not enough"): full CLI runs through
`scripts/run_verisafe.sh cli` with all three weight env vars provisioned,
showing learned scores in the per-check evidence of real pipeline output.

### Run 1 — audio fixture (`tests/fixtures/audio/smoke_5s.wav`, job_3e9cd7bc63c8)

```
[heavy] aasist_detector   ok  {"prob_deepfake": 0.997, "n_crops_scored": 3, "max_prob": 0.997}
fusion: {"target": "deepfake_audio", "raw_score": 0.7693, ..., "usable_checks": 4}
wall_s: 42.12   purged: true
```
Learned AASIST tier scored the real weights E2E (0.997 on a synthetic tone
fixture — sanity only). Heuristic offline features ran alongside (0.2);
fusion disagreement gate correctly flagged the divergence rather than hiding it.

### Run 2 — AV clip (`/tmp/p1proof/clip_av.mp4`, 4s testsrc+sine, job_71437ed82ce0)

```
[heavy] effort_face_forensics   ok  {"prob_deepfake": 0.382, "n_frames_scored": 8, "max_prob": 0.426}
[heavy] cross_modal_av          ok  {"av_correlation": 0.132, "alignment_class": "decorrelated"}
[heavy] havic_crossmodal_model  ok  {"prob_inconsistent": 0.999}
fusion: {"target": "deepfake_video", "raw_score": 0.2388, "calibrated": 0.8073,
         "gate": {"ok": true}, "usable_checks": 6}
stage_timings_s: DeepfakeVideoCapability 19.43, CrossModalCapability 11.72
wall_s: 31.15   purged: true
```
All three learned families appear in ONE pipeline run: EFFORT frame scoring,
HAVIC cross-modal consistency, plus the heuristic AV probe running alongside.
Fusion gate passed; zero-retention held (purged=true).

**Phase-1 verdict: PROOF BAR MET** — every gated family (aasist / effort /
havic) loads REAL weights through the production seam and produces scores in
the evidence JSON of an end-to-end CLI run.
