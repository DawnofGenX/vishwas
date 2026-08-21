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

## EFFORT — spatial face/AIGI detector (deepfake_video) — PENDING (Task 1.3)

Checkpoints on disk (all three, sha256 sidecars written):
- `effort/chameleon/effort_chameleon.pth` 1,213,769,519 B (sha `2fc1b970…`) — PRIMARY per Decision #2
- `effort/ffpp/effort_ffpp.pth` 1,213,769,519 B (sha `8d86711f…`) — fallback
- `effort/genimage/effort_genimage.pth` 1,213,769,519 B (sha `7c32ceb4…`) — fallback

License: CC BY-NC 4.0 (YZY-stack/Effort-AIGI-Detection). Operator opt-in via
`VERISAFE_EFFORT_WEIGHTS`; evidence record will carry `"license": "CC-BY-NC-4.0"`.
No verification run yet — arch class still an honest stub
(`build()` raises `ArchNotImplementedError`). Filled in Task 1.3.

## HAVIC — holistic AV coherence (cross_modal) — PENDING (Task 1.4)

Checkpoints on disk:
- `havic/best_ft/best_ft_model.pth` 858,837,738 B (sha `7a0e3ddc…`) — primary
- `havic/pt200/pt_model.200.pth` 972,770,538 B (sha `a8c44dd5…`) — secondary fallback

No verification run yet — arch class still an honest stub. Expected SLOW tier
(15-min CPU cap per roadmap). Filled in Task 1.4.

---

## Test-suite state at this date

`PYTHONPATH=src python3 -m pytest tests/ -q` → **241 passed, 1 skipped, 0 failed**
(hermetic tree; the skip is the pre-existing GPU/optional-dep case). The one
failure that existed before this file was written —
`test_14_model_adapters.py::test_get_arch_lazy_registry` asserting the old
stub contract for aasist — was updated to reflect that aasist is now vendored
(environment-aware: real-torch tree expects `implemented=True`; hermetic
stub-torch tree expects honest `None`).
