# Testing

How to run the suite, what the numbers mean, and why a bare run is not
enough.

## The canonical command

```bash
cd /home/hermes/vishwas
env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS \
    -u VISHWAS_EFFORT_WEIGHTS -u VISHWAS_IMAGE_FACE_WEIGHTS \
    -u VISHWAS_NYUAD_WEIGHTS -u VISHWAS_HAVIC_WEIGHTS \
    -u VISHWAS_FAKEMAMBA_WEIGHTS -u VISHWAS_DEMAMBA_WEIGHTS \
    -u VISHWAS_SSL_AUDIO_WEIGHTS \
  /usr/bin/python3 -m pytest tests/ -q
```

Three things about that command are not optional:

- **`/usr/bin/python3` (3.12), not `python3`.** The Hermes 3.14 tool
  interpreter ships only pip — it has no numpy of its own and gets its
  site-packages from `PYTHONPATH`. A 3.14 run therefore fails at
  *collection* for every test module rather than testing anything. 3.12 also
  has torch 2.4.1 and the numpy 2.5.2 that `pylibs` provides.
- **`env -u` on every weight variable.** If a previous shell exported them,
  `test_34` attempts a GPU load and fails with "weights present but
  load() failed" — which reads like product breakage but is env leakage.
- **No `PYTHONPATH=src`.** `tests/conftest.py` *appends* the repo's `src` to
  whatever `PYTHONPATH` already is. Setting `PYTHONPATH=src` **replaces** it,
  which displaces the interpreter's own site-packages and makes every
  third-party package vanish ("No module named 'pygments'").

The two QR tests (`test_24_qr_verify.py`, `test_25_qr_pipeline.py`) are
excluded by default in `pyproject.toml`: they need `qrcode` plus a
numpy/cv2 combination that agrees on the ABI, and otherwise fail at
collection. CI installs those deps and runs them explicitly.

## What the numbers mean

| run | collected | result |
|---|---|---|
| bare env, 3.12 | **544** | 530 passed / 6 skipped / 2 failed / 2 errors † |
| bare env, 3.14 | — | fails at collection (no numpy) |

† the 4 non-passes are **pre-existing**: the numpy-ABI `cv2` error and the
missing `torchvision` for SPAI. They are not caused by the code under test.

**The collected count depends on whether torch is installed, and that
matters more than the passed total.** Six test modules (`test_19_aasist_arch`,
`test_20_effort_arch`, `test_21_havic_arch`, `test_28_xlsrmamba_spec`,
`test_39_spai_arch`, `test_51_effnet_image_arch`) use a module-level
`try: import torch` guard. Without torch they degrade to a single skip
each; with torch they collect **60 real architecture tests**.

So "the suite is green" can mean either "the code is correct" or "the
detectors are not installed". The CI `test-full` job asserts the second
reading is impossible by requiring those modules to **collect ≥ 50 tests**.

## The two gates

- **test-bare** — the hermetic suite with every weight gate unset. Proves
  the cheap tiers work on a bare interpreter. Asserts a **floor** (≥ 465
  passed), not equality, so adding tests does not break the build but a
  regression that loses tests does.
- **test-full** — the suite with the learned detectors installed. Proves the
  torch-gated architecture modules actually run.

## Invariants

The `invariants` CI job checks four things that are easy to break and hard
to notice:

1. **The aasist3 helper path exists on disk.** This one assertion would
   have caught the `parents[2]` regression that made *both* audio gates
   unloadable on both branches — the symptom was real deepfake audio scoring
   LOW. The existing test monkeypatched the path to a temp file, so the real
   path was never exercised.
2. **Shell scripts have LF line endings.** `core.autocrlf=true` on a
   contributor's machine turns `#!/usr/bin/env bash` into `bash\r`, and cron
   execs the health watchdog directly — it then fails 100% of the time,
   silently.
3. **The launcher does not put `pylibs` on the `PYTHONPATH`.** Its
   `cryptography` v50 no longer ships the `lib` submodule pyOpenSSL imports,
   so Python resolves `lib` to a top-level namespace package and the whole
   `accelerate`/`transformers` import chain dies.
4. **No secret file is tracked.** `deploy/vishwas-secrets.env` holds the
   VirusTotal key and is gitignored.

## Running the detectors for real

The hermetic suite never loads a real checkpoint. To exercise the real
path, use the launcher (it selects the venv that can load them):

```bash
scripts/run_vishwas.sh cli --text "https://www.wikipedia.org"
scripts/run_vishwas.sh cli --file ./photo.jpg
scripts/run_vishwas.sh cli --file ./clip.mp4
scripts/run_vishwas.sh cli --file ./speech.wav
```

Verified end-to-end through the running webhook
(`POST /webhook/inbound`, HMAC-signed):

```
benign URL  -> RISK LEVEL: LOW
AI image    -> RISK LEVEL: MEDIUM   (SPAI 0.999)
real image  -> RISK LEVEL: LOW      (SPAI 0.0)
bonafide    -> RISK LEVEL: LOW      (aasist 0.0000)
```

## Measuring accuracy

`tools/build_crossmodal_corpus.py` and `tools/score_crossmodal.py` build
and score a video corpus that actually has an audio track — the published
87-clip corpus has none on 83 of 87 rows, so the cross-modal gates were
`known_gap` on 95% of the set the video weights were tuned on.

The corpus is **synthetic**: the audio is muxed from the official
ASVspoof2019-LA eval onto unrelated clips, so it is not synchronised with
the video. It measures whether the cross-modal gate is *wired* and degrades
honestly; it does **not** measure natural AV consistency.

The per-detector accuracy measured 2026-10-21 by re-scoring the raw media
is recorded in the README's "Measured results" section.
