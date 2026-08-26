# _spectra — vendored Spectra-AASIST3 network definition

- **Source:** lab260/Spectra-AASIST3 (HuggingFace), `model.py` fetched 2026-08-26.
- **License:** Apache-2.0 (per model card).
- **Baseline encoder:** `facebook/wav2vec2-xls-r-300m` (MIT). Weights (1.27 GB)
  are **NOT** committed to git; loaded from
  `/opt/verisafe/models/aasist3/wav2vec2-xls-r-300m/` (config.json +
  preprocessor_config.json are the small type metadata; pytorch_model.bin lives
  under /opt, see `src/vishwas/model_archs/aasist3.py` build()).
- **Status (2026-08-26): VENDORED + LOADED but NOT WIRED to serving.** See
  `/tmp/pkgB/PROOF.txt`: 1022/1022 keys match and weights apply, but the model
  does NOT separate the measured 240-clip ASVspoof2019-LA validation slice
  (AUC 0.54, posterior overlap). Rejected on the pre-committed acceptance bar.
  Re-proof against official ASVspoof2019-LA eval wavs before any wire-up.
- This directory is vendored verbatim (imports re-homed through
  `vishwas.model_archs.aasist3`); `model.py` is unchanged from upstream except
  import resolution.