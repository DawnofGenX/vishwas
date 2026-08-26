# Integrate malindard/phishing-checker-flask ML model into URL-Phishing Pipeline — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add the MIT-licensed classical-ML URL-phishing classifier from `malindard/phishing-checker-flask` as a **local, deterministic url_phishing evidence signal** in the Vishwas fusion, engaged specifically to resolve the "VirusTotal didn't confirm" case (VT low/zero detections or unavailable). The verdict stays a calibrated fusion of heuristics + ML + VT, not a single-gate flip.

**Architecture:** Vendor the XGBoost model + feature extractor into a new `url_phishml` capability that emits a `phishml.phishing_prob` CheckResult into the existing `url_phishing` fusion target. Default = **offline-first, gated-scrape**: use the 16 lexical features deterministically; the 7 network-scrape features (hyperlinks/title/domain_age/google_index/page_rank) are computed only if wall-clock budget remains, else emitted as honest known-gaps. Deps `xgboost` + `tldextract` installed into the existing docling-python serving tree.

**Tech Stack:** vendored `model/url_phishing_model.pkl` + `scaler.pkl` + `selected_features.pkl` (XGBoost, 23→ (offline subset) features), vendored `features/url_feature_extractor.py`, xgboost + tldextract (new deps), the existing pytest suite.

---

## Current context / assumptions

- Working from plan `2026-08-26_141500-overfit-fix-image-multimodal.md` (all 3 workstreams shipped, suite 370/8 green). HEAD: `817c66e`. Webhook live :2790 (12 deps).
- **Source verified (this session):** `malindard/phishing-checker-flask` (MIT, 0 stars, pushed 2025-08-26). Assets cloned to `/tmp/phish_ref/`.
  - `model/url_phishing_model.pkl` (2,620,498 B) + `scaler.pkl` (1,743 B) + `selected_features.pkl` (357 B, 23 feature names) + `model_info.pkl` (model_type=XGBoost, 23 features, best params gamma 0.130 / lr 0.018 / max_depth 19 / n_est 499 / subsample 0.343).
  - Prediction path (`api/api_url.py:271..300`): `scaler.transform(features_df) -> model.predict/predict_proba`, index1 = **phishing probability**.
  - Feature extractor: `features/url_feature_extractor.py` (25.6 KB, 40+ raw features; 16 deterministic + 7 network-scrape).
- **Dependency audit (verified):** serving tree (`PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python`) HAS pandas 3.0.5, numpy 2.5.2, joblib 1.5.3, sklearn 1.9.0, bs4 4.15.0, requests 2.34.2. **MISSING: xgboost, tldextract** (fail import). Both needed for full replication; `tldextract` is required even for the 16 lexical features (used to split domain/subdomain/path).
- Serving `url_phishing` fusion weights: phish_heuristics + vt_url_reputation (see `fusion.py`; VT greedy on quota).
- **User intent:** run the ML model "after VirusTotal if it doesn't confirm anything" — i.e. ML is the local fallback that adds evidence when VT is low/zero/unavailable.

## Proposed approach

Vendor the three `.pkl` files + `url_feature_extractor.py` under `src/vishwas/_urlphish_vendor/`. New `src/vishwas/capabilities/url_mal_ml.py` (+ `model_archs`? no — keep it a capability, not a learned arch needing device): loads the sklearn/XGBoost pipeline with the same silent-failure discipline as other gates, computes features, emits a `phishml.phishing_prob` CheckResult. Wire it into the `url_phishing` fusion `_SIGNAL_SOURCES` with a modest weight. Add deps. Recalibrate the url_phishing weights on a fresh small corpus of known-benign + known-phish URLs (reuse existing test fixtures; label source honest).

### Decision matrix (grill timed out → stated defaults, all changeable)
1. **Feature fidelity = OFFLINE-FIRST**: 16 lexical features deterministic; 7 network features → known-gap unless wall-clock budget remains. *Default; change to full-replica if the user wants max fidelity and accepts per-URL scrape latency.*
2. **Deps = install xgboost + tldextract into docling-python** serving tree (same isolated location as sklearn/torch). *Default.*
3. **Role in fusion = new evidence signal** feeding url_phishing, not a replacement. *Default.*
4. **Model trust bar = must separate a known-benign vs known-phish mini-corpus ≥0.75 AUC on OUR fixtures before it carries weight** — same honesty bar as every other gate. *Default; if it flunks, weight stays ~0 (documented) rather than shipping an unverified gate.*

---

## Step-by-step plan

### Task 1: Vendor model + feature extractor
**Files:**
- Create `src/vishwas/_urlphish_vendor/url_phishing_model.pkl` (cp from /tmp/phish_ref/model/)
- Create `src/vishwas/_urlphish_vendor/scaler.pkl`
- Create `src/vishwas/_urlphish_vendor/selected_features.pkl`
- Create `src/vishwas/_urlphish_vendor/url_feature_extractor.py` (cp; re-home imports so `from .. import` none needed — pure functions)
- Create `src/vishwas/_urlphish_vendor/PROVENANCE.md` (source repo+commit, MIT, file sizes, license, date)
**Steps:** `cp` the 4 files; verify the extractor imports standalone under the serving PYTHONPATH (it may need `tldextract`, `bs4`, `requests`). `READ ONLY` the extractor's function list (`HINTS`, helpers referenced by api_url.py import block).
**Pass:** all 4 files present; `python3 -c "import sys; sys.path.insert(0,'...'); from vishwas._urlphish_vendor.url_feature_extractor import ..."` for the 16 lexical functions succeeds.

### Task 2: Add deps
**Command:** `pip install --target /home/hermes/docling-python xgboost tldextract` (or the appropriate install path used for that tree).
**Verify:** under `PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python` both import; `xgboost.XGBClassifier` constructible.
**Pass:** both import; sklearn/joblib still import (no version clash — if xgboost pulls a newer numpy/scipy that breaks the tree, STOP and report, use a constrained pin).

### Task 3: Feature-extraction module (16 lexical, gated 7)
**Files:** Create `src/vishwas/capabilities/url_mal_ml.py`
**Objective:** a loader (joblib-loads model+scaler, silent-None on failure + reason) and a `_extract_features(url, budget_remaining_s) -> (features_dict_or_None, gaps:list[str])`:
- Always compute the 16 lexical features.
- For the 7 network features: if `budget_remaining_s > 15` and network available → compute (scrape hyperlinks/title, whois domain_age, google_index, page_rank), else set them as known-gaps.
- Build a 23-feature row in `selected_features` order; missing network features → 0.0 value + tagged in gaps.
**Pass:** for a benign URL, returns a full 23-col row; for `budget_remaining_s=0`, returns row with 7 gap features zeroed + gaps list.

### Task 4: Capability → CheckResult
**Files:** Modify `src/vishwas/capabilities/url_mal_ml.py`
**Objective:** `analyze(url, ctx)` emits:
- `CheckResult("url_phishml", "mid", status, {"phishing_prob": p, "n_lexical": 16, "n_network": k, "model_type": "xgboost"})` where status = "ok" when the model loaded and produced a finite prob, else "unavailable"/"failed" with reason in notes.
- Polarity documented in module header: **p = phishing probability** (index1), high = risky (matches positive-weight convention).
**Pass:** CLI smoke on a benign + a phish URL produces the check with finite prob both ways.

### Task 5: Fusion wiring
**Files:** Modify `src/vishwas/fusion.py` (url_phishing WEIGHTS + _SIGNAL_SOURCES + _EXPECTED_PROB_DET if present), `src/vishwas/capabilities/__init__.py` or orchestrator capability dispatch (to include url_mal_ml in the url_phishing stage).
**Objective:** add signal key `phishml.prob` → check `url_phishml`, signal `phishing_prob`, contributes_to_probs=True. Weight **chosen by Task 6** (start at 1.0, pending bar).
**Pass:** suite's url_phishing tests still green; the new check appears in a CLI url run.

### Task 6: Trust bar + weight calibration (the honesty gate)
**Files:** Create `tests/test_40_url_phishml.py`; `/tmp/phish_ml_status.md` evidence
**Steps:**
1. Build a mini-corpus from existing fixtures + a few held URLs: ≥8 known-benign (wikipedia, google, github, our own) + ≥8 known-phish (paypa1 typosquat style, suspicious TLDs, the existing test phishing fixtures under tests/fixtures/url if present, plus obvious ones). Label = source of truth (manual, documented per row).
2. Score all through `phishml.phishing_prob`.
3. **BAR: AUC ≥ 0.75 on that mini-corpus**, else the weight must stay ≤ 0.3 (evidence-only, honestly documented) until the model is revalidated on better data.
4. If it passes: grid `phishml.prob` weight {0.8,1.0,1.5} against phish_heuristics to keep benign NOT-DNU and phish ≥CAUTION (reuse the /tmp/vid_calib replay harness pattern reading a rows file).
**Pass:** AUC number + chosen weight + both-split table in `/tmp/phish_ml_status.md`.

### Task 7: Suite + live gate + docs + commit
1. Full hermetic suite: `env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS PYTHONPATH=src python3 -m pytest tests/ -q` → 374+ passed (incl new test_40), 8 skipped.
2. Live CLI: benign URL → not-DNU; phish URL with VT-low → gets the ML check + appropriate fused verdict; one URL where VT is unavailable → ML still runs (local).
3. Restart webhook, health ok.
4. Update `docs/research/VERIFY_SECURITY_STACK.md` + `MODEL_WEIGHTS_MANIFEST.json` (add `url-phishml-xgboost-MIT` gate) + `FUSION_FINAL_2026-08.md`; skill verisafe-operations ✓ weight-gates note.
5. Commit: `feat(url): vendored MIT xgboost phishing model as VT-fallback evidence signal (url_phishml)`.

---

## Files likely to change
- `src/vishwas/_urlphish_vendor/` (new: 3 pkl + extractor + PROVENANCE)
- `src/vishwas/capabilities/url_mal_ml.py` (new)
- `src/vishwas/fusion.py` (url_phishing weights + signal source)
- capability dispatch (add to url_phishing stage)
- `tests/test_40_url_phishml.py` (new) + any url_phishing expectation updates
- `docs/research/VERIFY_SECURITY_STACK.md`, `MODEL_WEIGHTS_MANIFEST.json`, `FUSION_FINAL_2026-08.md`
- `/home/hermes/docling-python` (new deps: xgboost, tldextract)

## Tests / validation
- Deps import checks (Task 2).
- 16-lexical row correctness smoke (Task 3).
- CheckResult emission smoke (Task 4).
- Suite green (Task 7).
- **AUC ≥ 0.75 trust bar on known-benign/known-phish mini-corpus; weight safe otherwise (Task 6).**
- Live: benign not-DNU; VT-unavailable still runs ML; webhook healthy.

## Risks / tradeoffs / open questions
- **Model is unproven** (0-star, no published eval) → the AUC bar is non-negotiable; if it fails, it ships as an evidence signal with low weight, honestly documented (do NOT fake a pass).
- **xgboost dep** could shift numpy/scipy in the serving tree → must verify sklearn/torch still import; if clash, pin xgboost or isolate into the new dep target.
- **Network-scrape features** add real latency + network calls per-URL if enabled → default OFF unless budget remains; keeps the webhook fast and deterministic.
- **Missing pandas in base python** but present in docling-python → the capability must run under the serving PYTHONPATH (it will, via run_vishwas.sh).
- **Open Q (user decision fetches later):** full-replica vs offline-first feature fidelity (default offline-first); whether to also serve email phishing detection (out of scope unless asked).