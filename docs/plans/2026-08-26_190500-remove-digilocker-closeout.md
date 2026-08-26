# Remove DigiLocker/API-Setu + Close Out Remaining Vishwas Work — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** (1) Cleanly remove the unimplementable external DigiLocker + API-Setu integrations from the gov_document path (they were blocked on registration; the local gov-verification path — QR-sha1, signature, official-web — does NOT depend on them and stays fully intact), and (2) plan and close the remaining Vishwas work items.

**Architecture:** Pure deletion of an external-API dead chapter. `_digilocker` and `_api_setu` methods, their fusion negative-weights, the `rag_cache` apisetu-catalog digest, and the `qr_verify/classifier` digilocker doc-type hint all get removed. No new capability replaces them — gov docs verify locally exactly as they do today. The remaining work items are sequenced as independent tracks.

**Tech Stack:** Python 3.12, pytest (hermetic 370 passed / 12 skipped baseline), existing vishwas package + webhook.

> **Defaults applied (grill timed out, user's "apply stated defaults" pref):**
> - Removal depth = **clean full removal** of BOTH DigiLocker AND API-Setu (no dead code left), NOT keep-and-disable, NOT DigiLocker-only.
> - Rest-of-work scope = the **full remaining list** in one plan, with user-blocked items (phone round-trip, sudo mv, HF token) sequenced as separate waiting tracks.

---

## Part 1 — Remove DigiLocker / API-Setu integration

### Task 1: Remove the two external-API methods
**Files:** Modify `src/vishwas/capabilities/gov_document.py`
**Steps:**
1. `analyze()` → delete the `out.extend(self._digilocker(...))` and `out.extend(self._api_setu(...))` calls (lines ~356-357).
2. Delete the `_digilocker` method (lines ~568-595) and `_api_setu` method (lines ~597-620).
3. Remove now-unused helpers `_find_docid` and `_dt_of` IF they are not referenced elsewhere (grep first; keep any still used by the official-web path).
4. Remove the `digilocker.gov.in` / `uidai.gov.in` entries from `OFFICIAL_DOMAINS` ONLY if they are not also used by the QR/official-web allowlist (grep `OFFICIAL_DOMAINS` usages — likely keep uidai/passportincome; remove only digilocker if unused).
**Pass:** `grep -rn "digilocker\|apisetu\|api_setu\|DigiLocker\|_find_docid\|_dt_of" src/` returns only the module docstring + any genuinely-shared allowlist entries. Setup code-path still imports; gov doc CLI run green.

### Task 2: Remove fusion weights + signal sources
**Files:** Modify `src/vishwas/fusion.py`
**Steps:**
1. Delete `"digilocker.verified": -5.0` and `"apisetu.records_found": -1.0` from `WEIGHTS["gov_document"]`.
2. Delete `"digilocker.verified": ("digilocker_verify", "dl_verified", …)` and `"apisetu.records_found": ("api_setu_lookup", "records_found", …)` from `_SIGNAL_SOURCES`.
3. Check the ReliabilityGate conflict rules (fusion.py ~line 734 references `digilocker_verify`) — remove that branch if it becomes unreachable.
**Pass:** `grep -n "digilocker\|apisetu" fusion.py` empty; `python3 -c "import vishwas.fusion"` ok.

### Task 3: Remove rag_cache apisetu catalog
**Files:** Modify `src/vishwas/rag_cache.py`
**Steps:**
1. Remove the `_CATALOG_GLOB = "apisetu_catalog_digest_*.json"` + the `newest_catalog_digest`/catalog-dir logic + the docstring references.
2. Keep the RAG retrieval itself (it serves gov-doc template deviation, which stays).
3. Remove any catalog `.json` recipes under a catalog dir if present (git rm).
**Pass:** `grep -rn "apisetu_catalog\|_CATALOG_GLOB" src/ tests/` empty; test_13_rag_cache still green.

### Task 4: qr_verify classifier doc-type hint
**Files:** Modify `src/vishwas/qr_verify/classifier.py`
**Steps:**
1. Remove the `digilocker_url` subtype + `_HOST_HINT_RE` digilocker branch (keep `udyam` if it routes elsewhere / is harmless).
2. Update the module docstring + `test_24_qr_verify.py` expectations if they assert `digilocker_url`.
**Pass:** `grep -n "digilocker" qr_verify/` empty; test_24 + test_25 (QR pipeline) green.

### Task 5: Tests + docs + suite
1. Re-run hermetic suite: `env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS PYTHONPATH=src python3 -m pytest tests/ -q` → **374+ passed (the removed gov sigs may reduce a count or two; re-count), 12 skipped**. Update tests that asserted the removed checks (search `digilocker_verify`, `api_setu_lookup`, `dl_verified`, `apisetu.records_found`).
2. Live gov-doc CLI smoke: a gov artifact (aadhaar/pan fixture) → still routes `gov_document`, verifies via QR/signature, NO digilocker/apisetu check in output.
3. Docs: update `docs/ARCHITECTURE.md` (if it names DigiLocker/API-Setu among gov evidence), `docs/research/VERIFY_SECURITY_STACK.md`, `FUSION_FINAL_2026-08.md`, `MODEL_WEIGHTS_MANIFEST.json` → mark digilocker/apisetu gates REMOVED. Skill `verisafe-operations` → remove the "DigiLocker/API Setu registration" blocker line.
4. Commit: `chore(govdoc): remove unimplementable DigiLocker + API-Setu external-API integration (local QR/signature/official-web verify unchanged)`.

---

## Part 2 — Remaining Vishwas work (full list, sequenced tracks)

> Definition of done for the overfit-related remainder: no real content falsely flagged HIGH, real content verified presentable, audio + cross-year measurement honest.

### Track A — Real audio-bearing real-video corpus (can do autonomously, needs operator media)
**Goal:** close the no-audio `UNVERIFIED` gap → real videos read MEDIUM/CAUTION not UNVERIFIED.
**Steps:**
1. Operator sends ≥4 real phone videos WITH speech to the WhatsApp number (or points to real clips with audio on disk). REQUIREMENT (user action).
2. Score through the live pipeline → build `rows_audio_reals` corpus of real-audio reals.
3. Recalibrate the no-audio spread-abort/effort-vs-frameheur carve-out against this corpus (currently the reliability gate returns UNVERIFIED when effort and frameheur conflict on a silent clip).
4. Bar: real-audio reals → CAUTION or trust, never UNVERIFIED-false; AI remains DNU; no false-HIGH.
**Blocked on:** operator (media). Everything after step 2 is autonomous.

### Track B — Operator-phone live WhatsApp round-trip (ultimate E2E confidence)
**Goal:** real phone ↔ real reply, not just HMAC-smoke.
**Steps:** send a real photo / real video / AI video from the operator's phone → confirm RISK lines correct + zero-retention holds.
**Blocked on:** operator phone action.

### Track C — ASVspoof2019-LA eval re-proof of Spectra-AASIST3 (audio channel fix)
**Goal:** honest audio-channel measurement on the model's Arena-verified domain.
**Steps:**
1. Download official ASVspoof2019-LA eval wavs (~2.7GB; network is UP).
2. Score through vendored Spectra-AASIST3 (`aasist3` family; weights already at /opt/verisafe/models/aasist3/).
3. If AUC ≥0.85 on eval → wire audio gate properly, raise its weight; else keep audio calibration-only (documented).
**Autonomous** (post-download). Delivers the honest audio answer.

### Track D — Path cleanup (sudo)
`sudo mv /opt/verisafe/models → /opt/vishwas/models` + update secrets env + `/health` dep check. **Blocked on user sudo.**

### Track E — HF token (optional extra)
Provision HF token → enable SSL-audio extra detector. **Blocked on user.**

---

## Files likely to change (Part 1)
- `src/vishwas/capabilities/gov_document.py`
- `src/vishwas/fusion.py`
- `src/vishwas/rag_cache.py`
- `src/vishwas/qr_verify/classifier.py`
- `tests/test_01_router.py`, `test_24_qr_verify.py`, `test_13_rag_cache.py`, `test_05_fusion.py` (only if they assert removed checks)
- docs: `ARCHITECTURE.md`, `VERIFY_SECURITY_STACK.md`, `FUSION_FINAL_2026-08.md`, `MODEL_WEIGHTS_MANIFEST.json`; skill `verisafe-operations`

## Tests / validation
- Full hermetic suite green (recount after removals; 12 skipped unchanged).
- Live gov-doc CLI: routes gov_document, QR+signature verify, NO digilocker/apisetu checks.
- Webhook restart → /health deps unchanged (12).
- Grep asserts zero remaining references in src + tests.

## Risks / tradeoffs / open questions
- **Fusion weights ripple:** `digilocker.verified −5.0` and `apisetu.records_found −1.0` were the ONLY clean-side gov negative signals. Removing them makes the gov target rely on `signature.valid −4.0` / `sig_object.present −1.5` / `qr.sha1_match −3.0` (already the real working signals) + positive flags. Verify a genuine gov doc STILL lands TRUST after removal (re-run the gov-trust test).
- **`rag_cache` bleed:** removing the apisetu catalog must not break `rag.template_deviation` (used by `fin.invalid_upi`/`rag.template_deviation` weights). Keep retrieval, remove only the catalog-digest feature.
- **Open Q (defaulted):** removal was chosen over keep-and-disable; if the operator later registers credentials, re-adding is a ~40-line restore from git history.
- **Track order:** Part 1 (removal) is fully autonomous and done first; Track A/C are autonomous-after-media/download; B/D/E are user-blocked and wait without leaking into the autonomous scope.