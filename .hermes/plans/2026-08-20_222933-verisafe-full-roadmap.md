# VeriSafe Full-Scale Roadmap

> **For Hermes:** Use subagent-driven-development skill to implement this plan phase-by-phase, task-by-task. Each task is independently verifiable; commit after every task.

**Goal:** Take VeriSafe from current state (205 tests green, heuristics-live + learned-weights-on-disk-but-unusable, 2 of 7 languages translated) to a fully operational WhatsApp verification platform on this laptop: learned detectors active, non-blocking heavy results, all 7 languages, deployment artefacts shipped, and a re-baselined performance/security posture.

**Architecture:** Deterministic router → capability tiers (T0 stdlib → T1 mid → T2 heavy) → progressive short-circuit → template-first reports over OpenWA/CLI. Zero retention via JobQuarantine + purger. All new work slots into these seams; nothing here changes routing determinism or the zero-cloud constraint.

**Tech Stack:** Python 3.12, torch (CPU-only), numpy, Pillow, FFmpeg (thread-capped via `VERISAFE_FFMPEG_THREADS`), gpg, ClamAV, YARA-X, Docling (isolated tree via `scripts/run_verisafe.sh`), OpenWA v0.21.0, pytest. Host: i5-8250U thermal-sensitive — heavy smokes run sequentially, one at a time.

---

## Verified current state (2026-08-20, audited this session)

| Area | State | Evidence |
|---|---|---|
| Tests | **205 passed**, ~12s | `pytest tests/ -q` run today |
| Git | Repo exists, **zero commits**, everything untracked | `git log` fails on master |
| Model weights | aasist/effort×3/havic×2 on disk at `/opt/verisafe/models/` (9.26 GB, sha256 sidecars) but adapters report unusable — **arch classes never vendored** | `is_usable_model()` rejects raw state-dicts |
| EFFORT license | **CC BY-NC 4.0** (README badge; no LICENSE file in repo) | fetched live this session |
| i18n | `_SUPPORTED = (en, hi, ta, te, ml, kn, bn)`; only **en + hi** actually have strings; other 5 keys exist but fall back to English. `load_custom_strings()` overlay mechanism present and unused | `src/verisafe/i18n.py:16,71,105` |
| Service surface | `ThreadingHTTPServer`: `GET /health`, `POST` webhook with signature verify | `app.py:9,184,199` |
| Quarantine | TTL sweeper function exists (`scan_stale_quarantines`, default 7200s) but **nothing invokes it periodically**; purge_audit.log healthy (59 entries, 0 failures, 0 residual) | `quarantine.py:20,131`; `logs/purge_audit.log` |
| Empty dirs | `src/verisafe/analysis/`, `channels/`, `fusion/` are EMPTY next to live modules `channels.py`, `fusion.py` — import resolution currently correct (`.py` wins; `analysis` imports as bare namespace pkg) but these are shadowing landmines + dead scaffold | import probe ran this session |
| Deploy | `deploy/` empty despite DEPLOYMENT.md describing OpenWA setup | `ls deploy/` |
| Trust store | `assets/ca_truststore/` = README only, no anchor certs | inspected today |
| Credentials | VT key, DigiLocker e-KYC, API Setu consumer, HF token, IMAGE_FACE weights — all absent (external/ops-gated) | GAPS_AND_ENABLEMENT.md |
| Decisions locked (this session) | #1 EFFORT: ship CC-BY-NC with prominent notice + env opt-in; #2 chameleon checkpoint primary, others lazy fallback; #3 SLOW-tier = non-blocking follow-up message | see prior plan `2026-08-20_212814` |

**Companion plan:** `.hermes/plans/2026-08-20_212814-verisafe-remaining-work.md` holds the earlier gap list; this roadmap supersedes it and absorbs its Task Groups A/C/D into Phases 0/4/2 below.

---

# PHASE 0 — Hygiene & de-risking (est. 1 h)

Purpose: make the tree versioned and eliminate latent import hazards before any new code lands.

### Task 0.1: Baseline git commit
1. Create `.gitignore`:
   ```
   .delegation/
   .test-quarantine/
   logs/
   __pycache__/
   *.pyc
   *.egg-info/
   output/renders/
   ```
2. `git add .gitignore src/ tests/ scripts/ docs/ fusion/ deploy/ README.md && git commit -m "chore: baseline import of VeriSafe (P0-P10 complete, pre-roadmap)"`
3. Verify: `git log --oneline | wc -l` → 1; `git status --short` shows only ignored/untracked noise.

### Task 0.2: Remove empty shadowing dir-scaffold
1. Confirm all three dirs are truly empty: `find src/verisafe/{analysis,channels,fusion} -type f` → nothing (already confirmed this session; re-confirm as guard).
2. Also grep the whole tree for any `from verisafe.analysis import` / `verisafe.channels.` submodule-style imports that would break if namespaces shifted — expect none:
   `grep -rn "verisafe\.analysis\|verisafe\.channels\.\|verisafe\.fusion\." src/ tests/ scripts/`
3. `rmdir src/verisafe/analysis src/verisafe/channels src/verisafe/fusion` (use rmdir, not rm — fails loudly if anything was added since audit).
4. Add regression test `tests/test_18_imports.py`:
   ```python
   def test_module_vs_dir_no_shadowing():
       import verisafe.channels as c
       import verisafe.fusion as f
       assert c.__file__.endswith("channels.py")
       assert f.__file__.endswith("fusion.py")
   def test_analysis_not_a_package_landmine():
       try:
           import verisafe.analysis
       except Exception:
           pass
       else:
           import verisafe.analysis as a
           assert getattr(a, "__file__", None) is None  # bare namespace at worst, never shadowing a module
   ```
5. Run `pytest tests/test_18_imports.py -q` → 2 passed. Commit: `chore: remove empty analysis/channels/fusion dir-scaffold; lock import-resolution test`.

### Task 0.3: Update the two plan docs to match reality
- In `docs/GAPS_AND_ENABLEMENT.md`, mark model-weights row with arch-vendor status = "NOT yet" (currently implies more than true), and add pointer to this roadmap. Commit.

**Phase 0 proof bar:** fresh-clone dry run in tempdir (`git clone` the repo to `/tmp/vs-p0-check`, run `PYTHONPATH=... python3 -m pytest tests/ -q` there) must give 205+ passed, 0 failed. Clean up tempdir.

---

# PHASE 1 — Learned detectors online (the core value unlock; est. 2–4 days incl. smokes)

Depends on: Phase 0. Order inside phase is fixed: B0 hooks → AASIST → EFFORT → HAVIC (smoke cost grows with each).

### Task 1.1 (B0): Arch-loading plumbing in adapter registry
**Files:** Modify `src/verisafe/model_adapters.py` (~line 149 `_default_load`, line 175 `is_usable_model`); create `src/verisafe/model_archs/__init__.py`, `src/verisafe/model_archs/base.py`.
1. Test first (TDD) — extend `tests/test_14_model_adapters.py`:
   ```python
   def test_adapter_prefers_vendored_arch_over_raw_dict(tmp_path, monkeypatch):
       # write fake state dict pth-shaped json? NO — keep hermetic: monkeypatch
       # torch.load equivalent via an injectable loader seam:
       ...  # stub arch module exposing build(); assert Adapter.run() returns usable score obj
   ```
   Run → FAIL (no hook). Implement minimal hook → PASS.
2. Implementation contract:
   ```python
   # model_archs/base.py
   class ArchSpec:
       name: str                       # "aasist" | "effort" | "havic"
       weight_env: str                 # e.g. VERISAFE_AASIST_WEIGHTS
       def build(self) -> Any: ...     # constructs network skeleton
       def apply_state(self, model, sd: dict) -> bool: ...  # strict=False; False = shape mismatch
       def score(self, model, x: Any) -> float: ...        # calibrated 0..1
   ```
   Adapter load order: (a) env var unset/missing → None (unchanged); (b) env set, arch module for family importable AND `apply_state` True → callable ready; (c) otherwise → return None with reason string recording `weight file loaded but architecture unavailable`. Existing heuristic fallbacks untouched either way.
3. Manifest: add `"arch": {family: {"vendored": bool}}` block to `docs/research/MODEL_WEIGHTS_MANIFEST.json`; keep it hand-updated after each task (YAGNI: no auto-writer yet).
4. Commit: `feat(adapters): arch-class loading seam for gated model families`.

### Task 1.2 (B1): AASIST audio anti-spoofing
**Objective:** deepfake_audio T2 moves from heuristics to learned scoring with fallback intact.
1. Probe: print `sorted(torch.load('/opt/verisafe/models/aasist/best_model.pth')['model_state'])` + shapes of first/last tensors (read-only script in tempdir; cleaned up). Record findings in the module docstring as the key-map provenance.
2. `src/verisafe/model_archs/aasist.py`: mel front-end (reuse existing `_mel_aasist_preprocess`) + spectro-temporal graph-attention trunk matching the probed key names (port algorithm per Jung et al. ASVspoof lineage arXiv:2110.01200 reference impls; attribution header). `score()` = softmax posterior of spoof class → float.
3. Wire into `deepfake_audio.py` T2 slot behind the existing availability gate; on any exception → CheckResult(status="failed", notes=...) exactly like other stages (failure-matrix pattern, no new error paths).
4. **Live smoke** (thermal-safe, sequential): `source scripts/provision_weight_env.sh` + 5-second fixture WAV from `tests/fixtures/`, wall cap 120s, while `tail -f ~/.hermes/logs/thermal_power_monitor.log` in second terminal window. Record actual ms in `docs/PERFORMANCE.md`. If > 120s: mark stage SLOW (Phase 2.1 handles delivery), do NOT delete.
5. Docs: GAPS row update; commit: `feat(audio): vendor AASIST arch — learned tier live`.

### Task 1.3 (B2): EFFORT spatial face/AIGI detector — CC-BY-NC, opt-in
**Objective:** deepfake_video frame scoring upgrades when operator explicitly opts in.
1. Probe all 3 checkpoints' key sets (assert one arch class covers them; chameleon primary per Decision #2; ffpp/genimage only load as fallback when chameleon file missing/unreadable — logic lives in the arch loader: `primary → fallback chain`).
2. `src/verisafe/model_archs/effort.py`: OrthAlign subspace-decomposition head over the backbone named by the weights (probe reveals whether backbone weights are embedded or assumed frozen-CLIP — record which; if a pretrained CLIP/ViT backbone is required and none is on disk, document that variant stays off and why — honesty rule: don't half-load).
   **Module docstring MUST contain:** `License: CC BY-NC 4.0 (YZY-stack/Effort-AIGI-Detection). Port of method, not copy of repo code. Non-commercial use only.`
3. Opt-in enforcement: env var export = enabled (existing model). When enabled, the evidence record carries `"license": "CC-BY-NC-4.0"` (add to CheckResult signals) and `report.py` adds a one-line plain-language note: "This check uses a research tool licensed for non-commercial use."
4. Smoke: single-frame extract from a `tests/fixtures` mp4 via existing ffmpeg helpers (thread-capped), 180s cap, SLOW marking available per Decision #3.
5. Docs: NC-license notice sections in `GAPS_AND_ENABLEMENT.md` (new top-level "⚠️ Licensing" section) and `USER_GUIDE.md`; manifest `arch.effort.vendored=true, license=CC-BY-NC-4.0`. Commit: `feat(video): vendor EFFORT arch (CC-BY-NC, operator opt-in, chameleon primary)`.

### Task 1.4 (B3): HAVIC holistic AV coherence
**Objective:** cross_modal gains learned consistency scoring (largest/slowest model; expect SLOW tier).
1. Probe both checkpoints (`best_ft_model.pth` primary, `pt_model.200.pth` secondary fallback) — determine input conventions (both modalities? feature dims?).
2. `src/verisafe/model_archs/havic.py` per arXiv:2603.23960 design; `score()` = inconsistency probability.
3. Wire into `cross_modal.py`; same failure-matrix semantics.
4. Smoke: one short AV clip (≤5s, existing fixture or 5s cut of one), **hard wall cap 15 min on CPU** with thermal-watch; outcome recorded as SLOW-tier timing in PERFORMANCE.md either way. If it never completes within cap: stage stays `unavailable` with honest note; do not force.
5. Docs + commit: `feat(cross-modal): vendor HAVIC arch; SLOW-tier AV coherence`.

### Phase 1 proof bar (your standard — suite green is not enough)
Per family, ad-hoc script in tempdir (deleted after): load REAL weights from `/opt/verisafe/models/` → `is_usable_model(...)` is True → one real inference → print measured time + PASS/FAIL lines. Then full `pytest tests/ -q` (expect ≥208 passed, 0 failed). Then a fresh `run_verisafe.sh cli --file <audio-fixture>` E2E showing the new learned score appearing in the evidence JSON of the report. Record all outputs in a dated evidence file under `docs/research/` (e.g. `ARCH_VENDOR_EVIDENCE_2026-08-XX.md`) so future sessions can trust them without re-running.

---

# PHASE 2 — Delivery, ops & observability (est. 1–2 days)

Depends on: Phase 1 (follow-ups need the SLOW stages to exist). Some tasks (2.2, 2.3) are independent and could start earlier if desired.

### Task 2.1: Non-blocking heavy results (implements Decision #3)
**Files:** Modify `src/verisafe/orchestrator.py` (stage loop ~lines 190-280; `stage_timings_s` site), `src/verisafe/report.py` (+ `pending_heavy` field), `src/verisafe/channels.py` (`MessageProcessor.deliver` path).
1. Contract: when a T2 learned stage exceeds its stage budget but keeps running in the background pool, the fast verdict ships immediately containing `"pending_heavy": [{"cap": "...", "expected_s": N}]` and a plain-language line: "I'll send you an update when my deeper check finishes." The follow-up reply is **template-composed only** (deterministic verdict wording from i18n + confidence number), never LLM-generated.
2. Test: `tests/test_10_performance.py` gets a case with a monkeypatched slow stage asserting (a) first reply contains pending_heavy, (b) follow-up fires ≤ stage cap + margin, (c) follow-up text matches the exact template for lang=en and lang=hi.
3. OpenWA delivery of the follow-up reuses the same chat_id/session captured at job start (persist on JobContext); CLI simulator prints the follow-up inline (no-op transport works identically).
4. Failure paths: if follow-up's result arrives after user session ended → drop silently, log to stdout (no retry queue — YAGNI).
5. Commit: `feat(orchestrator): non-blocking heavy-stage follow-up replies (deterministic templates)`.

### Task 2.2: Stale-quarantine sweeper actually runs
**Files:** New `scripts/stale_sweep.sh` (invokes `scan_stale_quarantines()` once, prints removed count); register a Hermes cron `no_agent=True` job, every 15 min, silent-on-empty (watchdog pattern: script prints nothing unless it removed something or errored).
1. Script body: cd project root, `exec bash scripts/run_verisafe.sh`? NO — standalone: `cd $(repo) && PYTHONPATH=src python3 -c "from verisafe.quarantine import scan_stale_quarantines as s; r=s(); print(len(r)) if r else None"` (empty output = silent tick by cron design).
2. Verify: plant a fake old quarantine dir (manifest ts − 3 h), run script → gone + audit line appended to purge_audit.log. Plant fresh one (ts now) → survives. Clean up.
3. Commit: `ops: periodic stale-quarantine sweep (cron, silent watchdog)`.

### Task 2.3: Ops visibility for a long-lived webhook process
**Files:** Modify `src/verisafe/app.py` (`/health` handler).
1. Extend `GET /health` response: `{status:"ok", uptime_s, jobs_total, jobs_ok, jobs_failed, quarantines_open, deps:{...detect_available_deps() summary...}}` — counts from a tiny thread-safe counter incremented in MessageProcessor.process outcome path (no DB; counters reset on restart — documented).
2. Test: `tests/test_09_redteam.py` or a small new `tests/test_19_ops.py` hitting the handler via direct call (no socket binding needed — construct Handler instance against a dummy environ pattern used by BaseHTTPRequestHandler; reuse how existing tests exercise app pieces if any).
3. Write `docs/OPERATIONS.md`: service lifecycle (start via `scripts/run_verisafe.sh webhook --port 2785`, stop = kill; supervisor recommendation = systemd unit file included as `deploy/verisafe.service.example`), daily checks (curl /health, tail purge_audit, tail thermal log), weekly: `freshclam`, RAG cache rebuild cadence (`scripts/build_rag_cache.py`), model weights re-provision procedure, escalation table (what CRITICAL thermal alert means + where the watchdog log lives).
4. Commit: `ops: rich /health, OPERATIONS.md, systemd example`.

### Task 2.4: RAG cache freshness check
**Files:** Modify `src/verisafe/rag_cache.py`; add test to `tests/test_13_rag_cache.py`.
1. Cache dir stores `BUILT_AT` + source digests (builder already writes version metadata — confirm and extend if needed). At load: if age > 14 days OR digest mismatch vs latest catalog file in `docs/research/data/`, mark entries `stale` → consumers drop their confidence contribution silently (existing degradation semantics) and the report's `evidence_missing` line lists "gov-template-cache-stale".
2. Test: backdate a fixture cache → stale flag observed. Commit: `feat(rag): freshness/gating on template cache`.

---

# PHASE 3 — Language completion (est. 2–3 days, largely review-gated)

Independent of Phases 1–2; parallelizable. Target: all 7 in `_SUPPORTED` deliver real verdict sentences for elderly non-technical users — the product's core audience.

### Task 3.1: Translation workflow infrastructure
1. Dump current English corpus: every key in `i18n._DEFAULTS` (greeting, analyzing, progress_url/file/media, verdict_trust/caution/do_not_use/unable, confidence_line, advice_avoid_links, evidence_missing, credits) → `docs/i18n/en.md` (machine-generated script `scripts/i18n_export.py`).
2. Per language X in (ta, te, ml, kn, bn): produce `docs/i18n/X.draft.md` via a **dedicated translation subagent** (one per language, parallel dispatch OK — they're independent), instructed: translate for a 65-year-old Hindi-region elder; no tech jargon; preserve `%(conf)s` placeholders verbatim; mark uncertain lines with `[?]`.
3. Human/native review gate: drafts go to you (DawnofGenX) for sign-off or correction before merge — **do not merge machine translations as final**; the existing `load_custom_strings()` overlay is the merge vehicle.
4. Merge: reviewed strings land directly in `i18n._DEFAULTS[X]` (keep the overlay mechanism for future corrections, but the committed source of truth is the module). Update module docstring status line ("hi best-effort" → per-language state: reviewed / draft / fallback).
5. Test: `tests/test_08_i18n_report.py` extended — for ALL 7 langs: every key renders, `%(conf)s` interpolates, no placeholder leaked, no empty strings; `t(key, "ta")` must differ from English (catches silent fallback regressions) — but exempt keys still in draft (assert they fall back to en *by design* per a declared `reviewed_languages` set).
6. Commits per language as reviews land: `i18n: verified Tamil strings (native-reviewed)` etc.

### Task 3.2: Follow-up-message strings (pairs with 2.1)
Add `heavy_pending_notice` + `heavy_followup` keys to all 7 languages through the same pipeline (these are new user-facing surfaces, so they ride the review gate too).

**Phase 3 proof bar:** render all 7 verdicts × 4 outcomes in a printed matrix (script in tempdir) — visual eyeball pass, plus the placeholder-leak test above. Anything still `[?]` after your review stays on the fallback path and is listed as such in GAPS doc — never presented as localized.

---

# PHASE 4 — Enablement & deployment (est. 1–2 days + external waits)

### Task 4.1: VirusTotal live verification (needs YOUR key)
Client code + hermetic tests exist (`vt_client.py`, `test_17`). Once you set `VERISAFE_VT_API_KEY`: run ad-hoc live probe on 1 known-clean + 1 known-malicious hash/domain, capture real responses into `docs/research/VT_LIVE_EVIDENCE.md`, and update GAPS row ⚠️→✅. Until then: blocked-on-credentials, tracked not scheduled.

### Task 4.2: DigiLocker / API Setu registration checklist (operator steps, documented)
In `docs/OPERATIONS.md` new section: exact registration paths (partners.apisetu.gov.in client-credentials flow; DigiLocker partner/OAuth program contact page), which env vars each unlocks, expected verification improvements per doc class (pull mapping from `docs/research/INDIA_GOV_VERIFICATION.md` §1-2). We cannot perform registrations; we make them a 30-minute checklist.

### Task 4.3: CA trust-store seeding (absorbs old Task Group C)
Identify anchor public certs actually referenced by PAdES fixtures in `tests/fixtures/` (probe signer issuer chains offline via our own `pades_check` parsers — no network); save `<issuer>.der` + provenance; if no suitable published anchors exist for our fixture issuers, generate a self-issued test CA for the trusted-chain test and document that production anchoring is operator-supplied. One new `test_12_pades.py` case → `chain=="trusted"` path exercised. Commit: `feat(pades): trust-store anchors + anchored-chain test`.

### Task 4.4: Deployment artefacts (absorbs old Task Group D)
1. `deploy/docker-compose.yml` — pinned `rmyndharis/OpenWA:v0.21.0` + verisafe service (entrypoint `scripts/run_verisafe.sh webhook`, env wiring incl. weight-provision step calling `scripts/provision_weight_env.sh --quiet` at container start, volumes for `/opt/verisafe/models`, port 2785, healthcheck hitting `/health`).
2. `deploy/webhook.example.json` — OpenWA webhook config pointing at the verisafe route with `X-OpenWA-Signature` secret placeholder, matching `parse_openwa_webhook` expectations.
3. `deploy/verisafe.service.example` — systemd unit (started by 2.3).
4. Validate: `docker compose -f deploy/docker-compose.yml config -q` (if docker present on host — check; if absent, YAML parse via python yaml + manual lint, and say so honestly in the doc). DO NOT bring OpenWA up against a real WhatsApp number.
5. Sync deltas into `docs/DEPLOYMENT.md`. Commit: `deploy: compose + webhook + systemd examples`.

---

# PHASE 5 — Re-baseline & sign-off (est. 1 day)

Everything before changed runtime behaviour and timings; the evidence base must be regenerated, not assumed.

1. **Performance re-measure:** run `tests/test_10_performance.py` (extended per 2.1) + full-suite wall time; rewrite `docs/PERFORMANCE.md` tables with real numbers incl. learned-model stage timings from Phase 1 smokes. Any stage now exceeding its budget tier gets an explicit budget-bump proposal in the doc (not silently applied).
2. **Red-team re-run:** full `tests/test_09_redteam.py` + the transform-matrix suite against the new learned stages (adversarial check: do compressed/transcoded variants of deepfake fixtures flip the learned score? If unstable, that finding goes into `DEEPFAKE_DETECTION.md` robustness section — honest reporting even if negative).
3. **Zero-retention final proof:** end-to-end run of every input kind (url, wav, mp4, pdf-sig, apk-stub) via CLI sim; after each: assert original + all derived files deleted, purge_audit entry with `residual_paths == []`. One consolidated evidence file.
4. **Capability status report v2:** rewrite the "Current overall capability posture" table in `GAPS_AND_ENABLEMENT.md` reflecting what actually switched live this cycle; produce the honest delta vs August-19 baseline (which targets moved Reduced→Good, which stayed, which got SLOW-tier follow-ups).
5. Final full `pytest tests/ -q` (expect 230± range; whatever the real number is, record it) + tag: `git tag v0.2.0-postroadmap` (baseline was implicit v0.1). Done.

---

## Sequencing summary

```
P0 hygiene ──► P1 models (AASIST→EFFORT→HAVIC) ──► P2 delivery/ops
                   │                                    │
                   └──────────► P3 languages ◄──────────┘ (independent, parallel)
                                              ──► P4 enablement/deploy (some cred-gated waits)
                                              ──► P5 re-baseline + sign-off
```

Parallelization: P3 translation subagents can fly while P1 smokes run (they're pure text work; smokes are the thermally exclusive resource — never two heavy inferences concurrently). P2.2/P2.3 can pull forward if P1 slips.

**Standing guardrails (every phase):** heavy compute sequential + thermal log watch (Aug 7 trip lesson); every task commit-sized; every new user-facing string rides the i18n review gate; every "done" claim backed by an ad-hoc verify script output, not just suite green; zero-cloud constraint non-negotiable (HF-token-dependent parts stay OFF by design, noted not apologized for).
