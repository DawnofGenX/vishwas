# Vishwas — Remaining Work Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Close the honest remaining gaps so Vishwas's three "Reduced"-posture targets (deepfake_video, deepfake_audio, cross_modal) can use their provisioned model weights, and ship the missing hygiene/deploy artefacts.

**Architecture:** Weights for AASIST (3.79 GB), EFFORT×3 (1.21 GB each), HAVIC×2 (~0.86 GB + ~0.97 GB) are on disk at `/opt/vishwas/models/` (sha256-verified, `scripts/provision_weight_env.sh` exports env vars). The adapter registry (`src/vishwas/model_adapters.py`, 7 families) correctly reports all raw state-dicts as unusable because **the architecture classes have never been vendored**. Each capability then falls back to heuristics and records `unavailable` evidence. Vendoring one class per family flips that stage from heuristic-only to learned-model scoring while keeping every existing fallback path intact (adapters return `None` when load fails — zero behaviour change on regression).

**Tech Stack:** Python 3.12, torch (importable in bare python3, CPU-only), numpy, pytest suite at `tests/` (currently 205 passed in ~12s), i5-8250U thermal constraints (keep heavy CPU load modest; use `VISHWAS_FFMPEG_THREADS` caps where media involved).

**Verified current state (2026-08-20):**
- `pytest tests/ -q` → **205 passed**, 0 failed
- Git repo exists but has **zero commits** — all project files untracked
- `docs/GAPS_AND_ENABLEMENT.md` is current and accurate
- `/opt/vishwas/models/`: aasist, effort, havic populated; demamba/fakemamba/ssl_audio/image_face intentionally skipped (no public checkpoint / needs HF token)
- `src/vishwas/assets/ca_truststore/` contains only `README.md` (no anchor certs)
- `deploy/` is empty (no OpenWA compose/webhook artefacts)

---

## Out-of-scope (ops/credentials, tracked but NOT planned here)

| Item | What blocks it | Owner action |
|---|---|---|
| VirusTotal reputation (`vt` gate) | No `VISHWAS_VT_API_KEY`; client code already live + tested (test_17) | Get a key, export env |
| DigiLocker e-KYC / API Setu consumer creds | Partnership-gated OAuth; discovery-only surface reachable | Register consumer identity |
| DeMamba / Fake-Mamba weights | No public checkpoint exists anywhere | Revisit quarterly |
| SSL-audio weights | Requires HF auth token (zero-cloud constraint keeps it off) | N/A unless policy changes |
| IMAGE_FACE weights | No public small-weight URL found | Research follow-up |

These stay as `unavailable`-evidence rows in reports — by design, nothing silently pretends to work.

---

## Task Group A — Baseline hygiene (do first, 2 min)

### Task A1: Initial git commit of the whole tree

**Objective:** Stop the entire project living in the working tree with zero version history.

**Steps:**
1. `cd /home/hermes/vishwas && printf '.delegation/\n.test-quarantine/\nlogs/\n__pycache__/\n*.pyc\n' > .gitignore`
2. `git add .gitignore src/ tests/ scripts/ docs/ fusion/ deploy/ README.md` (exclude `.delegation/`, `.test-quarantine/`, `logs/` via ignore)
3. `git commit -m "chore: baseline import of Vishwas platform (pre-gap-close)"`
4. Verify: `git log --oneline | head -1` shows the commit; `git status --short` clean except ignored dirs.

**Files:** Create `.gitignore`; new root commit.

---

## Task Group B — Vendor architecture classes (the main remaining technical work)

Ordering rationale: AASIST first (single audio model, best-understood public architecture), then EFFORT (spatial face forensics, 3 checkpoint variants), then HAVIC (largest, both modalities, slowest expected runtime on this laptop). After each, run a **real CPU smoke inference** on a tiny fixture with a hard wall-clock cap and record timing into `docs/PERFORMANCE.md`.

Shared contract per task (DRY — put helpers once, reuse everywhere):

**Task B0: Add `load_from_state_dict` helper + weight-provenance metadata**

**Files:** Modify `src/vishwas/model_adapters.py` (near `_default_load`, line ~149); Create `src/vishwas/model_archs/__init__.py`, `src/vishwas/model_archs/aasist.py`, `effort.py`, `havic.py`.

- `src/vishwas/model_archs/base.py`: 
  ```python
  class ArchSpec:
      name: str
      def build(self) -> Any: ...            # returns nn.Module (or plain callable)
      def apply_state(self, model, sd: dict) -> None: ...  # key-map + strict=False load
  ```
- Extend adapter loading: if env var resolves AND `model_archs/<family>.py` exists AND exposes `build()`, construct model, `apply_state()`, hand to existing `_call_model`. Failure at any step → return `None` (existing unavailable-evidence path). No new failure modes leak to callers.
- Record in manifest (`docs/research/MODEL_WEIGHTS_MANIFEST.json`) per gate: `"arch_vendored": true/false` + `"smoke_result"` after each task.

**Test:** `tests/test_14_model_adapters.py` — add a case asserting that with fake weights + stub arch module, adapter yields a usable object; keep hermetic (no torch forward pass in unit layer). Run: `PYTHONPATH=/home/hermes/pylibs:$PWD/src python3 -m pytest tests/test_14_model_adapters.py -q` (expect pass before AND after real arch lands; real verification happens in smoke steps below). Commit after.

### Task B1: AASIST audio-spoofing network

**Objective:** `VISHWAS_AASIST_WEIGHTS` (`best_model.pth`, torch dict `{model_state, optimizer_state, epoch}`) becomes loadable; `deepfake_audio` T-tier moves from heuristics to learned AASIST scores with heuristic fallback preserved.

1. Read weight key shapes from disk: `PYTHONPATH=$PWD/src python3 - <<EOF  ... torch.load('/opt/vishwas/models/aasist/best_model.pth')['model_state'] ... print(sorted(sd.items())[:40])` — derive exact layer names/dims before writing code (don't guess; mirror the pattern used for pades_check RDN probing).
2. Implement `src/vishwas/model_archs/aasist.py`: mel-spectrogram front-end (reuse `_mel_aasist_preprocess` already in adapters) + spectro-temporal graph-attention trunk matching those keys (Jung et al., ASVspoof lineage, arXiv:2110.01200; official reference implementations exist — port minimal layers, YAGNI: score head only, no auxiliary training heads).
3. Wire into adapter via B0 hook.
4. **Live smoke (thermal-safe):** `source scripts/provision_weight_env.sh && PYTHONPATH=... python3 -c "...adapter.run(small_wav_fixture)"` with a 5s wav from `tests/fixtures/`; wall-clock cap 120s; if > 120s, fall back to recording timings + marking tier as SLOW in PERFORMANCE.md rather than removing it.
5. Update `docs/GAPS_AND_ENABLEMENT.md` row for `model-weights` + posture table row `deepfake_audio` (Reduced → Good-or-Reduced-SLOW, be precise in the doc).
6. Commit: `feat(audio): vendor AASIST arch; learned tier active for audio spoofing`.

### Task B2: EFFORT (OrthAlign) spatial face-AIGI detector

**Objective:** `VISHWAS_EFFORT_WEIGHTS` (effort_chameleon.pth; ffpp/genimage siblings share shape family) loadable; `deepfake_video` frame-scoring upgrades.

1. Probe tensor names on all 3 checkpoints (`torch.load` each, compare key sets). Per Decision #2: **chameleon is primary**; ffpp/genimage load into the same class only as startup fallback if chameleon is missing/unreadable — assert all three share the key family so one arch class covers them (cheap to verify once, at probe time).
2. Implement `model_archs/effort.py` around the OrthAlign subspace-decomposition design of arXiv:2411.15633 (official repo `YZY-stack/Effort-AIGI-Detection`). **License confirmed CC BY-NC 4.0** (README badge; no LICENSE file) — port the algorithm, do NOT copy repo code wholesale; attribution + NC header in the module docstring; env-var opt-in per Decision #1.
3. Adapter hook (B0), reuse `_face_crops_preprocess` multi-crop median aggregation already present.
4. Smoke: 1-frame extract from `tests/fixtures` mp4, run chameleon variant only (cap 180s; thermal-watch `tail ~/.hermes/logs/thermal_power_monitor.log`). If the stage exceeds budget, mark SLOW tier — verdict ships fast, heavy result follows via the non-blocking follow-up mechanism (Decision #3).
5. Docs update (incl. NC-license notice sections) + commit: `feat(video): vendor EFFORT arch (CC-BY-NC, opt-in); chameleon primary with lazy fallback`.

### Task B3: HAVIC holistic AV intrinsic-coherence model

**Objective:** `VISHWAS_HAVIC_WEIGHTS` (`best_ft_model.pth` primary, `pt_model.200.pth` secondary) loadable; `cross_modal` gains learned coherence scoring.

1. Probe both checkpoints; determine which pairs audio+video features and its input conventions.
2. Implement `model_archs/havic.py` (paper 2603.23960; RUNNABLE-SLOW expectation is documented — budget 4–6 h worst-case was our own estimate; on this laptop expect slower, hence the smoke cap and explicit "slow-tier" UX wording in the report template if it exceeds).
3. Adapter hook, smoke on one short AV clip, docs, commit: `feat(cross-modal): vendor HAVIC arch; AV coherence tier active`.

**Group B acceptance (proof bar — not just suite green):** for EACH family run an ad-hoc verification script in a tempdir (cleaned up after) that (a) loads the real weights from `/opt/vishwas/models/`, (b) confirms `is_usable_model(adapter result) is True`, (c) runs one real inference with measured time, prints PASS/FAIL lines — matching the user's standard that "suite green alone is not proof." Then full `pytest tests/ -q` must still be 205+/0-fail.

---

## Task Group C — Seed CA trust store (small, optional-but-cheap)

**Objective:** `src/vishwas/assets/ca_truststore/` currently holds only a README. Ship 2–3 well-known government-document issuer anchor public certs (e.g., UIDAI/DigiLocker ecosystem issuing CA if publicly published; otherwise generic national PKI roots relevant to the supported doc classes) so PAdES chain-anchoring isn't purely operator-configured.

1. Identify candidate anchor certificates for the doc classes in the RAG cache (`scripts/build_rag_cache.py` digests 6 issuer-trust sources) — only include cert chains actually referenced by verified fixtures in `tests/fixtures`.
2. Save as `<issuer_name>.der` + per-cert provenance comment file; update truststore README with fetch date + source URL.
3. Extend `tests/test_12_pades.py` with one case: fixture-signed-by-trusted-anchor → `chain == "trusted"` (if no suitable public anchor exists that signs our fixtures, add a generated self-issued test CA instead and document that production anchoring is operator-supplied — be honest in the doc either way).
4. Commit: `feat(pades): seed ca_truststore anchors + anchor-verification test`.

---

## Task Group D — Deployment artefacts

**Objective:** `deploy/` is empty even though `docs/DEPLOYMENT.md` describes the OpenWA docker-compose setup. Materialise it so a fresh host can start the stack without reverse-engineering the doc.

1. `deploy/docker-compose.yml`: OpenWA service (rmyndharis/OpenWA v0.21.0 image tag pinned) + vishwas service (env-wired: `OPENWA_BASE_URL`, `X-OpenWA-Signature` secret, `VISHWAS_*` gates, docling launcher `scripts/run_vishwas.sh` as entrypoint) + healthcheck.
2. `deploy/webhook.example.json` — the OpenWA webhook registration pointing at vishwas's route (matching `parse_openwa_webhook` in `src/vishwas/channels/`).
3. Smoke: `docker compose -f deploy/docker-compose.yml config` (validate YAML/env interpolation only — do NOT bring up OpenWA against a real WhatsApp account; CLI sim path `app.py` remains the default dev loop).
4. Sync any deltas back into `docs/DEPLOYMENT.md`. Commit: `feat(deploy): openwa+vishwas compose and webhook examples`.

---

## Sequencing & effort summary

| Order | Task | Est. effort | Risk |
|---|---|---|---|
| 1 | A1 git baseline | 2 min | none |
| 2 | B0 shared arch-loading hooks | ~30 min | adapter refactor touch — guarded by test_14 |
| 3 | B1 AASIST | 2–4 h incl. smoke | medium: state-dict key-mapping unknown until probed; CPU latency may mark tier SLOW |
| 4 | B2 EFFORT | 3–5 h | medium: license-unconfirmed reference impl → port-not-copy; 3 variants |
| 5 | B3 HAVIC | 4–6 h | high: largest/slowest; may end up SLOW-tier only — acceptable, document honestly |
| 6 | C truststore | 1–2 h | low |
| 7 | D deploy | 1–2 h | low |

Thermal guardrail for ALL of B: run smokes sequentially (never two heavy inferences concurrently — lesson from the Aug 7 trip), tail `~/.hermes/logs/thermal_power_monitor.log` during runs, abort-and-log if CRITICAL alerts fire.

## Decisions (resolved 2026-08-20, user answers)

1. **EFFORT licensing** — RESOLVED by inspection: repo `YZY-stack/Effort-AIGI-Detection` has no LICENSE file; its README badge declares **CC BY-NC 4.0** (non-commercial only). Decision: **ship B2 with a prominent NC-license notice in docs + disable-by-default-unless-operator-opts-in**. Concretely: adapter env var stays opt-in (`VISHWAS_EFFORT_WEIGHTS` exported = enabled); when enabled, reports carry a `"license": "CC-BY-NC-4.0"` flag on the evidence line; `GAPS_AND_ENABLEMENT.md` + `USER_GUIDE.md` get an explicit notice section stating Vishwas is not for commercial use while this stage runs. Port algorithm, no wholesale repo-code copy, attribution header in `model_archs/effort.py`.
2. **EFFORT variant selection** — **chameleon checkpoint only (primary)**; ffpp/genimage load lazily as fallback ONLY if the chameleon file is missing/unreadable at startup. Rationale: keeps heavy compute at 1× on this laptop. Fallback logic lives in the arch loader, not the capability layer.
3. **SLOW-tier UX** — (no answer received within window; operator default applied) **Non-blocking follow-up message**: fast verdict ships immediately from heuristics/T1 tiers; if a learned-model stage finishes later it arrives as a second WhatsApp message, consistent with the existing progressive short-circuit design and keeping reply latency off the thermal-sensitive path. Implement via the orchestrator's staged-report mechanism; report template gains a `pending_heavy` field so follow-ups are deterministic rather than LLM-composed.
