# Finish the Fusion Workflow — Implementation Plan (audio + video closeout)

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Close every remaining open item in the Vishwas fusion workflow — harden the VT domain-fallback path with a regression test, execute the AASIST checkpoint swap (the last real quality gap), scale the audio corpus, expand the video training set with audio-bearing AI clips — ending with a recalibrated operating point, green suite, and updated docs/skill.

**Architecture:** Four sequential work packages: (A) quick hardening wins, (B) AASIST checkpoint swap behind the existing env-var gate, (C) audio corpus scale-up using the already-downloaded ASVspoof2021 pool, (D) video set expansion + final re-evaluation. Every package ends in a measured proof (OOF AUC, corpus replay, or CLI E2E) — no self-reported success. Serving code changes are minimal because Fusion-v2 is already validated; most weight lands on data + calibration.

**Tech Stack:** pytest (`PYTHONPATH="/home/hermes/pylibs:/home/hermes/docling-python:$PWD/src"`), ffmpeg corpus tooling, existing harnesses `/home/hermes/fusion_audio/scripts/` + `/home/hermes/fusion_av/scripts/`, `fusion_train.py`, vendored detectors under `src/vishwas/model_archs/`, GPU opt-in via `VISHWAS_DEVICE=cuda` (webhook stays CPU).

---

## Current context / assumptions

- HEAD `6aff500` (2026-08-26). Suite baseline **365 passed / 7 skipped** (~16 s hermetic).
- Verification ladder all-PASS (`/tmp/verification/REPORT.md`); live WhatsApp round trip proven (AI video → do_not_use conf 0.655).
- **Live-probe findings 2026-08-26 (this planning session):**
  - VT Finding E fix **already implemented**: `src/vishwas/vt_client.py:105-115` falls back `/urls/{id}` → `/domains/{host}` on raw_status 404. Remaining gap = dedicated regression test coverage (verify, add if missing).
  - gov_document via CLI `--file` **already works** (Finding D fix in `router.py`; live probe of `aadhaar.pdf` routed `gov_document`, ran 5.04 s, digilocker/api_setu correctly `unavailable`). Remove from defect lists.
  - APK static tier **works end-to-end**: synthetic APK (dex + SEND_SMS/READ_SMS perms) → apk_statics ok, fused **do_not_use** (calibrated 0.70). PE tier works (pe_statics + ClamAV fresh DB + YARA-x + VT 76 engines). MoBFS/Quark/sandbox remain optional heavy tiers.
- Audio channel: AASIST checkpoint proven degenerate (post-mortem `35411b0`, swap steps in `docs/research/FUSION_DATASETS_2026-08-25.md` §AASIST — READ FIRST before Task B). XLSR separates but high variance (zero-pad fix `c2dfe14` landed).
- Data on disk: 240-clip ASVspoof2019 slice (`/home/hermes/fusion_audio/asv19_la/clips/` + protocol.csv), ASVspoof2021 eval tar intact 181k flacs (`/home/hermes/fusion_audio/asv21_la_eval/ASVspoof2021_LA_eval.tar.gz`), 84-row video feature corpus (`rows_video_v2.jsonl` under `/home/hermes/fusion_av/feat_vectors/`), FF++ subsets `subset/real2|fake2`.
- User-gated blockers (do NOT attempt autonomously): DigiLocker/API Setu registration, HF token for SSL-Audio, `sudo mv` of `/opt/verisafe/models`, physical second phone.

---

## Proposed approach

Strictly ordered packages A→D; each task is bite-sized with its own proof. Package B is the priority blocker for audio quality; C depends on B's outcome (a good checkpoint makes scaled data worthwhile); D is independent of B/C and can run in parallel waves if delegation capacity allows (max 2 concurrent calls — serialize heavy CUDA jobs regardless).

---

## Step-by-step plan

### Package A — Hardening quick wins

### Task A1: Verify/add VT /domains fallback regression test
**Objective:** lock the already-implemented Finding E fix against regression.
**Files:**
- Inspect: `tests/test_17_vt.py`, `tests/test_32_vt_cache.py`
- Modify (only if uncovered): `tests/test_17_vt.py`
**Steps:**
1. `grep -n "domains\|404\|fallback" tests/test_17_vt.py tests/test_32_vt_cache.py`
2. If a fake-opener test asserts `_get("/domains/<host>")` fires after a 404 on `/urls/<id>`: record PASS, done. Else write failing test:
```python
def test_url_404_falls_back_to_domain():
    calls = []
    def fake_opener(req, timeout=10):
        calls.append(req.full_url)
        if "/urls/" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"{}"))
        return FakeResponse(200, {"data": {"attributes": {"last_analysis_stats": {"malicious": 3, "undetected": 73}}}})
    cli = VtClient(opener=fake_opener)
    res = cli.check_url("https://evil.example.net/login")
    assert any("/domains/evil.example.net" in c for c in calls)
    assert res.counts.get("malicious") == 3
```
(Reuse the existing FakeResponse fixture style found in test_17.)
3. Run `PYTHONPATH=src python3 -m pytest tests/test_17_vt.py -q` → expect new test PASSes (fix already exists; if it FAILs the fix regressed — restore per vt_client.py:107-114).
4. Commit: `git commit -am "test(vt): regression coverage for url->domain 404 fallback (Finding E)"`

### Task A2: Refresh stale defect lists (skill + repo docs)
**Objective:** stop carrying closed defects (govdoc routing, VT fallback) as open.
**Files:** `~/.hermes/skills/devops/verisafe-operations/SKILL.md` ("Still user-gated" section), `docs/GAPS_AND_ENABLEMENT.md` if it lists them.
**Steps:**
1. Edit skill section: strike "VT Finding E" and "gov_document unreachable via CLI --file" (both fixed + verified 2026-08-26); keep AASIST/deMamba/user-gated items.
2. Grep repo docs for the same claims: `grep -rn "Finding E\|unreachable via CLI" docs/` → update wording.
3. No commit needed for skill (outside repo); commit repo doc change: `docs(security): mark Finding E + govdoc-routing resolved`.

---

### Package B — AASIST checkpoint swap (priority audio blocker)

### Task B0: Read the swap recipe
**Objective:** load exact candidate sources + drop-in steps.
**Files:** read `docs/research/FUSION_DATASATS…` → correct path `docs/research/FUSION_DATASETS_2026-08-25.md` §AASIST (post-mortem from commit `35411b0`).
**Pass:** can state candidate checkpoint names/sources + license status without guessing.

### Task B1: Source + license-check candidate checkpoint(s)
**Objective:** download ≥1 known-good WavLM-fronted AASIST-family checkpoint (e.g. DeepFense lineage per post-mortem) into `/opt/verisafe/models/aasist/candidates/`.
**Steps:** fetch via HF hub API (curl-only pattern in mamba scouting reference); record SHA256 + license tag in `candidates/PROVENANCE.md`. License must permit redistribution OR be eval-internal-only (note which).
**Pass:** file >100 MB, sha256 recorded, license line written. If ALL candidates license-blocked → STOP, report to operator (hard blocker).

### Task B2: Structure probe against vendored arch class
**Objective:** prove checkpoint keys map onto `src/vishwas/model_archs/aasist.py` before any scoring.
**Steps:**
1. Write throwaway probe (pattern: `scripts/verify_rawbmamba.py`): load state_dict, diff key sets vs model, print coverage n/m.
2. Expect ≥95% exact-name coverage; small mismatches → document mapping dict, do NOT blind-rename tensors.
**Pass:** key coverage reported; forward pass on a 4 s zeros tensor returns finite scalar.

### Task B3: Input-sensitivity quartet (GPU, serialized)
**Objective:** new weights must discriminate before we trust them.
**Steps:** reuse `/tmp/aasist_invariance.py` pattern + `tests/test_34_audio_input_sensitivity.py` with `VISHWAS_AASIST_WEIGHTS=<candidate>` exported, `VISHWAS_DEVICE=cuda`. Assert posteriors distinct across {silence, white noise, sine440, real speech clip from asv19 bona-fide}.
**Pass:** 4 distinct values (spread > 0.05 between min/max on speech vs silence).

### Task B4: THE PROOF — 240-clip OOF replay
**Objective:** independent replication of the acceptance bar.
**Steps:**
1. Score all 240 clips via the serving extract_features path (NOT the training harness) — byte-equal mechanism rule.
2. Compute OOF AUC vs protocol.csv labels; quantiles p25/p50/p75 per class.
**Pass (hard):** AUC ≥ 0.85 AND real-class p75 < fake-class p25 (non-overlap) AND scores not constant (std > 0.01). Any miss → checkpoint rejected, try next candidate; if none pass, audio stays calibration-only and Package C is descoped to data-prep-only.
3. Record numbers in `docs/research/FUSION_DATASETS_2026-08-25.md` §AASIST (append dated subsection).

### Task B5: Flip production + suite + webhook restart
**Objective:** make passing checkpoint the served default.
**Steps:**
1. Move candidate → canonical path referenced by `deploy/vishwas-secrets.env` `VISHWAS_AASIST_WEIGHTS` (keep old as `.degenerate.bak`).
2. Full suite: expect **365 passed / 7 skipped** (test_34 now exercises new weights — still green).
3. `systemctl --user restart vishwas-webhook` → poll `/health` until deps=12, device=cpu.
4. CLI smoke: `bash scripts/run_vishwas.sh cli --file <any asv19 fake clip> --media-type audio` → deepfake_audio target engaged, AASIST check status ok.
5. Commit env/doc changes: `feat(audio): swap AASIST checkpoint — 240-clip OOF AUC <X>, non-degenerate`.

---

### Package C — Audio corpus scale-up (conditional on B4 PASS)

### Task C1: Extract ASVspoof2021 eval slice
**Objective:** get a manageable scored pool without exploding disk.
**Steps:** `tar -xzf …/ASVspoof2021_LA_eval.tar.gz --wildcards '*/flac/*.flac' -C … | head` pattern — extract first 300 flacs + matching protocol file; verify count.
**Pass:** 300 flac files readable by soundfile; protocol rows join 1:1.

### Task C2: Score pool through serving path
**Objective:** honest OOF on unseen-year data for BOTH AASIST(new) and XLSR.
**Steps:** reuse fusion_audio harness scripts; write features jsonl; compute AUC per detector + fused.
**Pass:** report written to docs §scale-up (no threshold — measurement phase).

### Task C3: Re-run fusion_train on combined corpus
**Objective:** refit LR stack only if separating evidence now exists.
**Bar (pre-committed, from gbdt_report precedent):** OOF AUC ≥ 0.80 else NO wiring (keep heuristic/calibrated path). Wire via `VISHWAS_FUSION_DIR` checkpoint only on PASS.
**Pass:** either wired stack + suite green + 84-row-style audio replay distribution recorded, or documented rejection.

---

### Package D — Video set expansion + final re-evaluation

### Task D1: Curate audio-bearing full-AI video class
**Objective:** grow beyond the single 1-sample AI fingerprint.
**Steps:** collect ~20–60 full-AI clips WITH speech (operator generations preferred — ASK OPERATOR for their past generations; else public samples with provenance noted). Store `~/fusion_av/ai_full_av/` + manifest.csv (path, generator, has_speech).
**Pass:** ≥20 clips, manifest complete, each plays via ffprobe.

### Task D2: Feature extraction + replay through Fusion-v2
**Objective:** confirm current operating point catches the expanded class.
**Steps:** run effort/havic/frameheur/av_risk harness over new clips; append rows; replay FusionEngine; tabulate verdict distribution.
**Pass:** ≥90% of new AI clips land DO_NOT_USE; zero reals regress below CAUTION. Failures → analyze per-signal, document; do NOT retune thresholds ad hoc without corpus-level justification.

### Task D3: Final consolidated re-report + skill bump
**Objective:** one artifact closing the whole workflow.
**Steps:** update `docs/research/FUSION_FINAL_2026-08.md` (append 2026-08-XX section: new operating point, audio channel status, video expansion result); bump skill verisafe-operations to v1.3.0; refresh REPORT.md.
**Pass:** docs committed; suite green; health ok.

---

## Files likely to change

- `tests/test_17_vt.py` (maybe) — fallback regression test
- `src/vishwas/**` — expected NO changes except possibly none; all gating via env/checkpoints
- `deploy/vishwas-secrets.env` — VISHWAS_AASIST_WEIGHTS path flip (B5)
- `/opt/verisafe/models/aasist/{candidates/,*.bak}` — checkpoint files
- `docs/research/FUSION_DATASETS_2026-08-25.md`, `docs/research/FUSION_FINAL_2026-08.md`, `docs/GAPS_AND_ENABLEMENT.md`
- `~/fusion_av/ai_full_av/` + feat vectors jsonl (new data, outside repo)
- Skill `verisafe-operations` SKILL.md (v1.2.0 → v1.3.0)

## Tests / validation

- Hermetic gate throughout: `cd ~/vishwas && PYTHONPATH=src python3 -m pytest tests/ -q` → 365 passed / 7 skipped minimum; zero failures blocks merge.
- Weighted gates: test_34 sensitivity quartet on GPU after any checkpoint change.
- Corpus gates: B4 (240-clip AUC ≥0.85 + quantile non-overlap), D2 (≥90% DNU on expanded AI set, zero real regressions).
- Service gate after B5: `/health` deps=12 device=cpu + CLI audio smoke.

## Risks / tradeoffs / open questions

- **License wall (B1):** if every candidate checkpoint is non-redistributable, ship eval-internal-only and say so in docs — never silently ship unlicensed weights into verdict paths.
- **CUDA contention:** serialize all GPU jobs; never run two heavy gates concurrently (VRAM + timing skew).
- **C depends on B:** if no checkpoint passes B4, Package C becomes prep-only (extract + score + document) and the audio channel honestly remains calibration-only.
- **Threshold discipline:** any fusion weight/threshold change MUST be justified against full-corpus replays, not individual clips (gbdt_report lesson).
- **Open Q1:** does the operator have past AI video generations to seed D1 (preferred, adds real-world-distribution samples)?
- **Open Q2:** for C3 wiring, is `VISHWAS_FUSION_DIR` acceptable as the delivery vehicle (yes per README config table)?
- **Operator-gated items parked:** DigiLocker/Setu registration, HF token (SSL-Audio), `sudo mv /opt/verisafe/models` → `/opt/vishwas/models`, second-phone demos.
