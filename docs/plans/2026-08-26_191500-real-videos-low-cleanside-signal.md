# Real Videos Read LOW/trust — Clean-Side AV-Sync Evidence (deepfake_video) Implementation Plan

> **For Hermes:** Use subagent-driven-development skill. Implement task-by-task, run the independent sub-workstreams in parallel, then re-run the 4 operator videos through the updated version.

**Goal:** Make genuine real videos (esp. audio-bearing, AV-synced) reach **RISK LEVEL: LOW (TRUST)** instead of bottoming out at MEDIUM — while keeping AI/deepfake video at HIGH and never producing a false-clean on real deepfakes.

**Architecture:** Add a **clean-side (negative-weight) AV-sync evidence signal** to the `deepfake_video` fusion. Root cause: every current `deepfake_video` weight is positive (effort 1.5, frameheur 1.0, av_risk 3.5, havic 0.8), so `x = Σ(w·cal)/Σ(w)` is always ≥ 0 → `raw = sigmoid(4x) ≥ 0.5` → never TRUST (`raw ≤ 0.15` needs `x ≤ −0.44`). The fix mirrors how `gov_document` already uses negative weights (`signature.valid −4.0`) as clean evidence. `cross_modal` already emits the needed evidence: `alignment_class="synced"` (real lip-sync, `r ≥ 0.35`, `lag ≤ 100 ms`).

**Tech Stack:** Python 3.12, vishwas fusion.py, cross_modal.py, the existing hermetic pytest suite (370 pass / ~12 skip), live webhook route.

---

## Current context / assumptions
- HEAD will be `960d6ab` + pending sibling commits. **CRITICAL: sibling P1 (`deleg_ee4bbee1`, DigiLocker removal) is MID-EDIT on `src/vishwas/fusion.py` right now. This plan MUST NOT touch `fusion.py` until P1 commits** — wait for it, then apply on top. Sibling TC (`deleg_c47af292`, ASVspoof19-LA audio proof) only touches data dirs + reads `model_archs/aasist3.py` — no repo write conflict; its verdict may or may not be back, see Risk gate below.
- 4 real audio-bearing operator videos already verified live → all MEDIUM/caution, and are archived at `~/fusion_av/trackA_operator_reals/` (this is the re-verification corpus).
- Suerbase hermetic suite: `env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS PYTHONPATH=src python3 -m pytest tests/ -q` → 370 passed / 12 skipped.

## Proposed approach
1. **Clean-side negative signal:** add `av.synced_clean` to `WEIGHTS["deepfake_video"]` at `-2.5`, `_SIGNAL_SOURCES["av.synced_clean"] = ("cross_modal_av", "av_correlation", …, True)`. Wire it so it contributes its negative weight **only when** the cross-modal evidence independently confirms `alignment_class == "synced"` (real lip-sync). Because the existing `fully_generated` fake-pattern already requires `av_risk ≥ 0.45` (anti-correlated), a genuinely synced real video cannot also trip the fake pattern → the clean bonus is structurally safe.
2. **Negativity must be *conditional*** (not a flat negative weight on the raw continuous `av_correlation`): a negative weight applied to raw `av_correlation` would fire even on low/ambiguous correlation. Implement as a small `_extract`-side gate or a dedicated signal that yields `-cp` only when `alignment_class == "synced"` (else 0, neutral known-gap). Reflect this in the code comment so nobody later turns it into a flat negative.
3. **Retune + safety bar on the corpus:** re-run the v3 offline replay harness (`/tmp/merge_havic_check.py` pattern / `rows_video_v3.jsonl`) with the new signal and confirm the three bars hold BEFORE `fusion.py` is written: REALS (incl. the 4 audio reals) NOT-DNU, AI anchor still DNU, fakes ≥ CAUTION.
4. **Re-verify live:** after merge, run the 4 operator videos through the updated webhook/CLI; each real audio video must now read **LOW/trust** (or at minimum not HIGH), and the AI anchor + a synthetic fake still read HIGH/do_not_use.
5. Run this task **in parallel** with the other autonomous sub-workstreams (P1 removal, TC audio) per the user's request — but do NOT write `fusion.py` until P1 has committed (avoid git conflict).

---

## Step-by-step plan

### Task 1: Wait for P1 to commit, then branch cleanly
**Objective:** Ensure `fusion.py` is stable before this plan edits it.
**Steps:**
1. Confirm P1 (`deleg_ee4bbee1`) has committed the DigiLocker removal (check `git log --oneline -1` and that `src/vishwas/fusion.py` working tree is clean).
2. If P1 not done, do NOT edit `fusion.py` — work on Tasks 2–3 (which don't touch it) and revisit.
**Pass:** `git status --short src/vishwas/fusion.py` clean; local-only work proceeds in parallel.

### Task 2: Add the clean-side signal plumbing (TDD)
**Files:**
- Modify: `src/vishwas/fusion.py` (WEIGHTS + _SIGNAL_SOURCES + a conditional-extract helper) — ONLY after Task 1
- Create: `tests/test_41_av_synced_clean.py`
**Step 1 — failing test:**
```python
def test_synced_real_reaches_trust():
    checks = [
        _ck("effort_face_forensics", {"prob_deepfake": 0.45}),
        _ck("cross_modal_av", {"av_correlation": 0.62, "best_lag_ms": 40,
                               "alignment_class": "synced", "av_risk_addition": 0.0}),
        _ck("frame_heuristics", {"prob_deepfake": 0.2}),
    ]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict is Verdict.TRUST             # FAILS before: currently CAUTION

def test_decorrelated_still_caution_or_above():
    checks = [_ck("cross_modal_av", {"av_correlation": 0.10, "best_lag_ms": 200,
                                     "alignment_class": "decorrelated", "av_risk_addition": 0.45})]
    d = FusionEngine().decide("deepfake_video", checks)
    assert d.verdict in (Verdict.CAUTION, Verdict.DO_NOT_USE)  # never TRUST on bad sync
```
**Step 2 — run:** `PYTHONPATH=src python3 -m pytest tests/test_41_av_synced_clean.py -q` → FAIL (synced returns CAUTION).
**Step 3 — implement:** add `av.synced_clean` weight `-2.5` + the conditional extract (return `-correlation` when `alignment_class=="synced"` else gap/0) + `_SIGNAL_SOURCES` entry.
**Step 4 — run:** the two tests pass; sync→TRUST, decorrelated→CAUTION.
**Step 5 — commit:** `feat(video): clean-side AV-sync evidence allows real synced video to reach TRUST/LOW`.

### Task 3: Corpus retune + safety gate (no fusion.py write yet)
**Files:** `/tmp/vid_avsync_retune.py` (reuse `/tmp/merge_havic_check.py` replay pattern; read `rows_video_v3.jsonl`)
**Objective:** prove the new signal keeps all three bars under the updated weights.
**Steps:**
1. Replay v3 corpus with proposed weights (effort 1.5, av_risk 3.5, havic 0.8, + `av.synced_clean -2.5`) on the *live* feature set.
2. BARS: REALS (incl. the 4 audio reals scored to v3) NOT-DNU; AI anchor 3/3 DNU; fakes ≥ CAUTION (42/45). Hold if any bar fails.
3. Also score the 4 operator reals manually (they MUST reach TRUST/low with the new signal present, since they're audio-synced) — if they still read >0.35, strengthen `av.synced_clean` magnitude (−3.0/grounded).
**Pass:** numbers table written to `/tmp/vid_avsync_retune.md`.

### Task 4: Live re-verification (after merge + webhook restart)
**Files:** none (reads artifacts)
**Steps:**
1. Restart webhook; `curl /health` ok.
2. Run the **4 operator videos** from `~/fusion_av/trackA_operator_reals/` through `scripts/run_vishwas.sh cli --file …` (and/or the live HMAC smoke).
3. Expected: each real audio video → **RISK LEVEL: LOW (TRUST)**; re-confirm `ai_crf45.mp4` → **HIGH (do_not_use)**; a synthetic fake (e.g. FF++ FaceSwap_* clip) → ≥ CAUTION.
4. Log per-clip verdict + raw/confidence.
**Pass:** 4 reals → LOW/trust, AI → HIGH, fake → CAUTION+.

### Task 5: Full suite + docs + commit
1. Hermetic suite full: `env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS PYTHONPATH=src python3 -m pytest tests/ -q` → green (recount; expect 374+ pass incl. test_41, ~12 skip).
2. Update `docs/research/FUSION_FINAL_2026-08.md` (record clean-side signal + new op-point: real audio-synced → LOW/trust; AI anchor → HIGH), skill `verisafe-operations` reference note.
3. Commit: `feat(video): real synced video reads LOW via clean-side AV evidence — re-verified live on 4 operator videos`.

---

## Files likely to change
- `src/vishwas/fusion.py` (weights + signal source + conditional extract) — ONLY after P1 commits
- `tests/test_41_av_synced_clean.py` (new)
- `tests/test_30_fusion_trust.py` (extend clean fixture to include `av.synced_clean`-bearing synced case, or add a synced-real trust case)
- docs `FUSION_FINAL_2026-08.md`, skill reference

## Tests / validation
- test_41 (synced→TRUST, decorrelated→not-TRUST) — TDD
- v3 corpus retune bars (reals NOT-DNU, AI DNU, fakes ≥ CAUTION)
- live: 4 operator reals → LOW; ai_crf45 → HIGH; fake → CAUTION+
- full hermetic suite green

## Risks / tradeoffs / open questions
- **False-clean risk (highest):** if `av.synced_clean` is too strong, a *convincing* deepfake with real lip-sync (achievable by SOTA) could read LOW. Mitigation: the safety gate (Task 3) + require `synced` AND low effort/frameheur, and preferably await TC's audio verdict on whether AV-sync is genuinely discriminative before final magnitude.
- **P1 git conflict:** the #1 operational risk — `fusion.py` must not be edited until P1 commits; done via Task 1 sequencing.
- **TC dependency:** recommend wiring the negative signal AFTER TC returns its ASVspoof19-LA verdict if it lands within this window; if TC still running, this plan proceeds with the safety-gated magnitude and marks "final magnitude contingent on TC."
- **Open Q:** should a real video go straight TRUST/LOW, or should the clean path be weaker (LOW but not auto-TRUST) to be extra-safe? Default per user intent = reach LOW/trust. Tradeoff: aggressive = fewer real false-alarms but slightly higher false-clean surface; conservative = safer but some real videos stay MEDIUM. User explicitly asked to get true videos to say LOW, so default = reach TRUST, but pinned to the Task 3 corpus gate.