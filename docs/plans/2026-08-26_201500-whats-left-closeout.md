# What's Left — Message Polish + Cross-Media Consistency + Repo Cleanup — Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Close out the remaining loose ends: (1) finish the user-facing message polish, (2) prove the
new concern-bullet reply renders sensibly across every media type, and (3) tidy the untracked
plan/housekeeping files. Autonomous, no external credentials needed.

**Architecture:** Three small, independent workstreams. W-MSG is a one-line presentation tweak +
regression test. W-MATRIX verifies (tests-first) that each media path emits coherent concern bullets
and the correct risk-line-first format. W-CLEAN tidies untracked files (safe housekeeping, no logic).

**Tech Stack:** Python 3.12, pytest, `src/vishwas/report.py`, `src/vishwas/i18n.py`, `tests/`.

---

## Defaults applied (grill timed out — changeable)

| Decision | Default |
|---|---|
| Title capitalization | **`🔴 LIKELY FAKE · HIGH RISK`** (all-caps, matches existing `RISK LEVEL:` styling) — from `risk.lower()` t... |
| In-scope | Message polish + cross-media matrix + plan-file cleanup |
| Out-of-scope (Non-Goals) | Gold-plating (#3: second audio signal, SyncNet v2, image-corrobor... |
| Live phone round-trip | Deferred — flagged as user-dependent, not required for this commit |

---

## Current context / assumptions
- Head `a41830e`; suite green (386 passed / 13 skipped); webhook healthy (deps 12).
- The reply now emits tile + ⚠️ bullets + recommendation for CAUTION/DO_NOT_USE (report.py:126-144; verified
  live on an AI video). Risk-line-first invariant holds.
- `report.py:134` passes `risk.lower()` to the tile → currently renders "high Risk". The `_VERDICT_TILE`
  values are `("LIKELY FAKE","High","🔴")` etc. (trusted source of the labeled text).
- 14 untracked `.hermes/plans/*.md` + 1 `fusion/training/stack_deepfake_audio.json` clutter `git status`.
- Media types reachable: video, image, audio, url_phishing, gov_document, malware_file/APK.

---

## Step-by-step plan

### W-MSG Task 1: HIGH RISK capitalization tweak (TDD)
**Objective:** Render the tile risk adjective all-caps to match `RISK LEVEL:`.
**Files:** modify `src/vishwas/report.py:133-135`; extend `tests/test_43_concern_bullets.py`.
1. **Write failing test**: `test_build_renders_localized_bullets_not_keys` (extend) OR a new
   `test_tile_risk_all_caps` — assert `"LIKELY FAKE · HIGH"` appears in the built text (not "High").
2. Run `pytest tests/test_43_concern_bullets.py -q` → FAIL (current text "high Risk").
3. **Implement**: change line 134 to `risk=risk.upper()` (or drop `.lower()`).
4. Run → PASS. Commit: `git add src/vishwas/report.py tests/test_43_concern_bullets.py && git commit -m "style(report): verdict tile risk in caps (matches RISK LEVEL line)"`.

### W-MATRIX Task 2: cross-media consistency test (TDD, pure + hermetic)
**Objective:** Prove each media type emits a coherent, risk-line-first reply with the right concern
bullets and no raw-key leaks, at the `ReportBuilder.build` unit level (no model weights needed).
**Files:** create `tests/test_44_media_matrix.py`; reuse `_ck` pattern from test_43.
1. **Write tests** (parameterized across targets/checks):
   - video: `[aasist high, effort low]` → `RISK LEVEL: HIGH`, bullet contains "voice shows signs of
     AI manipulation", NO `concern_*` raw key.
   - video-sync: `cross_modal prob_inconsistent high` → bullet mentions "lips"/"audio/video".
   - image: SPAI evidence fires but verdict is `CAUTION` (image caps at CAUTION) → tile `SUSPICIOUS`,
     bullet "face appears digitally altered" (if fired) OR unverified-source only.
   - url: `vt positives_ratio high` + `phish_heuristics host_string_score high` → bullets "poor security
     reputation" + "disguised copy".
   - govdoc: `prob_forged high` → bullet "signs of forgery".
   - For each: first line `startswith("RISK LEVEL: ")`, and `"concern_" not in text`.
2. Run → FAIL (undefined) then implement nothing new — these pass against the shipped `concerns_for`
   (confirm expectations match actual mapping; correct the test's expected strings if a mapping differs
   — do NOT change product behavior to force a test).
3. Run full `tests/test_08_i18n_report.py` + `test_43` + `test_44` → green. Commit:
   `git add tests/test_44_media_matrix.py && git commit -m "test(report): cross-media concern-bullet matrix"`.

### W-CLEAN Task 3: tidy untracked housekeeping files
**Objective:** Remove the 14 untracked `.hermes/plans/*.md` and `stack_deepfake_audio.json` from
`git status` without deleting useful artifacts.
**Files:** `.gitignore`, optionally move plans to a repo-tracked `docs/plans/`.
1. Add to `.gitignore`: `.hermes/plans/` (if the runtime dir is ephemeral/scoped) — OR if these plans
   are valuable, move them into `docs/plans/` (tracked) instead. **Default: move to `docs/plans/`**
   (they're hand-offs from real sessions; keep them, just track or archive them).
2. Check `fusion/training/stack_deepfake_audio.json` — if it's a stray probe artifact, `git rm`/move to
   an archive dir; if it documents training params, keep and reference in docs.
3. Run `git status --short` → clean of untracked noise. Commit:
   `git add .gitignore docs/plans fusion/training && git commit -m "chore: archive session plans; tidy untracked artifacts"`.

### W-INT Task 4: full suite + docs/skill + commit
1. `env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS PYTHONPATH=src python3 -m pytest tests/ -q`
   → green (recount; expect 386 + new matrix). HARD BAR: no new failures, risk-line-first intact.
2. Update `docs/research/FUSION_FINAL_2026-08.md` (add the richer-reply commits) and the skill
   `~/.hermes/skills/devops/verisafe-operations/SKILL.md` (note the concern-bullet reply + the
   `references/reply-format-risk-level.md`).
3. Restart webhook to ship W-MSG; `/health` → `ok deps 12`. Commit docs.

---

## Tests / validation
- test_43 (extend): tile all-caps + localized bullets no-key-leak.
- test_44 (new): cross-media matrix — every reachable media type, risk-line-first, coherent bullets,
  no `concern_*` raw keys.
- Full hermetic suite + webhook health + (deferred) live phone round-trip.

## Risks / tradeoffs / open questions
- **All-caps vs sentence-case title**: caps match the existing `RISK LEVEL:` styling (visual consistency);
  tradeoff is mild shoutiness. Default caps; revertible if you dislike it.
- **Media-matrix expectations must match product mapping**: if a test's expected bullet string diverges
  from `_CONCERN_RULES`, correct the TEST (not the code) so we don't distort behavior to pass a test.
- **Plans hygiene**: moving `.hermes/plans/` into `docs/plans/` keeps the history; a pure `.gitignore`
  drop would lose it. Recommend move. (Open question — confirm you want them kept vs dropped.)
- **Live delivery** deferred (needs your phone) — not a blocker for this commit.

## Execution handoff
Plan complete and saved. Ready to execute via subagent-driven-development (W-MSG/W-MATRIX are pure +
independent → parallel; W-CLEAN independent; all merge in W-INT). Shall I proceed?