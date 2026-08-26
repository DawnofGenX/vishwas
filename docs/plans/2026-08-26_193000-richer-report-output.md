# Richer User-Facing Output: Verdict Tile + Concern Bullets + Recommendation — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Upgrade the WhatsApp verification reply from the current terse template stack to a
narrative-style message with a **verdict tile**, a bulleted list of **specific concerns** (from the
checks that actually fired), and a **clear user recommendation** — matching the draft:

```
After analyzing the video, the system gives a 🔴 LIKELY FAKE verdict with a High Risk rating.
It identifies two key concerns:
   ⚠️ the voice shows signs of AI manipulation
   ⚠️ the claim could not be verified through reliable sources
Based on these findings, the system recommends users: "Don't forward. Verify with a trusted source."
```

**Architecture:** Keep the deterministic verdict/confidence/advice templates in `ReportBuilder` (they
are the trust anchor + already localized), and ADD a **concern-extraction layer** that reads the
individual `CheckResult` objects (per-channel `prob_*` signals) to emit the ⚠️ bullets, plus a
**recommendation** line that follows the verdict's existing advice mapping. Presentation-layer only —
JSON API verdicts, CLI debug verdicts, and the fusion scoring are untouched.

**Tech stack:** Python 3.12; `src/vishwas/report.py`, `src/vishwas/i18n.py`, `tests/test_08_i18n_report.py`,
`tests/test_43_concern_bullets.py` (new). No new deps.

---

## Current context / assumptions

- `ReportBuilder.build()` (report.py:46-92) currently emits: `RISK LEVEL:` line → verdict sentence →
  optional pattern sentence → `confidence_line` → domain tip → `evidence_missing` → optional `llm_advice`.
  First line MUST remain `RISK LEVEL: X` (pinned by test_08:31-45).
- `reasons` is mostly machine tokens (`risk_raw:`, `incomplete_evidence:`, `pattern:...`) — NOT enough
  to source ⚠️ bullets. The per-channel evidence lives in `checks: list[CheckResult]`, each carrying
  signals like `prob_deepfake` (audio/video), `prob_inconsistent` (A/V), `positives_ratio` (VT),
  `host_string_score` (URL), `prob_forged` (docs). **Concerns must come from `checks`, not `reasons`.**
- i18n is a 7-language nested dict (`_DEFAULTS[key][lang]`); `t(key, lang, **fmt)`. New keys must be
  added to all 7 langs (draft fallback to `en` where not yet translated).
- Desired styling is single-message narrative; ⚠️🔴 emoji + "LIKELY FAKE" high-level label. The
  "LIKELY FAKE"-style label derives from verdict (DO_NOT_USE→"LIKELY FAKE/FAKE", CAUTION→"SUSPICIOUS",
  TRUST→"LIKELY GENUINE", UNABLE→"UNVERIFIED").
- The user-blocked todos (td: sudo mv, te: HF token) are **already resolved this session** — no longer
  part of scope.

---

## Proposed approach

Introduce a **`concerns_for(checks, target, verdict, lang)`** helper in `report.py` that:
1. Filters `checks` to those that "fired" (usable + above a per-channel threshold).
2. Maps each fired check to a *concern key* (see table below) and collects the unique set (max 3, in a
   fixed priority order).
3. Returns localized bullet strings. Build the reply as:
   `RISK LEVEL` → **verdict tile** (`label` + risk rating) → "It identifies N key concern(s):" →
   bullets → recommendation → (existing confidence_line/tip appended after).

### Concern mapping (thresholds mirror the existing fusions)
| Check / signal | Fires when | Concern key (English draft) |
|---|---|---|
| audio `aasist_detector`/`xlsr` `prob_deepfake` | > 0.5 | `concern_audio_ai` — "the voice shows signs of AI manipulation" |
| video `effort_face_forensics`/`frame_heuristics` `prob_deepfake` | > 0.5 | `concern_video_face` — "the face appears digitally altered" |
| cross_modal `prob_inconsistent` | > 0.5 | `concern_av_sync` — "the lips don't match the voice (audio/video mismatch)" |
| url `vt` `positives_ratio` | > 0.05 | `concern_url_flag` — "the link/domain has a poor security reputation" |
| url `host_string_score` | ≥ 0.5 | `concern_url_typo` — "the address looks like a spoof of a real site" |
| gov `prob_forged` | > 0.5 | `concern_doc_forged` — "the document shows signs of forgery" |
| (always for DO_NOT_USE unknown-source) | — | `concern_unverified_source` — "the claim could not be verified through reliable sources" |

### Impact of the changes

**Files:**
- Modify: `src/vishwas/report.py` — add `concerns_for`, produce tile + bullets + recommendation; keep
  the existing template-first ordering, prepending the new tile/bullets.
- Modify: `src/vishwas/i18n.py` — add keys: `verdict_tile_{label}`, `concern_count_line`,
  `concern_{audio_ai,video_face,av_sync,url_flag,url_typo,doc_forged,unverified_source}`,
  `recommend_line` (per verdict), plus maybe `reason_lead_{video,audio,image,url,document}`.
- Modify: `tests/test_08_i18n_report.py` — keep risk-line-first invariants; update any assertion that
  depends on the exact prior template string shape (strip-tolerant).
- Create: `tests/test_43_concern_bullets.py` — new coverage.

**Not changed:** fusion scoring, JSON API verdicts, CLI debug output, `CheckResult` schema, adapters.

---

## Step-by-step plan (TDD)

### Task 1: Add `concerns_for` + unit test (pure, hermetic)
**Objective:** Given a set of `CheckResult`s, return the localized concern bullets.
**Files:** Create `tests/test_43_concern_bullets.py`; modify `src/vishwas/report.py`.
1. **Write failing test** `test_43_concern_bullets.py::test_audio_high_yields_voice_bullet`:
   build a `CheckResult("aasist_detector","heavy","ok",{"prob_deepfake":0.97})` and a
   `CheckResult("effort_face_forensics","heavy","ok",{"prob_deepfake":0.1})`; call
   `concerns_for([...],"deepfake_video",Verdict.DO_NOT_USE)` and assert
   `"concern_audio_ai" in result` and `"concern_video_face" not in result`.
   Add a multi-check test asserting cap-at-3 + deterministic order, and a no-fire test returning `[]`.
2. Run: `python3 -m pytest tests/test_43_concern_bullets.py -q` → expected FAIL (`concerns_for` undefined).
3. **Implement** `concerns_for` in `report.py` (pure function; threshold table above;
   `from .i18n import t` reuse; return list of translated strings or keys).
4. Run to green; commit: `git add tests/test_43_concern_bullets.py src/vishwas/report.py && git commit -m "feat(report): concern-extraction layer for ⚠️ bullets"`.

### Task 2: Verdict tile + recommendation + i18n keys
**Objective:** Add the `🔴 LIKELY FAKE · High Risk` tile and the recommendation line, localized.
**Files:** modify `src/vishwas/report.py`, `src/vishwas/i18n.py`; extend `tests/test_43_concern_bullets.py`.
1. **Write failing test** `test_likely_fake_tile_and_recommend`:
   `ReportBuilder().build(..., verdict=do_not_use, confidence=0.8, lang="en", checks=[...])` →
   its text contains `LIKELY FAKE`, `High Risk`, and the `recommend_line` string (keep first line check).
2. Run → FAIL.
3. **Implement**: `_VERDICT_TILE = {DO_NOT_USE:("LIKELY FAKE","High"), CAUTION:("SUSPICIOUS","Medium"),
   TRUST:("LIKELY GENUINE","Low"), UNABLE:("UNVERIFIED","Unknown")}` + emoji per level (🔴🟠🟢⚪).
   Add i18n keys (all 7 langs; en filled, others draft-fallback to en). Append tile after the risk line
   and recommendation after bullets.
4. Run → green; commit.

### Task 3: Wire into `build()` (presentation, first-line invariant intact)
**Objective:** Make shipped replies use the new tile + bullets + recommendation while keeping
`RISK LEVEL:` first and templates as fallback.
**Files:** modify `src/vishwas/report.py`.
1. **Write test** `test_43::test_risk_line_still_first_across_verdicts` — loop 4 verdicts, assert
   `.splitlines()[0].startswith("RISK LEVEL:")` (reflects test_08 invariant).
2. Implement: in `build()`, after the risk line insert the tile; build concern line +
   bullets from `checks`; append recommendation; keep confidence_line + tip for completeness.
   Guard: concern bullets only for CAUTION/DO_NOT_USE (not TRUST/UNABLE, which keep current text).
3. Run full `tests/test_08_i18n_report.py` + `test_43` → green; commit.

### Task 4: Full hermetic suite + live verification
1. `env -u VISHWAS_AASIST_WEIGHTS -u VISHWAS_XLSRMAMBA_WEIGHTS PYTHONPATH=src python3 -m pytest tests/ -q`
   → recount (expect prior 373/13 ± new test). **HARD BAR:** green, no new failures, risk-line-first intact.
2. Live CLI smoke (secrets sourced) on: an AI video (expect 🔴 LIKELY FAKE + ⚠️ voice/face bullets),
   a benign URL (expect no 🔴, likely TRUST/CAUTION path), and a real synced video (expect 🟢/LOW or
   MEDIUM with correct bullets). Capture actual reply text.
3. Optional webhook HMAC-smoke (like `references/live-webhook-hmac-smoke.md`) to confirm the real
   WhatsApp reply path renders the new format.
4. Commit docs/FUSION_FINAL update + skill note (`references/reply-format-risk-level.md` update + add
   `email`-free concern-bullets note).

---

## Tests / validation summary
- `tests/test_43_concern_bullets.py` — concerns_for (single/multi/cap/order/empty) + tile+recommend +
  first-line invariant.
- `tests/test_08_i18n_report.py` — unchanged invariants (risk line first for all verdicts).
- Full hermetic suite + live CLI/webhook round-trips (AI video, benign URL, real video).

## Risks, tradeoffs, open questions
- **Overclaiming certainty**: the "LIKELY FAKE" label is a strong claim — must map ONLY from verdict
  (DO_NOT_USE), never from a lone signal; bullet thresholds are conservative (>0.5) and the
  recommendation stays "verify with a trusted source" for CAUTION/DO_NOT_USE.
- **i18n bloat**: adding ~9 keys × 7 langs; non-English get `en` fallback (draft) per the existing
  convention (do not block on native review — per user profile: English-only approval, no Indic review).
- **Emoji rendering on WhatsApp**: 🔴⚠️ are safe; avoid rare glyphs. Confirm the tile renders correctly
  in the live WhatsApp reply.
- **Back-compat of tests**: any existing test asserting exact prior `verdict_caution`/`do_not_use`
  sentence shape may need strip-tolerant updates — search `tests/` for those strings (Task 4 covers).
- **Open question**: should bullets appear for TRUST/UNVERIFIED? Default = NO (keep concise; only
  CAUTION/DO_NOT_USE get bullets). Changeable.

## Execution handoff
Plan complete and saved. Ready to execute via subagent-driven-development — dispatch a fresh subagent
per task (Task 1 pure/concurrent, then the rest sequentially since they share report.py/i18n.py), with
two-stage review (spec compliance then code quality). Shall I proceed?