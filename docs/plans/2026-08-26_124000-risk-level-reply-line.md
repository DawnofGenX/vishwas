# Risk-Level Line in WhatsApp Replies — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Prepend a machine-assigned risk level ("RISK LEVEL: HIGH / MEDIUM / LOW") as the first line of every WhatsApp reply, derived deterministically from the existing verdict — verdict vocabulary, JSON API and CLI debug output stay unchanged.

**Architecture:** Pure presentation-layer change inside `ReportBuilder.build()`: map Verdict→risk token (TRUST→LOW, CAUTION→MEDIUM, DO_NOT_USE→HIGH, UNABLE→UNVERIFIED), add one i18n string per language (7 langs), prepend via the existing `parts` list so ordering stays template-first. No fusion/router/orchestrator changes; zero new dependencies.

**Tech Stack:** stdlib only; pytest (existing suites test_08_i18n_report.py = 15 tests + webhook reply tests are the regression net).

---

## Current context / assumptions

- HEAD `6aff500`; suite baseline 365 passed / 7 skipped hermetic (NOTE: with VISHWAS_* weight env leaked into shell, test_34 runs weighted variants — run suite WITHOUT sourcing secrets env for the hermetic gate).
- Reply assembly is centralized: `src/vishwas/report.py::ReportBuilder.build()` builds `parts[]` → joined into UserReport.text. WhatsApp channel (`app.py` webhook) sends `UserReport.text` verbatim.
- i18n: `src/vishwas/i18n.py::_S` dict keyed `"verdict_trust"` etc., 7 languages (en hi ta te ml kn bn); `t(key, lang)` falls back to en. Pattern strings ship en+hi only by design.
- Verdict enum: TRUST / CAUTION / DO_NOT_USE / UNABLE_TO_VERIFY (fusion.Verdict).
- UNABLE replies currently carry NO confidence line (deliberate UX fix) — the risk line must still appear for UNABLE but read "UNVERIFIED", not a number.

## Proposed approach

TDD: failing tests first in test_08_i18n_report.py (assert first line of built report matches expected risk token per verdict), then minimal impl in report.py + i18n.py. One commit.

## Step-by-step plan

### Task 1: Failing tests
**Files:** Modify `tests/test_08_i18n_report.py`
**Step 1:** Add 4 tests:
```python
def test_risk_line_first_high():
    r = ReportBuilder().build(target="deepfake_video", verdict=Verdict.DO_NOT_USE,
                              confidence=0.8, reasons=[], checks=[], lang="en")
    assert r.text.splitlines()[0].startswith("RISK LEVEL: HIGH")

def test_risk_line_medium(): ...CAUTION... "RISK LEVEL: MEDIUM"
def test_risk_line_low():    ...TRUST...    "RISK LEVEL: LOW"
def test_risk_line_unverified(): ...UNABLE_TO_VERIFY... "RISK LEVEL: UNVERIFIED"
```
(Import style copied from file's existing imports.)
**Step 2:** Run `PYTHONPATH=src python3 -m pytest tests/test_08_i18n_report.py -q` → expect exactly 4 FAIL.

### Task 2: i18n strings
**Files:** Modify `src/vishwas/i18n.py`
**Step 1:** Add key `"risk_line"` with `"%(level)s"` placeholder, en first:
```python
"risk_line": {
    "en": "RISK LEVEL: %(level)s",
    "hi": "जोखिम स्तर: %(level)s",
    # ta/te/ml/kn/bn drafts follow docs/i18n/*.draft.md pattern
},
```
Draft Indic strings acceptable (user cannot review scripts; back-translation QA exists separately). English levels stay untranslated tokens (HIGH/MEDIUM/LOW/UNVERIFIED) — language-neutral, matches gap-token convention.
**Step 2:** Run i18n-related tests → green (new key breaks nothing).

### Task 3: ReportBuilder change
**Files:** Modify `src/vishwas/report.py` (build(), ~line 41)
**Step 1:** Add module-level map + prepend:
```python
_RISK_OF_VERDICT = {
    Verdict.TRUST: "LOW",
    Verdict.CAUTION: "MEDIUM",
    Verdict.DO_NOT_USE: "HIGH",
    Verdict.UNABLE_TO_VERIFY: "UNVERIFIED",
}
```
In build(): insert as FIRST part before `parts = [t(key, lang)]`:
```python
parts = [t("risk_line", lang, level=_RISK_OF_VERDICT[verdict]), t(key, lang)]
```
**Step 2:** Run task-1 tests → 4 PASS.

### Task 4: Full gates + commit
**Step 1:** Full hermetic suite (no secrets env): expect **369+ passed** (365 + 4 new), 7 skipped, ZERO failures of pre-existing tests (some existing report tests may need updating IF they assert exact full-text equality — check failures case-by-case; prefer updating their expected text to include the new first line rather than loosening assertions).
**Step 2:** CLI smoke: `bash scripts/run_vishwas.sh cli --text "https://www.wikipedia.org" | head -5` → reply begins "RISK LEVEL: ..." followed by verdict sentence.
**Step 3:** Restart webhook `systemctl --user restart vishwas-webhook`, poll `/health` ok; live leg is operator's next WhatsApp message (any verdict now leads with the risk line).
**Step 4:** Commit: `feat(ux): lead WhatsApp replies with deterministic RISK LEVEL line`.

## Files likely to change
- `tests/test_08_i18n_report.py` (+4 tests, maybe 1-2 expectation updates)
- `src/vishwas/i18n.py` (+1 key × 7 langs)
- `src/vishwas/report.py` (+map, +1 prepend line)

## Tests / validation
- New unit tests (4). Full hermetic suite green. CLI smoke shows the line. Webhook health ok after restart.

## Risks / tradeoffs / open questions
- Exact-equality assertions elsewhere (webhook e2e fixtures?) may break → update expected strings, don't weaken checks.
- LLM narration path appends AFTER templates — unaffected.
- Open Q: should UNABLE say "UNVERIFIED" (chosen) or be omitted? Chosen: show it — consistent shape helps users parse replies at a glance.
