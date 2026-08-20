#!/usr/bin/env python3
"""Machine-generate docs/i18n/en.md from the live i18n module.

Reads src/verisafe/i18n.py PROGRAMMATICALLY (import, never regex-scraping the
source) and writes a clean markdown table of every key present in
``i18n._DEFAULTS`` (each key maps to per-language strings; the English value
is the authoritative corpus row), PLUS the two Task-3.2 follow-up keys
(``heavy_pending_notice``, ``heavy_followup``) which are pre-drafted here as
English bases until the human native-review gate lands them in the module.

Usage:
    python3 scripts/i18n_export.py            # writes <repo>/docs/i18n/en.md

The output file is machine-generated; do not hand-edit it. Re-run after any
change to ``i18n._DEFAULTS`` (e.g. after a reviewed translation merge) to
refresh the corpus.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import verisafe.i18n as i18n_mod  # noqa: E402

# Task 3.2: new user-facing surfaces (Phase 2 orchestrator) pre-drafted into
# the corpus. Deliberately NOT merged into i18n._DEFAULTS here — that is the
# human native-review gate's job (plan Task 3.1 step 4). The exporter simply
# makes sure they always appear in en.md so drafts can be built against them.
NEW_FOLLOWUP_EN: dict[str, str] = {
    "heavy_pending_notice": (
        "The quick check is done. My deeper check is still running — "
        "I'll send you an update when it finishes."
    ),
    "heavy_followup": (
        "My deeper check (%(cap)s) finished. Result: %(verdict)s — "
        "confidence %(conf)s."
    ),
}

_PLACEHOLDER_RE = re.compile(r"%\((\w+)\)s")


def _placeholders(s: str) -> str:
    found = _PLACEHOLDER_RE.findall(s)
    return ", ".join("%(" + p + ")s" for p in sorted(set(found))) or "—"


def build_en_rows() -> list[tuple[str, str, str]]:
    """Return (key, english_text, origin) for the full corpus."""
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for key in i18n_mod._DEFAULTS:  # preserves the module's own definition order
        table = i18n_mod._DEFAULTS[key]
        en = table.get(i18n_mod._DEFAULT_LANG_FALLBACK, "")
        if isinstance(en, str):
            rows.append((key, en, "in module"))
            seen.add(key)
    for key, en in NEW_FOLLOWUP_EN.items():
        if key not in seen:
            # Pre-drafted base; lands in _DEFAULTS only after the review gate.
            rows.append((key, en, "pre-drafted (Task 3.2)"))
        else:
            # Already merged into the module: trust live text at its original
            # position (already in `rows`).
            pass
    return rows


def main() -> int:
    rows = build_en_rows()
    out_path = ROOT / "docs" / "i18n" / "en.md"
    n_mod = sum(1 for _, _, o in rows if o == "in module")
    n_new = len(rows) - n_mod
    lines: list[str] = [
        "# VeriSafe i18n — English corpus (machine-generated)",
        "",
        "> **Auto-generated** by `scripts/i18n_export.py` from the live",
        "> `src/verisafe/i18n.py` module (`_DEFAULTS`). Do not hand-edit —",
        "> re-run the script after any change to the module.",
        ">",
        "> English is the authoritative base language. Every other language",
        "> draft in this directory must cover exactly this key set.",
        ">",
        f"> Corpus size: **{len(rows)} keys** ({n_mod} in module, {n_new} pre-drafted for Task 3.2).",
        "",
        "| Key | English text | Placeholders | Source |",
        "|---|---|---|---|",
    ]
    for key, text, origin in rows:
        esc = text.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{key}` | {esc} | {_placeholders(text)} | {origin} |")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    module_keys = set(i18n_mod._DEFAULTS)
    corpus_keys = {k for k, _, _ in rows}
    assert corpus_keys >= module_keys, "corpus lost a module key!"
    print(f"wrote {out_path} ({len(rows)} keys; module carries {len(module_keys)})")
    for k, _, o in rows:
        print(f"  {k:<22} [{o}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
