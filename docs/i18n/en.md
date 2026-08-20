# VeriSafe i18n — English corpus (machine-generated)

> **Auto-generated** by `scripts/i18n_export.py` from the live
> `src/verisafe/i18n.py` module (`_DEFAULTS`). Do not hand-edit —
> re-run the script after any change to the module.
>
> English is the authoritative base language. Every other language
> draft in this directory must cover exactly this key set.
>
> Corpus size: **14 keys** (12 in module, 2 pre-drafted for Task 3.2).

| Key | English text | Placeholders | Source |
|---|---|---|---|
| `greeting` | Hi, how can I help you? | — | in module |
| `analyzing` | Checking it now. Please wait about a minute. | — | in module |
| `verdict_trust` | Good news: this looks genuine. I found no signs of tampering or fraud. | — | in module |
| `verdict_caution` | Caution: some details look unusual. Do not pay money or share personal information until you confirm with the official source yourself. | — | in module |
| `verdict_do_not_use` | Warning: this looks like a scam or harmful item. Do not open it, do not press any links, delete the message. If you already sent money, call your bank right away. | — | in module |
| `verdict_unable` | I could not fully verify this right now. Please check it directly on the official website or app before doing anything. | — | in module |
| `confidence_line` | My confidence: %(conf)s. This tool helps, but a human double-check is always safer. | %(conf)s | in module |
| `advice_avoid_links` | Tip: never click links from unknown numbers. Official institutions never ask for passwords on WhatsApp links. | — | in module |
| `progress_file` | Scanning the file (%(name)s)… | %(name)s | in module |
| `progress_url` | Analysing the link… | — | in module |
| `progress_media` | Studying the video/audio in detail. This takes a little longer… | — | in module |
| `evidence_missing` | Some checks were skipped because a required service was unavailable; the verdict above reflects only what I could actually test. | — | in module |
| `heavy_pending_notice` | The quick check is done. My deeper check is still running — I'll send you an update when it finishes. | — | pre-drafted (Task 3.2) |
| `heavy_followup` | My deeper check (%(cap)s) finished. Result: %(verdict)s — confidence %(conf)s. | %(cap)s, %(conf)s, %(verdict)s | pre-drafted (Task 3.2) |
