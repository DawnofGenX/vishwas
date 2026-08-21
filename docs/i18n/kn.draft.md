# VeriSafe i18n — Kannada (kn) draft

> **DRAFT — pending native review.** Do NOT merge into `i18n._DEFAULTS`
> until a native reviewer (the DawnofGenX gate) signs off or corrects each line.
>
> printf-style placeholders are preserved **verbatim** and must never be
> translated, altered, or reordered: `%(conf)s`, `%(name)s`, `%(cap)s`, `%(verdict)s`.
>
> Audience: a ~65-year-old non-technical elder. Warm, plain, respectful;
> short sentences; no tech jargon. Any line ending in `[?]` is a best-effort
> attempt with sub-90% confidence — please double-check it specifically.
>
> Source corpus: `docs/i18n/en.md` (14 keys). This draft covers all of them.

| Key | Kannada text | Reviewer note |
|---|---|---|
| `greeting` | ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ? [?] | polite-plural form chosen; some elders expect 'ನಮಸ್ಕಾರ' with 'ನೀವು' — verify |
| `analyzing` | ಈಗ ಅದನ್ನು ಪರೀಕ್ಷಿಸುತ್ತಿದ್ದೇನೆ. ಕೆಲವು ನಿಮಿಷ ಕಾಯಿರಿ. |  |
| `verdict_trust` | ಸರಿ ಸುದ್ದಿ: ಇದು ನೈಜವಾಗಿ ಕಾಣುತ್ತಿದೆ. ಬದಲಾವಣೆಯ ಅಥವಾ ಮೋಸದ ಯಾವ ಚಿಹ್ನೆಯೂ ಕಂಡುಬಂದಿಲ್ಲ. |  |
| `verdict_caution` | ಎಚ್ಚರಿಕೆ: ಕೆಲವು ವಿವರಗಳು ಅಸಾಮಾನ್ಯವಾಗಿ ಕಾಣುತ್ತಿವೆ. ನೀವು ಆಧಿಕಾರಿತಮೂಲದಿಂದ ಸ್ವಯಂ ಪರಿಶೀಲಿಸುವ ವರೆಗೆ, ಹಣ ಅಥವಾ ವ್ಯಕ್ತಿಗತ ಮಾಹಿತಿ ಹಂಚಬೇಡಿ. [?] | long sentence; consider splitting |
| `verdict_do_not_use` | ಎಚ್ಚರಿಕೆ: ಇದು ಮೋಸ ಅಥವಾ ಹಾನಿಕರವಾದದ್ದಿನಂತೆ ಕಾಣುತ್ತಿದೆ. ಇದನ್ನು ತೆರೆಯಬೇಡಿ, ಲಿಂಕ್‌ಗಳನ್ನು ಒತ್ತಬೇಡಿ, ಸಂದೇಶವನ್ನು ಅಳಿಸಿ. ನೀವು ಈಗಾಗಲೇ ಹಣ ಕಳುಹಿಸಿದರೆ, ತಕ್ಷಣ ಬ್ಯಾಂಕಿಗೆ ಕರೆ ಮಾಡಿ. [?] | borrowed 'ಲಿಂಕ್' kept (widely understood); verify verb chain |
| `verdict_unable` | ಇದನ್ನು ಈಗ ಪೂರ್ಣವಾಗಿ ಪರೀಕ್ಷಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಯಾವುದನ್ನಾದರೂ ಮಾಡುವ ಮೊದಲು, ಆಧಿಕಾರಿತಮೂಲದ ವೆಬ್‌ಸೈಟ್ ಅಥವಾ ಆ್ಯಪ್‌ನಲ್ಲಿ ನೇರವಾಗಿ ಪರೀಕ್ಷಿಸಿ. [?] | slightly stiff; native smoothing welcome |
| `confidence_line` | ನನ್ನ ನಂಬಿಕೆ: %(conf)s. ಈ ಸಾಧನ ಸಹಾಯ ಮಾಡುತ್ತದೆ, ಆದರೆ ಜನರ ಮರು ಪರಿಶೀಲನೆ ಯಾವಾಗಲೂ ಕൂಡೂ ಸುರಕ್ಷಿತ. |  |
| `advice_avoid_links` | ಸೂಚನೆ: ಅಪರಿಚಿತ ಸಂಖ್ಯೆಗಳಿಂದ ಬರುವ ಲಿಂಕ್‌ಗಳನ್ನು ಎಂದಿಗೂ ಒತ್ತಬೇಡಿ. ಸರ್ಕಾರಿ ಸಂಸ್ಥೆಗಳು WhatsApp ಲಿಂಕ್‌ನಲ್ಲಿ ಪಾಸ್‌ವರ್ಡ್ ಕೇಳುವುದಿಲ್ಲ. |  |
| `progress_file` | ಫೈಲ್ (%(name)s) ಸ್ಕಾನ್ ಮಾಡುತ್ತಿದ್ದೇನೆ. |  |
| `progress_url` | ಲಿಂಕ್ ವಿಶ್ಲೇಷಿಸುತ್ತಿದ್ದೇನೆ… |  |
| `progress_media` | ವೀಡಿಯೊ/ಆಡಿಯೊ ವಿಸ್ತಾರವಾಗಿ ಅಧ್ಯಯನ ಮಾಡುತ್ತಿದ್ದೇನೆ. ಇದಕ್ಕೆ ಕೆಲವು ಸಮಯ ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ… [?] | phrasing check needed |
| `evidence_missing` | ಅಗತ್ಯ ಸೇವೆ ಲಭ್ಯವಿರದ ಕಾರಣ ಕೆಲವು ಪರೀಕ್ಷೆಗಳು ಹಾದುಹೋದವು; ಮೇಲಿನ ಫಲಿತಾಂಶ ನಾನು ನಿಜವಾಗಿ ಪರೀಕ್ಷಿಸಿದದ್ದನ್ನು ಮಾತ್ರ ಪ್ರತಿಬಿಂಬಿಸುತ್ತದೆ. [?] | formal; needs simplification for an elder |
| `heavy_pending_notice` | ವೇಗದ ಪರೀಕ್ಷೆ ಮುಗಿದಿದೆ. ನನ್ನ ಆಳ ಪರೀಕ್ಷೆ ಇನ್ನೂ ನಡೆಯುತ್ತಿದೆ — ಅದು ಮುಗಿದಾಗ ನಿಮಗೆ ಮತ್ತೊಂದು ಸಂದೇಶ ಕಳುಹಿಸುತ್ತೇನೆ. [?] | 'ಇನ್ನೂ ನಡೆಯುತ್ತಿದೆ' — confirm natural spoken form |
| `heavy_followup` | ನನ್ನ ಆಳ ಪರೀಕ್ಷೆ (%(cap)s) ಮುಗಿದಿದೆ. ಫಲಿತಾಂಶ: %(verdict)s — ನಂಬಿಕೆ %(conf)s. |  |
