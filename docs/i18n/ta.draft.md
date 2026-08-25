# Vishwas i18n — Tamil (ta) draft

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

| Key | Tamil text | Reviewer note |
|---|---|---|
| `greeting` | வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்? |  |
| `analyzing` | இதை இப்போது ஆய்வு செய்து கொண்டிருக்கிறேன். ஒரு நிமிடம் காத்திருங்கள். |  |
| `verdict_trust` | நல்ல செய்தி: இது உண்மையானதாகத் தெரிகிறது. மாற்றம் அல்லது ஏமாற்றத்தின் சிறு குறியே இல்லை. |  |
| `verdict_caution` | எச்சரிக்கை: சில விவரங்கள் அசாதாரணமாகத் தெரிகின்றன. நீங்களே அதிகாரப்பூர்வ மூலத்தில் கேட்டுப் பார்த்துவிடுவதற்குள், பணத்தை அனுப்பவும் தனிப்பட்ட தகவலைப் பகிரவும் வேண்டாம். [?] | longest sentence in the file; split or simplify if too much for an elder |
| `verdict_do_not_use` | எச்சரிக்கை: இது ஏமாற்று அல்லது கேடு விளைவிப்பது போல் தெரிகிறது. அதைத் திறக்கவும், எந்த இணைப்பையும் அழுத்தவும் வேண்டாம்; செய்தியை நீக்குங்கள். ஏற்கனவே பணம் அனுப்பியிருந்தால், உடனே உங்கள் வங்கிக்கு தொலைபாஷை செய்யுங்கள். [?] | four imperative clauses; confirm the verb forms sound natural |
| `verdict_unable` | இதை இப்போது முழுமையாகச் சரிபார்க்க முடியவில்லை. எதாவது செய்வதற்கு முன், அதிகாரப்பூர்வ இணையதளத்தில் அல்லது பயன்பாட்டில் நேரடியாகச் சரிபார்க்குங்கள். |  |
| `confidence_line` | என் நம்பிக்கை: %(conf)s. இந்தக் கருவி உதவுகிறது; ஆனால் ஒரு மனிதர் மீண்டும் சரிபார்ப்பது எப்போதும் பாதுகாப்பானது. [?] | 'நம்பிக்கை' (trust/confidence) chosen over a calque of 'confidence' |
| `advice_avoid_links` | குறிப்பு: தெரியாத எண்களிலிருந்து வரும் இணைப்புகளை எப்போதும் அழுத்த வேண்டாம். அரச அமைப்புகள் WhatsApp இணைப்பில் கடவுச்சொல் கேட்பதில்லை. |  |
| `progress_file` | கோப்பு (%(name)s) ஐச் சோதித்து வருகிறேன். |  |
| `progress_url` | இந்த இணைப்பை ஆய்வு செய்கிறேன்… |  |
| `progress_media` | விடியோ/ஆடியோவை விரிவாக ஆய்வு செய்கிறேன். இதற்கு ஒரு சிறிது நேரம் ஆகும்… [?] | video/audio word order — check ellipsis and phrasing |
| `evidence_missing` | தேவையான சேவை கிடைக்காமல் இருந்ததால் சில சோதனைகள் தவிர்க்கப்பட்டன; மேலே உள்ள முடிவு, நான் மெய்யில் சோதித்ததை மட்டுமே காட்டுகிறது. [?] | slightly formal; consider 'செயலாக்க முடியவில்லை'-style phrasing |
| `heavy_pending_notice` | விரைவான சோதனை முடிந்தது. எனது ஆழமான சோதனை இன்னும் நடந்து கொண்டிருக்கிறது — அது முடிந்தவுடன் உங்களுக்கு மீண்டும் செய்தி அனுப்புவேன். [?] | word order tuned for readability; please confirm |
| `heavy_followup` | எனது ஆழமான சோதனை (%(cap)s) முடிந்தது. முடிவு: %(verdict)s — நம்பிக்கை %(conf)s. |  |
