# Vishwas i18n — Telugu (te) draft

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

| Key | Telugu text | Reviewer note |
|---|---|---|
| `greeting` | హాయ్! నేను మీకు ఎలా సహాయం చేయగలను? |  |
| `analyzing` | ఇప్పుడు దీనిని తనిఖీ చేస్తున్నాను. దయచేసి ఒక నిమిషం వేచి ఉండండి. |  |
| `verdict_trust` | మంచి సమాచారం: ఇది నిజమైనట్లు కనిపిస్తోంది. మార్పిడి లేదా మోసం గురించి ఏ లక్షణాలూ కనబడలేదు. |  |
| `verdict_caution` | జాగ్రత్త: కొన్ని వివరాలు అసాధారణంగా కనిపిస్తున్నాయి. మీరు ఆధికారిక మూలం దగ్గర స్వయంగా పరిశీలించే వరకు, డబ్బు పంపకూడదు లేదా వ్యక్తిగత సమాచారం భాగించకూడదు. [?] | long; consider splitting into two shorter sentences |
| `verdict_do_not_use` | హెచ్చరిక: ఇది మోసం లేదా హానికరమైనదిగా కనిపిస్తోంది. దీనిని తెరవకండి, ఏ లింక్‌నొక్కకండి, సందేశాన్ని తీసేయండి. మీరు ఇప్పటికే డబ్బు పంపிட்டట్లైతే, వెంటనే మీ బ్యాంకును సంప్రదించండి. [?] | 'తెరవకండి / నొక్కకండి' — confirm polite-imperative register for elders |
| `verdict_unable` | దీనిని ఇప్పుడు పూర్తిగా పరిశీలించలేకపోయాను. ఏదైనా చేయడానికి ముందు, ఆధికారిక వెబ్‌సైట్ లేదా యాప్ మీద స్వయంగా తనిఖీ చేసుకోండి. |  |
| `confidence_line` | నా నమ్మకం: %(conf)s. ఈ సాధనం సహాయపడుతుంది, కానీ ఒక మనిషి మళ్ళీ తనిఖీ చేయడం ఎల్లప్పుడూ మరింత సురక్షితం. [?] | 'నమ్మకం' (trust) used instead of a formal 'confidence' calque |
| `advice_avoid_links` | సలహు: తెలియని నంబర్‌ల నుండి వచ్చే లింక్‌లను ఎప్పుడూ నొక్కకూడదు. ప్రభుత్వ సంస్థలు WhatsApp లింక్‌లలో పాస్‌వర్డ్ అడుగుతాయి. |  |
| `progress_file` | ఫైల్ (%(name)s) ను స్కాన్ చేస్తున్నాను. |  |
| `progress_url` | లింక్‌ను విశ్లేషిస్తున్నాను… |  |
| `progress_media` | వీడియో/ఆడియోను వివరంగా అధ్యయనం చేస్తున్నాను. దీనికి కొంచెం సమయం పడుతుంది… |  |
| `evidence_missing` | అవసరమైన సేవ అందుబాటులో లేనందున కొన్ని తనిఖీలు దాటబడ్డాయి; పైన ఉన్న ఫలితం నేను నిజంగా పరీక్షించినదాన్ని మాత్రమే ప్రతిబింబిస్తుంది. [?] | formal compound; native simplification welcome |
| `heavy_pending_notice` | వేగవంతమైన తనిఖీ పూర్తయింది. నా లోతైన తనిఖీ ఇంకా జరుగుతోంది — అది పూర్తయినప్పుడు మరో సందేశం పంపుతాను. [?] | please confirm word order sounds spoken, not written |
| `heavy_followup` | నా లోతైన తనిఖీ (%(cap)s) పూర్తయింది. ఫలితం: %(verdict)s — నమ్మకం %(conf)s. |  |
