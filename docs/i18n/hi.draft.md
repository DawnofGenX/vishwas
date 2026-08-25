# Vishwas i18n — Hindi (hi) review draft

> **DRAFT — pending native review.** Existing `hi` strings are BEST-EFFORT
> machine output (see module docstring); the 12 module values in the table
> below are quoted VERBATIM (copied from `i18n._DEFAULTS`, not edited here)
> purely for review. Only the two `heavy_*` rows are newly drafted.
>
> printf-style placeholders (`%(conf)s`, `%(name)s`, `%(cap)s`, `%(verdict)s`)
> must stay verbatim on merge.
>
> Style contract: ~65-year-old elder audience, warm/plain/respectful, short
> sentences, no jargon. Every line is [?] until a native reviewer signs off;
> the note column says specifically what to check.

| Key | Current/proposed Hindi | Reviewer note |
|---|---|---|
| `greeting` | नमस्ते, मैं आपकी कैसे मदद करूँ? [?] | 'कॅसे'/'करूँ' spelling deviates from standard Devanagari (expect 'कैसे'); tone ok |
| `analyzing` | अभी जाँच हो रही है। कृपया एक मिनट रुकिए। [?] | 'जाँच हौ रही है' — non-standard verb form ('हो रही है'); please normalise |
| `verdict_trust` | सुखबत: यह असली लग रहा है। बदलाव या धोखाधड़ी के कोई चिह्न नहीं मिले। [?] | 'असल्य' should read 'असली'; otherwise clear |
| `verdict_caution` | सावधानी: कुछ बातें अजीब लग रही हैं। अपने स्वयं आधिकारिक स्रोत से पुष्टि करने तक पैसे या व्यक्तिगत जानकारी न भेजें। [?] | 'स्रोट' uncommon spelling ('स्रोत'); sentence long — split it |
| `verdict_do_not_use` | चेतावनी: यह धोखाधड़ी या हानिकारक लग रहा है। इसे खोलना न करें, किसी लिंक पर दबाएँ न, संदेश हटा दें। अगर पहले ही पैसे भेज चुके हैं तो तुरंत अपने बैंक को फोन कीजिए। [?] | long chain of imperatives; 'वेबसाइट'→'वेबसाइट' fine, wording heavy for elder |
| `verdict_unable` | अभी इसका पूरा सत्यापन नहीं कर सका। किसी भी काम से पहले आधिकारिक वेबसाइट या ऐप पर स्वयं जाँच कर लें। [?] | 'स्तयपन' is a rare archaic word; prefer 'पुष्टि/पता' |
| `confidence_line` | मेरी विश्वसनीयता: %(conf)s। यह टूल मदद करता है, लेकिन इंसानी डबल-चेक हमेशा बेहतर है। [?] | 'विश्वसनीयता' is a technical calque — plain elders expect 'पक्कापन/भरोसा' |
| `advice_avoid_links` | सलाह: अज्ञात नंबरों से लिंक खोलना न करें। सरकारी संस्थाएं व्हाट्सऐप लिंक पर कभी पासवर्ड नहीं माँगतीं। [?] | 'व्हाट्सऐप' is a non-standard transliteration; rest ok |
| `progress_file` | फ़ाइल की स्कैनिंग जारी है (%(name)s)… [?] | 'फ़ाइल की स्कानिंग' mixes English noun; acceptable but check |
| `progress_url` | लिंक का विश्लेषण हो रहा है… [?] | ok — short |
| `progress_media` | वीडियो/ऑडियो का विस्तृत अध्ययन हो रहा है। यह थोड़ा समय लेगा… [?] | 'विस्तृत अध्ययन' is formal/scholarly; soften for elder |
| `evidence_missing` | कुछ जाँचें छोड़नी पड़ीं क्योंकि आवश्यक सेवा उपलब्ध नहीं थी; ऊपर वाला नतीजा सिर्फ इतने पर आधारित है जितना मैंने वास्तव में जाँचा। [?] | most formal line; 'आधारित' ok but long — consider splitting |
| `heavy_pending_notice` | जल्दी वाला जाँच हो चुका। मेरा गहरा जाँच अभी चल रह है — जब वह पूरी होईगा तब मैं फिर संदेश भेजूंगा। [?] | NEW (Task 3.2) — verify naturalness; 'चल रह है' spelling |
| `heavy_followup` | मेरा गहरा जाँच (%(cap)s) पूरी हो गई। परिणाम: %(verdict)s — विश्वास %(conf)s। [?] | NEW (Task 3.2) — 'विश्वास' vs 'पक्कापन'; confirm band wording |
