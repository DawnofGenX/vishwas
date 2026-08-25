# Vishwas i18n — Malayalam (ml) draft

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

| Key | Malayalam text | Reviewer note |
|---|---|---|
| `greeting` | ഹായ്! ഞാൻ നിങ്ങൾക്ക് എങ്ങനെ സഹായിക്കാം? |  |
| `analyzing` | ഇപ്പോൾ അത് പരിശോധിക്കുന്നുണ്ട്. ഒരു മിനിറ്റ് കാത്തിരിക്കൂ. |  |
| `verdict_trust` | നല്ല വാർത്ത: ഇത് യഥാർത്ഥമാണെന്ന് തോന്നുന്നു. മാറ്റമോ തട്ടിപ്പോ — അത്തരം ചിഹ്നങ്ങൾ ഒന്നുമില്ല. |  |
| `verdict_caution` | ശ്രദ്ധിക്കുക: ചില വിശദാംശങ്ങൾ അസാധാരണമായി തോന്നുന്നു. നിങ്ങൾ ആധികാരിക ഉറവിടത്തിൽ നിന്ന് സ്വയം പരിശോധിച്ചു തീർക്കുന്നതു വരെ, പണം അയക്കരുത്, വ്യക്തിഗത വിവരങ്ങളും പങ്കുവയ്ക്കരുത്. [?] | long; consider splitting |
| `verdict_do_not_use` | അറിയിപ്പ്: ഇത് തട്ടിപ്പോ ദോഷം തീരുമുള്ള ഒന്നോ എന്ന് തോന്നുന്നു. ഇത് തുറക്കരുത്, ലിങ്കുകളിൽ അമർത്തരുത്, സന്ദേശം നീക്കം ചെയ്യുക. നേരത്തെ പണം അയച്ചിട്ടുണ്ടെങ്കിൽ, ഉടൻ ബാങ്കിനെ ബന്ധപ്പെടുക. [?] | check the negative-imperative chain for natural spoken form |
| `verdict_unable` | ഇത് ഇപ്പോൾ പൂർണ്ണമായി പരിശോധിക്കാനായില്ല. എന്തെങ്കിലും ചെയ്യുന്നതിനായി, ആധികാരിക വെബ്‌സൈറ്റിലോ ആപ്പിലോ നേരിട്ട് പരിശോധിക്കുക. |  |
| `confidence_line` | എന്റെ വിശ്വാസം: %(conf)s. ഈ ഉപകരണം സഹായിക്കുന്നു, പക്ഷേ മനുഷ്യന്റെ രണ്ടാം പരിശോധന എപ്പോഴും കൂടുതൽ സുരക്ഷിതമാണ്. [?] | 'വിശ്വാസം' = trust/confidence; confirm it fits an elder's ear |
| `advice_avoid_links` | വേഗം: അജ്ഞാത നമ്പറുകളിൽ നിന്ന് വരുന്ന ലിങ്കുകൾ ഒരിക്കലും അമർത്തരുത്. സർക്കാർ സ്ഥാപനങ്ങൾ WhatsApp ലിങ്കിൽ പാസ്‌വേഡ് ചോദിക്കില്ല. |  |
| `progress_file` | ഫയൽ (%(name)s) സ്കാൻ ചെയ്യുന്നു. |  |
| `progress_url` | ലിങ്ക് വിശകലനം ചെയ്യുന്നു… |  |
| `progress_media` | വീഡിയോ/ഓഡിയോ വിശദമായി പഠിക്കുന്നു. ഇതിന് കുറച്ച് സമയം വാങ്ങും… |  |
| `evidence_missing` | ആവശ്യമായ സേവനം ലഭ്യമായില്ലെന്ന് കാരണം ചില പരിശോധനകൾ ഒഴിവാക്കി; മുകളിലെ നിരണ്ണയം ഞാൻ യഥാർത്ഥത്തിൽ പരിശോധിച്ചത് മാത്രം പ്രതിഫലിപ്പിക്കുന്നു. [?] | formal; a plainer rephrase is welcome |
| `heavy_pending_notice` | വേഗത്തിലുള്ള പരിശോധന അവസാനിച്ചു. എന്റെ ആഴത്തിലുള്ള പരിശോധന ഇപ്പോഴും നടക്കുന്നുണ്ട് — അത് അവസാനിച്ചാൽ ഞാൻ നിങ്ങൾക്ക് വീണ്ടും അറിയിക്കാം. [?] | confirm word order; 'അറിയിക്കാം' feels right but please verify |
| `heavy_followup` | എന്റെ ആഴത്തിലുള്ള പരിശോധന (%(cap)s) അവസാനിച്ചു. ഫലം: %(verdict)s — വിശ്വാസം %(conf)s. |  |
