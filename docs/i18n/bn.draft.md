# VeriSafe i18n — Bengali (bn) draft

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

| Key | Bengali text | Reviewer note |
|---|---|---|
| `greeting` | হাই! আমি আপনার কীভাবে সাহায্য করতে পারি? |  |
| `analyzing` | এখন সেটা পরীক্ষা করছি। অনুগ্রহ করে একটু মিনিট অপেক্ষা করুন। |  |
| `verdict_trust` | ভালো খবর: এটা আসল মনে হচ্ছে। কোনো পরিবর্তন বা প্রতারণার লক্ষণ পাওয়া যায়নি। |  |
| `verdict_caution` | সতর্কতা: কিছু বিস্তারিত অস্বাভাবিক মনে হচ্ছে। আপনি নিজে সরকারি উৎস থেকে যাচাই করার আগে পর্যন্ত টাকা বা ব্যক্তিগত তথ্য শেয়ার করবেন না। [?] | long sentence; consider splitting for an elder |
| `verdict_do_not_use` | সতর্কবার্তা: এটা প্রতারণা বা ক্ষতিকর মনে হচ্ছে। এটা খুলবেন না, কোনো লিংকে চাপ দিবেন না, বার্তা মুছে ফেলুন। আপনি ইতিমধ্যে টাকা পাঠিয়েছেন, তাহলে সঙ্গে সঙ্গে আপনার ব্যাংকে ফোন করুন। [?] | check the politeness level of 'খুলবেন না / চাপ দিবেন না' chain |
| `verdict_unable` | আমি এখন এটা সম্পূর্ণভাবে যাচাই করতে পারিনি। যেকোনো কাজের আগে, সরকারি ওয়েবসাইটে বা আ্য্পে সরাসরি যাচাই করুন। |  |
| `confidence_line` | আমার আস্থা: %(conf)s। এই টুল সাহায্য করে, তবে মানুষের পুনরায় যাচাই সবসময় নিরাপদ। [?] | 'আস্থা' (trust) chosen over a calque of 'confidence'; confirm |
| `advice_avoid_links` | পরামর্শ: অচেনা নম্বর থেকে আসা লিংকে কখনো চাপবেন না। সরকারি প্রতিষ্ঠান WhatsApp লিংকে পাসওয়ার্ড চায় না। |  |
| `progress_file` | ফাইল (%(name)s) স্ক্যান করছি। |  |
| `progress_url` | লিংক বিশ্লেষণ করছি… |  |
| `progress_media` | ভিডিও/অডিও ভালো করে দেখছি। এর জন্য একটু সময় লাগবে… |  |
| `evidence_missing` | প্রয়োজনীয় সেবা পাওয়া যায়নি বলে কিছু পরীক্ষা করা হয়নি; উপরের সিদ্ধান্ত শুধু সে কথা নির্দেশ করে যা আমি সত্যিই পরীক্ষা করতে পেরেছি। [?] | a bit formal; welcome a plainer rephrase |
| `heavy_pending_notice` | দ্রুত পরীক্ষা শেষ হয়ে গেল। আমার গভীর পরীক্ষা এখনও চলছে — সেটা শেষ হলে আমি আপনাকে আবার একটি বার্তা পাঠাব। [?] | 'আবার একটি বার্তা পাঠাব' — verify it sounds like WhatsApp speech |
| `heavy_followup` | আমার গভীর পরীক্ষা (%(cap)s) শেষ হয়েছে। ফলাফল: %(verdict)s — আস্থা %(conf)s। |  |
