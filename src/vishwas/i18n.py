"""Minimal i18n layer for user-facing strings (WhatsApp replies).

Design rule for non-technical elders: ONE short verdict sentence, ONE action
sentence. Detector jargon never reaches the user.

Status (post-roadmap 2026-08-21, per .hermes/plans/2026-08-20_222933-vishwas-full-roadmap.md
Phase 3, merged under the autonomy default — native review still recommended):
  * ``en`` — authoritative.
  * ``hi`` — best-effort machine output; native review PENDING. The two
    Task-3.2 follow-up keys are newly drafted for hi (review pending); the
    original twelve hi strings are untouched by the roadmap cycle.
  * ``ta te ml kn bn`` — drafted in docs/i18n/*.draft.md, merged into
    _DEFAULTS as DRAFT state; lines marked [?] in the drafts carry
    sub-90% confidence (see docs/GAPS_AND_ENABLEMENT.md for the list).
    No key falls back silently: every language renders its own string.
load_custom_strings() can overlay corrected/reviewed translations from a JSON
file without code changes — use that path when native reviewers arrive.
"""
from __future__ import annotations

import json

from pathlib import Path


_SUPPORTED = ("en", "hi", "ta", "te", "ml", "kn", "bn")
_DEFAULT_LANG_FALLBACK = "en"

_DEFAULTS: dict[str, dict[str, str]] = {
    "greeting": {
        "en": "Hi, how can I help you?",
        "hi": "नमस्ते, मैं आपकी कैसे मदद करूँ?",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        # draft (te); see docs/i18n/te.draft.md
        "te": "హాయ్! నేను మీకు ఎలా సహాయం చేయగలను?",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "ഹായ്! ഞാൻ നിങ്ങൾക്ക് എങ്ങനെ സഹായിക്കാം?",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ?",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "হাই! আমি আপনার কীভাবে সাহায্য করতে পারি?",
    },
    "analyzing": {
        "en": "Checking it now. Please wait about a minute.",
        "hi": "अभी जाँच हो रही है। कृपया एक मिनट रुकिए।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "இதை இப்போது ஆய்வு செய்து கொண்டிருக்கிறேன். ஒரு நிமிடம் காத்திருங்கள்.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "ఇప్పుడు దీనిని తనిఖీ చేస్తున్నాను. దయచేసి ఒక నిమిషం వేచి ఉండండి.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "ഇപ്പോൾ അത് പരിശോധിക്കുന്നുണ്ട്. ഒരു മിനിറ്റ് കാത്തിരിക്കൂ.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಈಗ ಅದನ್ನು ಪರೀಕ್ಷಿಸುತ್ತಿದ್ದೇನೆ. ಕೆಲವು ನಿಮಿಷ ಕಾಯಿರಿ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "এখন সেটা পরীক্ষা করছি। অনুগ্রহ করে একটু মিনিট অপেক্ষা করুন।",
    },
    # UX 2026-08-26: every reply LEADS with a deterministic risk level derived
    # from the verdict (report.py _RISK_OF_VERDICT); levels stay English tokens
    # across languages (language-neutral, same convention as gap tokens).
    "risk_line": {
        "en": "RISK LEVEL: %(level)s",
        "hi": "जोखिम स्तर: %(level)s",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "ஆபத்து நிலை: %(level)s",
        # draft (te); see docs/i18n/te.draft.md
        "te": "ప్రమాద స్థాయి: %(level)s",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "അപകട നില: %(level)s",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಅಪಾಯದ ಮಟ್ಟ: %(level)s",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "ঝুঁকির মাত্রা: %(level)s",
    },
    "verdict_trust": {
        "en": "Good news: this looks genuine. I found no signs of tampering or fraud.",
        "hi": "सुखबत: यह असली लग रहा है। बदलाव या धोखाधड़ी के कोई चिह्न नहीं मिले।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "நல்ல செய்தி: இது உண்மையானதாகத் தெரிகிறது. மாற்றம் அல்லது ஏமாற்றத்தின் சிறு குறியே இல்லை.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "మంచి సమాచారం: ఇది నిజమైనట్లు కనిపిస్తోంది. మార్పిడి లేదా మోసం గురించి ఏ లక్షణాలూ కనబడలేదు.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "നല്ല വാർത്ത: ഇത് യഥാർത്ഥമാണെന്ന് തോന്നുന്നു. മാറ്റമോ തട്ടിപ്പോ — അത്തരം ചിഹ്നങ്ങൾ ഒന്നുമില്ല.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಸರಿ ಸುದ್ದಿ: ಇದು ನೈಜವಾಗಿ ಕಾಣುತ್ತಿದೆ. ಬದಲಾವಣೆಯ ಅಥವಾ ಮೋಸದ ಯಾವ ಚಿಹ್ನೆಯೂ ಕಂಡುಬಂದಿಲ್ಲ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "ভালো খবর: এটা আসল মনে হচ্ছে। কোনো পরিবর্তন বা প্রতারণার লক্ষণ পাওয়া যায়নি।",
    },
    "verdict_caution": {
        "en": "Caution: some details look unusual. Do not pay money or share personal information until you confirm with the official source yourself.",
        "hi": "सावधानी: कुछ बातें अजीब लग रही हैं। अपने स्वयं आधिकारिक स्रोत से पुष्टि करने तक पैसे या व्यक्तिगत जानकारी न भेजें।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "எச்சரிக்கை: சில விவரங்கள் அசாதாரணமாகத் தெரிகின்றன. நீங்களே அதிகாரப்பூர்வ மூலத்தில் கேட்டுப் பார்த்துவிடுவதற்குள், பணத்தை அனுப்பவும் தனிப்பட்ட தகவலைப் பகிரவும் வேண்டாம்.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "జాగ్రత్త: కొన్ని వివరాలు అసాధారణంగా కనిపిస్తున్నాయి. మీరు ఆధికారిక మూలం దగ్గర స్వయంగా పరిశీలించే వరకు, డబ్బు పంపకూడదు లేదా వ్యక్తిగత సమాచారం భాగించకూడదు.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "ശ്രദ്ധിക്കുക: ചില വിശദാംശങ്ങൾ അസാധാരണമായി തോന്നുന്നു. നിങ്ങൾ ആധികാരിക ഉറവിടത്തിൽ നിന്ന് സ്വയം പരിശോധിച്ചു തീർക്കുന്നതു വരെ, പണം അയക്കരുത്, വ്യക്തിഗത വിവരങ്ങളും പങ്കുവയ്ക്കരുത്.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಎಚ್ಚರಿಕೆ: ಕೆಲವು ವಿವರಗಳು ಅಸಾಮಾನ್ಯವಾಗಿ ಕಾಣುತ್ತಿವೆ. ನೀವು ಆಧಿಕಾರಿತಮೂಲದಿಂದ ಸ್ವಯಂ ಪರಿಶೀಲಿಸುವ ವರೆಗೆ, ಹಣ ಅಥವಾ ವ್ಯಕ್ತಿಗತ ಮಾಹಿತಿ ಹಂಚಬೇಡಿ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "সতর্কতা: কিছু বিস্তারিত অস্বাভাবিক মনে হচ্ছে। আপনি নিজে সরকারি উৎস থেকে যাচাই করার আগে পর্যন্ত টাকা বা ব্যক্তিগত তথ্য শেয়ার করবেন না।",
    },
    "verdict_do_not_use": {
        "en": "Warning: this looks like a scam or harmful item. Do not open it, do not press any links, delete the message. If you already sent money, call your bank right away.",
        "hi": "चेतावनी: यह धोखाधड़ी या हानिकारक लग रहा है। इसे खोलना न करें, किसी लिंक पर दबाएँ न, संदेश हटा दें। अगर पहले ही पैसे भेज चुके हैं तो तुरंत अपने बैंक को फोन कीजिए।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "எச்சரிக்கை: இது ஏமாற்று அல்லது கேடு விளைவிப்பது போல் தெரிகிறது. அதைத் திறக்கவும், எந்த இணைப்பையும் அழுத்தவும் வேண்டாம்; செய்தியை நீக்குங்கள். ஏற்கனவே பணம் அனுப்பியிருந்தால், உடனே உங்கள் வங்கிக்கு தொலைபாஷை செய்யுங்கள்.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "హెచ్చరిక: ఇది మోసం లేదా హానికరమైనదిగా కనిపిస్తోంది. దీనిని తెరవకండి, ఏ లింక్‌నొక్కకండి, సందేశాన్ని తీసేయండి. మీరు ఇప్పటికే డబ్బు పంపிட்டట్లైతే, వెంటనే మీ బ్యాంకును సంప్రదించండి.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "അറിയിപ്പ്: ഇത് തട്ടിപ്പോ ദോഷം തീരുമുള്ള ഒന്നോ എന്ന് തോന്നുന്നു. ഇത് തുറക്കരുത്, ലിങ്കുകളിൽ അമർത്തരുത്, സന്ദേശം നീക്കം ചെയ്യുക. നേരത്തെ പണം അയച്ചിട്ടുണ്ടെങ്കിൽ, ഉടൻ ബാങ്കിനെ ബന്ധപ്പെടുക.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಎಚ್ಚರಿಕೆ: ಇದು ಮೋಸ ಅಥವಾ ಹಾನಿಕರವಾದದ್ದಿನಂತೆ ಕಾಣುತ್ತಿದೆ. ಇದನ್ನು ತೆರೆಯಬೇಡಿ, ಲಿಂಕ್‌ಗಳನ್ನು ಒತ್ತಬೇಡಿ, ಸಂದೇಶವನ್ನು ಅಳಿಸಿ. ನೀವು ಈಗಾಗಲೇ ಹಣ ಕಳುಹಿಸಿದರೆ, ತಕ್ಷಣ ಬ್ಯಾಂಕಿಗೆ ಕರೆ ಮಾಡಿ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "সতর্কবার্তা: এটা প্রতারণা বা ক্ষতিকর মনে হচ্ছে। এটা খুলবেন না, কোনো লিংকে চাপ দিবেন না, বার্তা মুছে ফেলুন। আপনি ইতিমধ্যে টাকা পাঠিয়েছেন, তাহলে সঙ্গে সঙ্গে আপনার ব্যাংকে ফোন করুন।",
    },

    # Fusion v2 pattern explanations (2026-08-25): appended to the verdict when
    # a coherent deepfake pattern fired. en+hi shipped; draft langs fall back
    # to en by design (see test_23 GT-overlay policy).
    "pattern_fully_generated": {
        "en": "Multiple independent checks agree this was generated by AI, not recorded from real life, because the frames, faces and audio/video sync all show synthetic artifacts.",
        "hi": "कई स्वतंत्र जाँचें सहमत हैं कि यह AI से बनाया गया था, असल जीवन से नहीं, क्योंकि फ़्रेम, चेहरे और ऑडियो/वीडियो तालमेल सब कृत्रिम निशान दिखाते हैं।",
    },
    "pattern_face_swap_partial": {
        "en": "The face appears heavily edited or replaced, and the audio and video are not fully consistent with the original.",
        "hi": "चेहरा काफ़ी बदला/जोड़ा गया दिखता है, और ऑडियो/वीडियो मूल से पूरी तरह मेल नहीं खाते।",
    },
    "pattern_corroborated_multi": {
        "en": "Several independent checks point the same way, which makes this finding stronger.",
        "hi": "कई स्वतंत्र जाँचें एक ही दिशा में हैं, जिससे यह निष्कर्ष अधिक मज़बूत होता है।",
    },
    "pattern_conflicting_detectors": {
        "en": "The checks disagree, so I could not settle this confidently.",
        "hi": "जाँचें अलग-अलग राय दे रही हैं, इसलिए मैं आत्मविश्वास से फ़ैसला नहीं कर सका।",
    },
    "pattern_weak_signal_single": {
        "en": "Only one check raised a concern on its own, so treat this with caution.",
        "hi": "अकेले केवल एक जाँच ने चेतावनी दी, इसलिए इसे सावधानी से लें।",
    },
    "verdict_unable": {
        "en": "I could not fully verify this right now. Please check it directly on the official website or app before doing anything.",
        "hi": "अभी इसका पूरा सत्यापन नहीं कर सका। किसी भी काम से पहले आधिकारिक वेबसाइट या ऐप पर स्वयं जाँच कर लें।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "இதை இப்போது முழுமையாகச் சரிபார்க்க முடியவில்லை. எதாவது செய்வதற்கு முன், அதிகாரப்பூர்வ இணையதளத்தில் அல்லது பயன்பாட்டில் நேரடியாகச் சரிபார்க்குங்கள்.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "దీనిని ఇప్పుడు పూర్తిగా పరిశీలించలేకపోయాను. ఏదైనా చేయడానికి ముందు, ఆధికారిక వెబ్‌సైట్ లేదా యాప్ మీద స్వయంగా తనిఖీ చేసుకోండి.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "ഇത് ഇപ്പോൾ പൂർണ്ണമായി പരിശോധിക്കാനായില്ല. എന്തെങ്കിലും ചെയ്യുന്നതിനായി, ആധികാരിക വെബ്‌സൈറ്റിലോ ആപ്പിലോ നേരിട്ട് പരിശോധിക്കുക.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಇದನ್ನು ಈಗ ಪೂರ್ಣವಾಗಿ ಪರೀಕ್ಷಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಯಾವುದನ್ನಾದರೂ ಮಾಡುವ ಮೊದಲು, ಆಧಿಕಾರಿತಮೂಲದ ವೆಬ್‌ಸೈಟ್ ಅಥವಾ ಆ್ಯಪ್‌ನಲ್ಲಿ ನೇರವಾಗಿ ಪರೀಕ್ಷಿಸಿ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "আমি এখন এটা সম্পূর্ণভাবে যাচাই করতে পারিনি। যেকোনো কাজের আগে, সরকারি ওয়েবসাইটে বা আ্য্পে সরাসরি যাচাই করুন।",
    },
    "unable_coverage": {
        # Coverage-aware unable (2026-08-25): state what DID run so the user can
        # distinguish 'checked N angles, still inconclusive' from 'nothing checked'.
        "en": "For reference, I was able to run %(n)s check(s) — they just didn't add up to a clear answer.",
        "hi": "संदर्भ हेतु: मैं %(n)s जाँच चला सका — पर उनसे कोई स्पष्ट निष्कर्ष नहीं मिला।",
    },
    "confidence_line": {
        "en": "My confidence: %(conf)s. This tool helps, but a human double-check is always safer.",
        "hi": "मेरी विश्वसनीयता: %(conf)s। यह टूल मदद करता है, लेकिन इंसानी डबल-चेक हमेशा बेहतर है।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "என் நம்பிக்கை: %(conf)s. இந்தக் கருவி உதவுகிறது; ஆனால் ஒரு மனிதர் மீண்டும் சரிபார்ப்பது எப்போதும் பாதுகாப்பானது.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "నా నమ్మకం: %(conf)s. ఈ సాధనం సహాయపడుతుంది, కానీ ఒక మనిషి మళ్ళీ తనిఖీ చేయడం ఎల్లప్పుడూ మరింత సురక్షితం.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "എന്റെ വിശ്വാസം: %(conf)s. ഈ ഉപകരണം സഹായിക്കുന്നു, പക്ഷേ മനുഷ്യന്റെ രണ്ടാം പരിശോധന എപ്പോഴും കൂടുതൽ സുരക്ഷിതമാണ്.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ನನ್ನ ನಂಬಿಕೆ: %(conf)s. ಈ ಸಾಧನ ಸಹಾಯ ಮಾಡುತ್ತದೆ, ಆದರೆ ಜನರ ಮರು ಪರಿಶೀಲನೆ ಯಾವಾಗಲೂ ಕൂಡೂ ಸುರಕ್ಷಿತ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "আমার আস্থা: %(conf)s। এই টুল সাহায্য করে, তবে মানুষের পুনরায় যাচাই সবসময় নিরাপদ।",
    },
    "advice_avoid_links": {
        "en": "Tip: never click links from unknown numbers. Official institutions never ask for passwords on WhatsApp links.",
        "hi": "सलाह: अज्ञात नंबरों से लिंक खोलना न करें। सरकारी संस्थाएं व्हाट्सऐप लिंक पर कभी पासवर्ड नहीं माँगतीं।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "குறிப்பு: தெரியாத எண்களிலிருந்து வரும் இணைப்புகளை எப்போதும் அழுத்த வேண்டாம். அரச அமைப்புகள் WhatsApp இணைப்பில் கடவுச்சொல் கேட்பதில்லை.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "సలహు: తెలియని నంబర్‌ల నుండి వచ్చే లింక్‌లను ఎప్పుడూ నొక్కకూడదు. ప్రభుత్వ సంస్థలు WhatsApp లింక్‌లలో పాస్‌వర్డ్ అడుగుతాయి.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "വേഗം: അജ്ഞാത നമ്പറുകളിൽ നിന്ന് വരുന്ന ലിങ്കുകൾ ഒരിക്കലും അമർത്തരുത്. സർക്കാർ സ്ഥാപനങ്ങൾ WhatsApp ലിങ്കിൽ പാസ്‌വേഡ് ചോദിക്കില്ല.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಸೂಚನೆ: ಅಪರಿಚಿತ ಸಂಖ್ಯೆಗಳಿಂದ ಬರುವ ಲಿಂಕ್‌ಗಳನ್ನು ಎಂದಿಗೂ ಒತ್ತಬೇಡಿ. ಸರ್ಕಾರಿ ಸಂಸ್ಥೆಗಳು WhatsApp ಲಿಂಕ್‌ನಲ್ಲಿ ಪಾಸ್‌ವರ್ಡ್ ಕೇಳುವುದಿಲ್ಲ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "পরামর্শ: অচেনা নম্বর থেকে আসা লিংকে কখনো চাপবেন না। সরকারি প্রতিষ্ঠান WhatsApp লিংকে পাসওয়ার্ড চায় না।",
    },
    # UX 2026-08-26: richer narrative reply. en-only for now; other langs fall
    # back to en via t() (no native-review gate — translate when a reviewer lands).
    "verdict_tile": {
        "en": "%(emoji)s %(label)s · %(risk)s RISK",
        "hi": "%(emoji)s %(label)s · %(risk)s जोखिम",
    },
    "concern_count": {
        "en": "It found %(n)s key concern%(plural)s:",
        "hi": "इसमें %(n)s प्रमुख चिंता%(plural)s मिलीं:",
    },
    "concern_audio_ai": {
        "en": "the voice shows signs of AI manipulation",
        "hi": "आवाज़ में AI हेरफेर के संकेत दिखते हैं",
    },
    "concern_video_face": {
        "en": "the face appears digitally altered",
        "hi": "चेहरा डिजिटल रूप से बदला हुआ प्रतीत होता है",
    },
    "concern_av_sync": {
        "en": "the voice and moving lips do not match (audio/video inconsistency)",
        "hi": "आवाज़ और होंठों की हरकत मेल नहीं खाते (ऑडियो/वीडियो असंगति)",
    },
    "concern_url_flag": {
        "en": "the link has a poor security reputation",
        "hi": "लिंक की सुरक्षा प्रतिष्ठा खराब है",
    },
    "concern_url_typo": {
        "en": "the address looks like a disguised copy of a real website",
        "hi": "पता किसी असली वेबसाइट की नक़ल लगता है",
    },
    "concern_doc_forged": {
        "en": "the document shows signs of forgery",
        "hi": "दस्तावेज़ में जालसाज़ी के संकेत दिखते हैं",
    },
    "concern_unverified_source": {
        "en": "the claim could not be verified through reliable sources",
        "hi": "विश्वसनीय स्रोतों से इस दावे की पुष्टि नहीं हो सकी",
    },
    "recommend_dont_forward": {
        "en": "Don't forward. Verify with a trusted source.",
        "hi": "आगे न भेजें। किसी विश्वसनीय स्रोत से जाँच करें।",
    },
    "recommend_verify_source": {
        "en": "Don't forward or pay until you check with the official source.",
        "hi": "आधिकारिक स्रोत से जाँच किए बिना आगे न भेजें या पैसे न दें।",
    },
    "recommend_nothing": {
        "en": "No action needed, but a human double-check is still safer.",
        "hi": "कोई कार्रवाई ज़रूरी नहीं, लेकिन मानवीय जाँच अब भी सुरक्षित है।",
    },
    "progress_file": {
        "en": "Scanning the file (%(name)s)…",
        "hi": "फ़ाइल की स्कैनिंग जारी है (%(name)s)…",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "கோப்பு (%(name)s) ஐச் சோதித்து வருகிறேன்.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "ఫైల్ (%(name)s) ను స్కాన్ చేస్తున్నాను.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "ഫയൽ (%(name)s) സ്കാൻ ചെയ്യുന്നു.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಫೈಲ್ (%(name)s) ಸ್ಕಾನ್ ಮಾಡುತ್ತಿದ್ದೇನೆ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "ফাইল (%(name)s) স্ক্যান করছি।",
    },
    "progress_url": {
        "en": "Analysing the link…",
        "hi": "लिंक का विश्लेषण हो रहा है…",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "இந்த இணைப்பை ஆய்வு செய்கிறேன்…",
        # draft (te); see docs/i18n/te.draft.md
        "te": "లింక్‌ను విశ్లేషిస్తున్నాను…",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "ലിങ്ക് വിശകലനം ചെയ്യുന്നു…",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಲಿಂಕ್ ವಿಶ್ಲೇಷಿಸುತ್ತಿದ್ದೇನೆ…",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "লিংক বিশ্লেষণ করছি…",
    },
    "progress_media": {
        "en": "Studying the video/audio in detail. This takes a little longer…",
        "hi": "वीडियो/ऑडियो का विस्तृत अध्ययन हो रहा है। यह थोड़ा समय लेगा…",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "விடியோ/ஆடியோவை விரிவாக ஆய்வு செய்கிறேன். இதற்கு ஒரு சிறிது நேரம் ஆகும்…",
        # draft (te); see docs/i18n/te.draft.md
        "te": "వీడియో/ఆడియోను వివరంగా అధ్యయనం చేస్తున్నాను. దీనికి కొంచెం సమయం పడుతుంది…",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "വീഡിയോ/ഓഡിയോ വിശദമായി പഠിക്കുന്നു. ഇതിന് കുറച്ച് സമയം വാങ്ങും…",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ವೀಡಿಯೊ/ಆಡಿಯೊ ವಿಸ್ತಾರವಾಗಿ ಅಧ್ಯಯನ ಮಾಡುತ್ತಿದ್ದೇನೆ. ಇದಕ್ಕೆ ಕೆಲವು ಸಮಯ ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ…",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "ভিডিও/অডিও ভালো করে দেখছি। এর জন্য একটু সময় লাগবে…",
    },
    "evidence_missing": {
        "en": "Some checks were skipped because a required service was unavailable; the verdict above reflects only what I could actually test.",
        "hi": "कुछ जाँचें छोड़नी पड़ीं क्योंकि आवश्यक सेवा उपलब्ध नहीं थी; ऊपर वाला नतीजा सिर्फ इतने पर आधारित है जितना मैंने वास्तव में जाँचा।",
        # draft (ta); see docs/i18n/ta.draft.md
        "ta": "தேவையான சேவை கிடைக்காமல் இருந்ததால் சில சோதனைகள் தவிர்க்கப்பட்டன; மேலே உள்ள முடிவு, நான் மெய்யில் சோதித்ததை மட்டுமே காட்டுகிறது.",
        # draft (te); see docs/i18n/te.draft.md
        "te": "అవసరమైన సేవ అందుబాటులో లేనందున కొన్ని తనిఖీలు దాటబడ్డాయి; పైన ఉన్న ఫలితం నేను నిజంగా పరీక్షించినదాన్ని మాత్రమే ప్రతిబింబిస్తుంది.",
        # draft (ml); see docs/i18n/ml.draft.md
        "ml": "ആവശ്യമായ സേവനം ലഭ്യമായില്ലെന്ന് കാരണം ചില പരിശോധനകൾ ഒഴിവാക്കി; മുകളിലെ നിരണ്ണയം ഞാൻ യഥാർത്ഥത്തിൽ പരിശോധിച്ചത് മാത്രം പ്രതിഫലിപ്പിക്കുന്നു.",
        # draft (kn); see docs/i18n/kn.draft.md
        "kn": "ಅಗತ್ಯ ಸೇವೆ ಲಭ್ಯವಿರದ ಕಾರಣ ಕೆಲವು ಪರೀಕ್ಷೆಗಳು ಹಾದುಹೋದವು; ಮೇಲಿನ ಫಲಿತಾಂಶ ನಾನು ನಿಜವಾಗಿ ಪರೀಕ್ಷಿಸಿದದ್ದನ್ನು ಮಾತ್ರ ಪ್ರತಿಬಿಂಬಿಸುತ್ತದೆ.",
        # draft (bn); see docs/i18n/bn.draft.md
        "bn": "প্রয়োজনীয় সেবা পাওয়া যায়নি বলে কিছু পরীক্ষা করা হয়নি; উপরের সিদ্ধান্ত শুধু সে কথা নির্দেশ করে যা আমি সত্যিই পরীক্ষা করতে পেরেছি।",
    },
    "heavy_pending_notice": {
        "en": "The quick check is done. My deeper check is still running — I'll send you an update when it finishes.",
        "hi": "जल्दी वाला जाँच हो चुका। मेरा गहरा जाँच अभी चल रह है — जब वह पूरी होईगा तब मैं फिर संदेश भेजूंगा।",
        "ta": "விரைவான சோதனை முடிந்தது. எனது ஆழமான சோதனை இன்னும் நடந்து கொண்டிருக்கிறது — அது முடிந்தவுடன் உங்களுக்கு மீண்டும் செய்தி அனுப்புவேன்.",
        "te": "వేగవంతమైన తనిఖీ పూర్తయింది. నా లోతైన తనిఖీ ఇంకా జరుగుతోంది — అది పూర్తయినప్పుడు మరో సందేశం పంపుతాను.",
        "ml": "വേഗത്തിലുള്ള പരിശോധന അവസാനിച്ചു. എന്റെ ആഴത്തിലുള്ള പരിശോധന ഇപ്പോഴും നടക്കുന്നുണ്ട് — അത് അവസാനിച്ചാൽ ഞാൻ നിങ്ങൾക്ക് വീണ്ടും അറിയിക്കാം.",
        "kn": "ವೇಗದ ಪರೀಕ್ಷೆ ಮುಗಿದಿದೆ. ನನ್ನ ಆಳ ಪರೀಕ್ಷೆ ಇನ್ನೂ ನಡೆಯುತ್ತಿದೆ — ಅದು ಮುಗಿದಾಗ ನಿಮಗೆ ಮತ್ತೊಂದು ಸಂದೇಶ ಕಳುಹಿಸುತ್ತೇನೆ.",
        "bn": "দ্রুত পরীক্ষা শেষ হয়ে গেল। আমার গভীর পরীক্ষা এখনও চলছে — সেটা শেষ হলে আমি আপনাকে আবার একটি বার্তা পাঠাব।",
    },
    "heavy_followup": {
        "en": "My deeper check (%(cap)s) finished. Result: %(verdict)s — confidence %(conf)s.",
        "hi": "मेरा गहरा जाँच (%(cap)s) पूरी हो गई। परिणाम: %(verdict)s — विश्वास %(conf)s।",
        "ta": "எனது ஆழமான சோதனை (%(cap)s) முடிந்தது. முடிவு: %(verdict)s — நம்பிக்கை %(conf)s.",
        "te": "నా లోతైన తనిఖీ (%(cap)s) పూర్తయింది. ఫలితం: %(verdict)s — నమ్మకం %(conf)s.",
        "ml": "എന്റെ ആഴത്തിലുള്ള പരിശോധന (%(cap)s) അവസാനിച്ചു. ഫലം: %(verdict)s — വിശ്വാസം %(conf)s.",
        "kn": "ನನ್ನ ಆಳ ಪರೀಕ್ಷೆ (%(cap)s) ಮುಗಿದಿದೆ. ಫಲಿತಾಂಶ: %(verdict)s — ನಂಬಿಕೆ %(conf)s.",
        "bn": "আমার গভীর পরীক্ষা (%(cap)s) শেষ হয়েছে। ফলাফল: %(verdict)s — আস্থা %(conf)s।",
    },
}


def t(key: str, lang: str = "en", **fmt) -> str:
    table = _DEFAULTS.get(key, {})
    s = table.get(lang)
    if not s:
        s = table.get(_DEFAULT_LANG_FALLBACK) or key
    if fmt:
        try:
            s = s % fmt
        except Exception:
            pass
    return s


def detect_language(text: str) -> str:
    """Cheap script-based detection to pick the reply language."""
    s = (text or "").strip()
    if not s:
        return "en"
    cp = [ord(c) for c in s[:64]]
    if any(0x0900 <= c <= 0x097F for c in cp):
        return "hi"
    if any(0x0980 <= c <= 0x09FF for c in cp):
        return "bn"
    if any(0x0B80 <= c <= 0x0BFF for c in cp):
        return "ta"
    if any(0x0C00 <= c <= 0x0C7F for c in cp):
        return "te"
    if any(0x0C80 <= c <= 0x0CFF for c in cp):
        return "kn"
    if any(0x0D00 <= c <= 0x0D7F for c in cp):
        return "ml"
    return "en"


# --------------------------------------------------------- language selection
# Lets a user CHOOSE their reply language instead of it being re-guessed from
# every message. Recognised before the scam-check pipeline (see channels.py) so
# a language request is never treated as content to verify.

import re as _re

_LANG_ORDER = ("en", "hi", "ta", "te", "ml", "kn", "bn")
_LANG_NAMES = {"en": "English", "hi": "हिंदी", "ta": "தமிழ்", "te": "తెలుగు",
               "ml": "മലയാളം", "kn": "ಕನ್ನಡ", "bn": "বাংলা"}

# Full language NAMES — safe to accept as a bare, whole message ("hindi").
_LANG_NAME_ALIASES = {
    "english": "en", "angrezi": "en", "इंग्लिश": "en",
    "hindi": "hi", "हिंदी": "hi", "हिन्दी": "hi",
    "tamil": "ta", "தமிழ்": "ta",
    "telugu": "te", "తెలుగు": "te",
    "malayalam": "ml", "മലയാളം": "ml",
    "kannada": "kn", "ಕನ್ನಡ": "kn",
    "bengali": "bn", "bangla": "bn", "বাংলা": "bn",
}
# 2-letter codes are only honoured alongside a trigger or while awaiting a menu
# reply — never as a bare message, so "hi" stays a greeting, not "set English".
_LANG_CODE_ALIASES = {c: c for c in _LANG_ORDER}
_LANG_TRIGGERS_LATIN = {"language", "lang", "bhasha", "bhaasha"}
_LANG_TRIGGERS_NATIVE = ("भाषा", "மொழி", "భాష", "ಭಾಷೆ", "ভাষা")
_TOKEN_RE = _re.compile(r"[a-zऀ-ൿ]+")


def language_display_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


def language_menu_text(lang: str = "en") -> str:
    """Numbered list of the supported languages, headed by a localized prompt."""
    header = t("language_choose", lang)
    body = "\n".join(f"{i + 1}. {_LANG_NAMES[c]}" for i, c in enumerate(_LANG_ORDER))
    return f"{header}\n{body}"


def parse_language_request(text: str, awaiting: bool = False):
    """Classify a message as a language-selection interaction.

    Returns ("set", code) to switch language, ("menu",) to offer the chooser,
    or None if the message is not about language (so it goes to the normal
    scam-check pipeline). `awaiting=True` means the previous reply WAS the
    chooser, so a bare number / code / name counts as the answer.
    """
    s = (text or "").strip().lower()
    if not s or len(s) > 40:            # long text => real content, not a command
        return None
    toks = set(_TOKEN_RE.findall(s))

    if awaiting:
        if s.isdigit():
            n = int(s)
            if 1 <= n <= len(_LANG_ORDER):
                return ("set", _LANG_ORDER[n - 1])
        if s in _LANG_NAME_ALIASES:
            return ("set", _LANG_NAME_ALIASES[s])
        if s in _LANG_CODE_ALIASES:
            return ("set", _LANG_CODE_ALIASES[s])

    # a bare full language name is an unambiguous switch on its own
    if s in _LANG_NAME_ALIASES:
        return ("set", _LANG_NAME_ALIASES[s])

    has_trigger = (bool(toks & _LANG_TRIGGERS_LATIN)
                   or any(nt in s for nt in _LANG_TRIGGERS_NATIVE))
    if has_trigger:
        for name, code in _LANG_NAME_ALIASES.items():   # "language hindi"
            if name in toks or name in s:
                return ("set", code)
        for code in _LANG_ORDER:                        # "set language hi"
            if code in toks:
                return ("set", code)
        return ("menu",)                                # just "language"
    return None


def load_custom_strings(path: str | Path | None = None) -> None:
    """Overlay user-supplied translations (json: {key:{lang:text}}) over defaults."""
    p = Path(path) if path else Path(__file__).parent / "i18n_extra.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            for k, langs in data.items():
                _DEFAULTS.setdefault(k, {}).update(langs)
        except Exception:
            pass


# Language-selection strings (2026-08-27). en authoritative; the Indic strings
# are best-effort (same native-review-pending status as the rest of the corpus).
_DEFAULTS.setdefault("language_choose", {
    "en": "Choose your language — reply with the number:",
    "hi": "अपनी भाषा चुनें — नंबर भेजकर जवाब दें:",
    "ta": "உங்கள் மொழியைத் தேர்ந்தெடுக்கவும் — எண்ணை அனுப்புங்கள்:",
    "te": "మీ భాషను ఎంచుకోండి — సంఖ్యను పంపండి:",
    "ml": "നിങ്ങളുടെ ഭാഷ തിരഞ്ഞെടുക്കുക — നമ്പർ അയയ്ക്കുക:",
    "kn": "ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆರಿಸಿ — ಸಂಖ್ಯೆಯನ್ನು ಕಳುಹಿಸಿ:",
    "bn": "আপনার ভাষা বেছে নিন — নম্বর পাঠান:",
})
_DEFAULTS.setdefault("language_set", {
    "en": "✅ Language set to %(name)s. Send me any link, file, photo, video, or audio and I'll check whether it's safe.",
    "hi": "✅ भाषा %(name)s पर सेट हो गई। कोई भी लिंक, फ़ाइल, फ़ोटो, वीडियो या ऑडियो भेजें — मैं जाँचूँगा कि वह सुरक्षित है या नहीं।",
    "ta": "✅ மொழி %(name)s ஆக அமைக்கப்பட்டது. ஏதேனும் இணைப்பு, கோப்பு, படம், வீடியோ அல்லது ஆடியோ அனுப்புங்கள் — பாதுகாப்பானதா எனச் சரிபார்ப்பேன்.",
    "te": "✅ భాష %(name)s కు సెట్ చేయబడింది. ఏదైనా లింక్, ఫైల్, ఫోటో, వీడియో లేదా ఆడియో పంపండి — సురక్షితమేనా అని తనిఖీ చేస్తాను.",
    "ml": "✅ ഭാഷ %(name)s ആയി സജ്ജീകരിച്ചു. ഏതെങ്കിലും ലിങ്ക്, ഫയൽ, ഫോട്ടോ, വീഡിയോ അല്ലെങ്കിൽ ഓഡിയോ അയയ്ക്കൂ — സുരക്ഷിതമാണോ എന്ന് പരിശോധിക്കാം.",
    "kn": "✅ ಭಾಷೆ %(name)s ಗೆ ಹೊಂದಿಸಲಾಗಿದೆ. ಯಾವುದೇ ಲಿಂಕ್, ಫೈಲ್, ಫೋಟೋ, ವೀಡಿಯೊ ಅಥವಾ ಆಡಿಯೊ ಕಳುಹಿಸಿ — ಸುರಕ್ಷಿತವೇ ಎಂದು ಪರಿಶೀಲಿಸುತ್ತೇನೆ.",
    "bn": "✅ ভাষা %(name)s এ সেট করা হয়েছে। যেকোনো লিঙ্ক, ফাইল, ছবি, ভিডিও বা অডিও পাঠান — নিরাপদ কিনা পরীক্ষা করব।",
})

# MT-audit overlay (2026-08-21): 32 Google-Translate-rendered strings that
# back-translated closer to the English corpus than the LLM drafts.
# Build-time QA only — the runtime product stays offline templates.
load_custom_strings()
