#!/usr/bin/env python3
"""Author the five Indian-language i18n DRAFT files (ta/te/ml/kn/bn).

Writes docs/i18n/<lang>.draft.md for each language. Every file covers the full
key set derived from docs/i18n/en.md (14 keys incl. the two Task-3.2 follow-up
keys). Translations are MY OWN zero-cloud generation aimed at a ~65-year-old
non-technical elder: warm, plain, respectful, short sentences, no tech jargon.
printf-style placeholders are preserved VERBATIM. Lines I am <~90% confident
about carry a trailing [?] marker.

This generator only writes DRAFT artifacts — it NEVER touches i18n._DEFAULTS.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EN_MD = ROOT / "docs" / "i18n" / "en.md"
OUT_DIR = ROOT / "docs" / "i18n"

LANG_NAMES = {"ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
              "kn": "Kannada", "bn": "Bengali"}


def en_keys() -> list[str]:
    """Ordered key list from the machine-generated en.md."""
    keys: list[str] = []
    for line in EN_MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*`(\w+)`\s*\|", line)
        if m:
            keys.append(m.group(1))
    return keys


# ------------------------------------------------------------------ text ------
# Values may end with " [?]" when sub-90% confident. Reviewer-note dict carries
# per-line guidance for the native-review gate.

TA = {
    "greeting": "வணக்கம்! நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    "analyzing": "இதை இப்போது ஆய்வு செய்து கொண்டிருக்கிறேன். ஒரு நிமிடம் காத்திருங்கள்.",
    "verdict_trust": "நல்ல செய்தி: இது உண்மையானதாகத் தெரிகிறது. மாற்றம் அல்லது ஏமாற்றத்தின் சிறு குறியே இல்லை.",
    "verdict_caution": "எச்சரிக்கை: சில விவரங்கள் அசாதாரணமாகத் தெரிகின்றன. நீங்களே அதிகாரப்பூர்வ மூலத்தில் கேட்டுப் பார்த்துவிடுவதற்குள், பணத்தை அனுப்பவும் தனிப்பட்ட தகவலைப் பகிரவும் வேண்டாம். [?]",
    "verdict_do_not_use": "எச்சரிக்கை: இது ஏமாற்று அல்லது கேடு விளைவிப்பது போல் தெரிகிறது. அதைத் திறக்கவும், எந்த இணைப்பையும் அழுத்தவும் வேண்டாம்; செய்தியை நீக்குங்கள். ஏற்கனவே பணம் அனுப்பியிருந்தால், உடனே உங்கள் வங்கிக்கு தொலைபாஷை செய்யுங்கள். [?]",
    "verdict_unable": "இதை இப்போது முழுமையாகச் சரிபார்க்க முடியவில்லை. எதாவது செய்வதற்கு முன், அதிகாரப்பூர்வ இணையதளத்தில் அல்லது பயன்பாட்டில் நேரடியாகச் சரிபார்க்குங்கள்.",
    "confidence_line": "என் நம்பிக்கை: %(conf)s. இந்தக் கருவி உதவுகிறது; ஆனால் ஒரு மனிதர் மீண்டும் சரிபார்ப்பது எப்போதும் பாதுகாப்பானது. [?]",
    "advice_avoid_links": "குறிப்பு: தெரியாத எண்களிலிருந்து வரும் இணைப்புகளை எப்போதும் அழுத்த வேண்டாம். அரச அமைப்புகள் WhatsApp இணைப்பில் கடவுச்சொல் கேட்பதில்லை.",
    "progress_file": "கோப்பு (%(name)s) ஐச் சோதித்து வருகிறேன்.",
    "progress_url": "இந்த இணைப்பை ஆய்வு செய்கிறேன்…",
    "progress_media": "விடியோ/ஆடியோவை விரிவாக ஆய்வு செய்கிறேன். இதற்கு ஒரு சிறிது நேரம் ஆகும்… [?]",
    "evidence_missing": "தேவையான சேவை கிடைக்காமல் இருந்ததால் சில சோதனைகள் தவிர்க்கப்பட்டன; மேலே உள்ள முடிவு, நான் மெய்யில் சோதித்ததை மட்டுமே காட்டுகிறது. [?]",
    "heavy_pending_notice": "விரைவான சோதனை முடிந்தது. எனது ஆழமான சோதனை இன்னும் நடந்து கொண்டிருக்கிறது — அது முடிந்தவுடன் உங்களுக்கு மீண்டும் செய்தி அனுப்புவேன். [?]",
    "heavy_followup": "எனது ஆழமான சோதனை (%(cap)s) முடிந்தது. முடிவு: %(verdict)s — நம்பிக்கை %(conf)s.",
}
TA_NOTES = {
    "verdict_caution": "longest sentence in the file; split or simplify if too much for an elder",
    "verdict_do_not_use": "four imperative clauses; confirm the verb forms sound natural",
    "confidence_line": "'நம்பிக்கை' (trust/confidence) chosen over a calque of 'confidence'",
    "progress_media": "video/audio word order — check ellipsis and phrasing",
    "evidence_missing": "slightly formal; consider 'செயலாக்க முடியவில்லை'-style phrasing",
    "heavy_pending_notice": "word order tuned for readability; please confirm",
}

TE = {
    "greeting": "హాయ్! నేను మీకు ఎలా సహాయం చేయగలను?",
    "analyzing": "ఇప్పుడు దీనిని తనిఖీ చేస్తున్నాను. దయచేసి ఒక నిమిషం వేచి ఉండండి.",
    "verdict_trust": "మంచి సమాచారం: ఇది నిజమైనట్లు కనిపిస్తోంది. మార్పిడి లేదా మోసం గురించి ఏ లక్షణాలూ కనబడలేదు.",
    "verdict_caution": "జాగ్రత్త: కొన్ని వివరాలు అసాధారణంగా కనిపిస్తున్నాయి. మీరు ఆధికారిక మూలం దగ్గర స్వయంగా పరిశీలించే వరకు, డబ్బు పంపకూడదు లేదా వ్యక్తిగత సమాచారం భాగించకూడదు. [?]",
    "verdict_do_not_use": "హెచ్చరిక: ఇది మోసం లేదా హానికరమైనదిగా కనిపిస్తోంది. దీనిని తెరవకండి, ఏ లింక్‌నొక్కకండి, సందేశాన్ని తీసేయండి. మీరు ఇప్పటికే డబ్బు పంపிட்டట్లైతే, వెంటనే మీ బ్యాంకును సంప్రదించండి. [?]",
    "verdict_unable": "దీనిని ఇప్పుడు పూర్తిగా పరిశీలించలేకపోయాను. ఏదైనా చేయడానికి ముందు, ఆధికారిక వెబ్‌సైట్ లేదా యాప్ మీద స్వయంగా తనిఖీ చేసుకోండి.",
    "confidence_line": "నా నమ్మకం: %(conf)s. ఈ సాధనం సహాయపడుతుంది, కానీ ఒక మనిషి మళ్ళీ తనిఖీ చేయడం ఎల్లప్పుడూ మరింత సురక్షితం. [?]",
    "advice_avoid_links": "సలహు: తెలియని నంబర్‌ల నుండి వచ్చే లింక్‌లను ఎప్పుడూ నొక్కకూడదు. ప్రభుత్వ సంస్థలు WhatsApp లింక్‌లలో పాస్‌వర్డ్ అడుగుతాయి.",
    "progress_file": "ఫైల్ (%(name)s) ను స్కాన్ చేస్తున్నాను.",
    "progress_url": "లింక్‌ను విశ్లేషిస్తున్నాను…",
    "progress_media": "వీడియో/ఆడియోను వివరంగా అధ్యయనం చేస్తున్నాను. దీనికి కొంచెం సమయం పడుతుంది…",
    "evidence_missing": "అవసరమైన సేవ అందుబాటులో లేనందున కొన్ని తనిఖీలు దాటబడ్డాయి; పైన ఉన్న ఫలితం నేను నిజంగా పరీక్షించినదాన్ని మాత్రమే ప్రతిబింబిస్తుంది. [?]",
    "heavy_pending_notice": "వేగవంతమైన తనిఖీ పూర్తయింది. నా లోతైన తనిఖీ ఇంకా జరుగుతోంది — అది పూర్తయినప్పుడు మరో సందేశం పంపుతాను. [?]",
    "heavy_followup": "నా లోతైన తనిఖీ (%(cap)s) పూర్తయింది. ఫలితం: %(verdict)s — నమ్మకం %(conf)s.",
}
TE_NOTES = {
    "verdict_caution": "long; consider splitting into two shorter sentences",
    "verdict_do_not_use": "'తెరవకండి / నొక్కకండి' — confirm polite-imperative register for elders",
    "confidence_line": "'నమ్మకం' (trust) used instead of a formal 'confidence' calque",
    "evidence_missing": "formal compound; native simplification welcome",
    "heavy_pending_notice": "please confirm word order sounds spoken, not written",
}

ML = {
    "greeting": "ഹായ്! ഞാൻ നിങ്ങൾക്ക് എങ്ങനെ സഹായിക്കാം?",
    "analyzing": "ഇപ്പോൾ അത് പരിശോധിക്കുന്നുണ്ട്. ഒരു മിനിറ്റ് കാത്തിരിക്കൂ.",
    "verdict_trust": "നല്ല വാർത്ത: ഇത് യഥാർത്ഥമാണെന്ന് തോന്നുന്നു. മാറ്റമോ തട്ടിപ്പോ — അത്തരം ചിഹ്നങ്ങൾ ഒന്നുമില്ല.",
    "verdict_caution": "ശ്രദ്ധിക്കുക: ചില വിശദാംശങ്ങൾ അസാധാരണമായി തോന്നുന്നു. നിങ്ങൾ ആധികാരിക ഉറവിടത്തിൽ നിന്ന് സ്വയം പരിശോധിച്ചു തീർക്കുന്നതു വരെ, പണം അയക്കരുത്, വ്യക്തിഗത വിവരങ്ങളും പങ്കുവയ്ക്കരുത്. [?]",
    "verdict_do_not_use": "അറിയിപ്പ്: ഇത് തട്ടിപ്പോ ദോഷം തീരുമുള്ള ഒന്നോ എന്ന് തോന്നുന്നു. ഇത് തുറക്കരുത്, ലിങ്കുകളിൽ അമർത്തരുത്, സന്ദേശം നീക്കം ചെയ്യുക. നേരത്തെ പണം അയച്ചിട്ടുണ്ടെങ്കിൽ, ഉടൻ ബാങ്കിനെ ബന്ധപ്പെടുക. [?]",
    "verdict_unable": "ഇത് ഇപ്പോൾ പൂർണ്ണമായി പരിശോധിക്കാനായില്ല. എന്തെങ്കിലും ചെയ്യുന്നതിനായി, ആധികാരിക വെബ്‌സൈറ്റിലോ ആപ്പിലോ നേരിട്ട് പരിശോധിക്കുക.",
    "confidence_line": "എന്റെ വിശ്വാസം: %(conf)s. ഈ ഉപകരണം സഹായിക്കുന്നു, പക്ഷേ മനുഷ്യന്റെ രണ്ടാം പരിശോധന എപ്പോഴും കൂടുതൽ സുരക്ഷിതമാണ്. [?]",
    "advice_avoid_links": "വേഗം: അജ്ഞാത നമ്പറുകളിൽ നിന്ന് വരുന്ന ലിങ്കുകൾ ഒരിക്കലും അമർത്തരുത്. സർക്കാർ സ്ഥാപനങ്ങൾ WhatsApp ലിങ്കിൽ പാസ്‌വേഡ് ചോദിക്കില്ല.",
    "progress_file": "ഫയൽ (%(name)s) സ്കാൻ ചെയ്യുന്നു.",
    "progress_url": "ലിങ്ക് വിശകലനം ചെയ്യുന്നു…",
    "progress_media": "വീഡിയോ/ഓഡിയോ വിശദമായി പഠിക്കുന്നു. ഇതിന് കുറച്ച് സമയം വാങ്ങും…",
    "evidence_missing": "ആവശ്യമായ സേവനം ലഭ്യമായില്ലെന്ന് കാരണം ചില പരിശോധനകൾ ഒഴിവാക്കി; മുകളിലെ നിരണ്ണയം ഞാൻ യഥാർത്ഥത്തിൽ പരിശോധിച്ചത് മാത്രം പ്രതിഫലിപ്പിക്കുന്നു. [?]",
    "heavy_pending_notice": "വേഗത്തിലുള്ള പരിശോധന അവസാനിച്ചു. എന്റെ ആഴത്തിലുള്ള പരിശോധന ഇപ്പോഴും നടക്കുന്നുണ്ട് — അത് അവസാനിച്ചാൽ ഞാൻ നിങ്ങൾക്ക് വീണ്ടും അറിയിക്കാം. [?]",
    "heavy_followup": "എന്റെ ആഴത്തിലുള്ള പരിശോധന (%(cap)s) അവസാനിച്ചു. ഫലം: %(verdict)s — വിശ്വാസം %(conf)s.",
}
ML_NOTES = {
    "verdict_caution": "long; consider splitting",
    "verdict_do_not_use": "check the negative-imperative chain for natural spoken form",
    "confidence_line": "'വിശ്വാസം' = trust/confidence; confirm it fits an elder's ear",
    "evidence_missing": "formal; a plainer rephrase is welcome",
    "heavy_pending_notice": "confirm word order; 'അറിയിക്കാം' feels right but please verify",
}

KN = {
    "greeting": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ? [?]",
    "analyzing": "ಈಗ ಅದನ್ನು ಪರೀಕ್ಷಿಸುತ್ತಿದ್ದೇನೆ. ಕೆಲವು ನಿಮಿಷ ಕಾಯಿರಿ.",
    "verdict_trust": "ಸರಿ ಸುದ್ದಿ: ಇದು ನೈಜವಾಗಿ ಕಾಣುತ್ತಿದೆ. ಬದಲಾವಣೆಯ ಅಥವಾ ಮೋಸದ ಯಾವ ಚಿಹ್ನೆಯೂ ಕಂಡುಬಂದಿಲ್ಲ.",
    "verdict_caution": "ಎಚ್ಚರಿಕೆ: ಕೆಲವು ವಿವರಗಳು ಅಸಾಮಾನ್ಯವಾಗಿ ಕಾಣುತ್ತಿವೆ. ನೀವು ಆಧಿಕಾರಿತಮೂಲದಿಂದ ಸ್ವಯಂ ಪರಿಶೀಲಿಸುವ ವರೆಗೆ, ಹಣ ಅಥವಾ ವ್ಯಕ್ತಿಗತ ಮಾಹಿತಿ ಹಂಚಬೇಡಿ. [?]",
    "verdict_do_not_use": "ಎಚ್ಚರಿಕೆ: ಇದು ಮೋಸ ಅಥವಾ ಹಾನಿಕರವಾದದ್ದಿನಂತೆ ಕಾಣುತ್ತಿದೆ. ಇದನ್ನು ತೆರೆಯಬೇಡಿ, ಲಿಂಕ್‌ಗಳನ್ನು ಒತ್ತಬೇಡಿ, ಸಂದೇಶವನ್ನು ಅಳಿಸಿ. ನೀವು ಈಗಾಗಲೇ ಹಣ ಕಳುಹಿಸಿದರೆ, ತಕ್ಷಣ ಬ್ಯಾಂಕಿಗೆ ಕರೆ ಮಾಡಿ. [?]",
    "verdict_unable": "ಇದನ್ನು ಈಗ ಪೂರ್ಣವಾಗಿ ಪರೀಕ್ಷಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ಯಾವುದನ್ನಾದರೂ ಮಾಡುವ ಮೊದಲು, ಆಧಿಕಾರಿತಮೂಲದ ವೆಬ್‌ಸೈಟ್ ಅಥವಾ ಆ್ಯಪ್‌ನಲ್ಲಿ ನೇರವಾಗಿ ಪರೀಕ್ಷಿಸಿ. [?]",
    "confidence_line": "ನನ್ನ ನಂಬಿಕೆ: %(conf)s. ಈ ಸಾಧನ ಸಹಾಯ ಮಾಡುತ್ತದೆ, ಆದರೆ ಜನರ ಮರು ಪರಿಶೀಲನೆ ಯಾವಾಗಲೂ ಕൂಡೂ ಸುರಕ್ಷಿತ.",
    "advice_avoid_links": "ಸೂಚನೆ: ಅಪರಿಚಿತ ಸಂಖ್ಯೆಗಳಿಂದ ಬರುವ ಲಿಂಕ್‌ಗಳನ್ನು ಎಂದಿಗೂ ಒತ್ತಬೇಡಿ. ಸರ್ಕಾರಿ ಸಂಸ್ಥೆಗಳು WhatsApp ಲಿಂಕ್‌ನಲ್ಲಿ ಪಾಸ್‌ವರ್ಡ್ ಕೇಳುವುದಿಲ್ಲ.",
    "progress_file": "ಫೈಲ್ (%(name)s) ಸ್ಕಾನ್ ಮಾಡುತ್ತಿದ್ದೇನೆ.",
    "progress_url": "ಲಿಂಕ್ ವಿಶ್ಲೇಷಿಸುತ್ತಿದ್ದೇನೆ…",
    "progress_media": "ವೀಡಿಯೊ/ಆಡಿಯೊ ವಿಸ್ತಾರವಾಗಿ ಅಧ್ಯಯನ ಮಾಡುತ್ತಿದ್ದೇನೆ. ಇದಕ್ಕೆ ಕೆಲವು ಸಮಯ ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ… [?]",
    "evidence_missing": "ಅಗತ್ಯ ಸೇವೆ ಲಭ್ಯವಿರದ ಕಾರಣ ಕೆಲವು ಪರೀಕ್ಷೆಗಳು ಹಾದುಹೋದವು; ಮೇಲಿನ ಫಲಿತಾಂಶ ನಾನು ನಿಜವಾಗಿ ಪರೀಕ್ಷಿಸಿದದ್ದನ್ನು ಮಾತ್ರ ಪ್ರತಿಬಿಂಬಿಸುತ್ತದೆ. [?]",
    "heavy_pending_notice": "ವೇಗದ ಪರೀಕ್ಷೆ ಮುಗಿದಿದೆ. ನನ್ನ ಆಳ ಪರೀಕ್ಷೆ ಇನ್ನೂ ನಡೆಯುತ್ತಿದೆ — ಅದು ಮುಗಿದಾಗ ನಿಮಗೆ ಮತ್ತೊಂದು ಸಂದೇಶ ಕಳುಹಿಸುತ್ತೇನೆ. [?]",
    "heavy_followup": "ನನ್ನ ಆಳ ಪರೀಕ್ಷೆ (%(cap)s) ಮುಗಿದಿದೆ. ಫಲಿತಾಂಶ: %(verdict)s — ನಂಬಿಕೆ %(conf)s.",
}
KN_NOTES = {
    "greeting": "polite-plural form chosen; some elders expect 'ನಮಸ್ಕಾರ' with 'ನೀವು' — verify",
    "verdict_caution": "long sentence; consider splitting",
    "verdict_do_not_use": "borrowed 'ಲಿಂಕ್' kept (widely understood); verify verb chain",
    "verdict_unable": "slightly stiff; native smoothing welcome",
    "progress_media": "phrasing check needed",
    "evidence_missing": "formal; needs simplification for an elder",
    "heavy_pending_notice": "'ಇನ್ನೂ ನಡೆಯುತ್ತಿದೆ' — confirm natural spoken form",
}

BN = {
    "greeting": "হাই! আমি আপনার কীভাবে সাহায্য করতে পারি?",
    "analyzing": "এখন সেটা পরীক্ষা করছি। অনুগ্রহ করে একটু মিনিট অপেক্ষা করুন।",
    "verdict_trust": "ভালো খবর: এটা আসল মনে হচ্ছে। কোনো পরিবর্তন বা প্রতারণার লক্ষণ পাওয়া যায়নি।",
    "verdict_caution": "সতর্কতা: কিছু বিস্তারিত অস্বাভাবিক মনে হচ্ছে। আপনি নিজে সরকারি উৎস থেকে যাচাই করার আগে পর্যন্ত টাকা বা ব্যক্তিগত তথ্য শেয়ার করবেন না। [?]",
    "verdict_do_not_use": "সতর্কবার্তা: এটা প্রতারণা বা ক্ষতিকর মনে হচ্ছে। এটা খুলবেন না, কোনো লিংকে চাপ দিবেন না, বার্তা মুছে ফেলুন। আপনি ইতিমধ্যে টাকা পাঠিয়েছেন, তাহলে সঙ্গে সঙ্গে আপনার ব্যাংকে ফোন করুন। [?]",
    "verdict_unable": "আমি এখন এটা সম্পূর্ণভাবে যাচাই করতে পারিনি। যেকোনো কাজের আগে, সরকারি ওয়েবসাইটে বা আ্য্পে সরাসরি যাচাই করুন।",
    "confidence_line": "আমার আস্থা: %(conf)s। এই টুল সাহায্য করে, তবে মানুষের পুনরায় যাচাই সবসময় নিরাপদ। [?]",
    "advice_avoid_links": "পরামর্শ: অচেনা নম্বর থেকে আসা লিংকে কখনো চাপবেন না। সরকারি প্রতিষ্ঠান WhatsApp লিংকে পাসওয়ার্ড চায় না।",
    "progress_file": "ফাইল (%(name)s) স্ক্যান করছি।",
    "progress_url": "লিংক বিশ্লেষণ করছি…",
    "progress_media": "ভিডিও/অডিও ভালো করে দেখছি। এর জন্য একটু সময় লাগবে…",
    "evidence_missing": "প্রয়োজনীয় সেবা পাওয়া যায়নি বলে কিছু পরীক্ষা করা হয়নি; উপরের সিদ্ধান্ত শুধু সে কথা নির্দেশ করে যা আমি সত্যিই পরীক্ষা করতে পেরেছি। [?]",
    "heavy_pending_notice": "দ্রুত পরীক্ষা শেষ হয়ে গেল। আমার গভীর পরীক্ষা এখনও চলছে — সেটা শেষ হলে আমি আপনাকে আবার একটি বার্তা পাঠাব। [?]",
    "heavy_followup": "আমার গভীর পরীক্ষা (%(cap)s) শেষ হয়েছে। ফলাফল: %(verdict)s — আস্থা %(conf)s।",
}
BN_NOTES = {
    "verdict_caution": "long sentence; consider splitting for an elder",
    "verdict_do_not_use": "check the politeness level of 'খুলবেন না / চাপ দিবেন না' chain",
    "confidence_line": "'আস্থা' (trust) chosen over a calque of 'confidence'; confirm",
    "evidence_missing": "a bit formal; welcome a plainer rephrase",
    "heavy_pending_notice": "'আবার একটি বার্তা পাঠাব' — verify it sounds like WhatsApp speech",
}

TABLES = {"ta": TA, "te": TE, "ml": ML, "kn": KN, "bn": BN}
NOTES = {"ta": TA_NOTES, "te": TE_NOTES, "ml": ML_NOTES, "kn": KN_NOTES,
         "bn": BN_NOTES}


def write_draft(lang: str) -> tuple[int, int]:
    keys = en_keys()
    table = TABLES[lang]
    notes = NOTES[lang]
    missing = [k for k in keys if k not in table]
    assert not missing, f"{lang}: missing keys {missing}"
    name = LANG_NAMES[lang]
    qmarks = 0
    lines = [
        f"# VeriSafe i18n — {name} ({lang}) draft",
        "",
        "> **DRAFT — pending native review.** Do NOT merge into `i18n._DEFAULTS`",
        "> until a native reviewer (the DawnofGenX gate) signs off or corrects each line.",
        ">",
        "> printf-style placeholders are preserved **verbatim** and must never be",
        "> translated, altered, or reordered: `%(conf)s`, `%(name)s`, `%(cap)s`, `%(verdict)s`.",
        ">",
        "> Audience: a ~65-year-old non-technical elder. Warm, plain, respectful;",
        "> short sentences; no tech jargon. Any line ending in `[?]` is a best-effort",
        "> attempt with sub-90% confidence — please double-check it specifically.",
        ">",
        f"> Source corpus: `docs/i18n/en.md` ({len(keys)} keys). This draft covers all of them.",
        "",
        f"| Key | {name} text | Reviewer note |",
        "|---|---|---|",
    ]
    for k in keys:
        val = table[k]
        if val.rstrip().endswith("[?]"):
            qmarks += 1
        disp = val.replace("|", "\\|")
        note = notes.get(k, "").replace("|", "\\|")
        lines.append(f"| `{k}` | {disp} | {note} |")
    lines.append("")
    path = OUT_DIR / f"{lang}.draft.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(keys), qmarks


def main() -> None:
    keys = en_keys()
    print(f"en.md provides {len(keys)} keys: {keys}")
    for lang in ("ta", "te", "ml", "kn", "bn"):
        nkeys, q = write_draft(lang)
        print(f"wrote docs/i18n/{lang}.draft.md  ({nkeys} keys, {q} lines flagged [?])")


if __name__ == "__main__":
    main()
