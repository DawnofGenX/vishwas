"""Minimal i18n layer for user-facing strings (WhatsApp replies).

Design rule for non-technical elders: ONE short verdict sentence, ONE action
sentence. Detector jargon never reaches the user.

Status: `en` is authoritative; `hi` is best-effort and needs native review;
other script keys exist for routing but fall back to English until reviewed.
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
    },
    "analyzing": {
        "en": "Checking it now. Please wait about a minute.",
        "hi": "अभी जाँच हो रही है। कृपया एक मिनट रुकिए।",
    },
    "verdict_trust": {
        "en": "Good news: this looks genuine. I found no signs of tampering or fraud.",
        "hi": "सुखबत: यह असली लग रहा है। बदलाव या धोखाधड़ी के कोई चिह्न नहीं मिले।",
    },
    "verdict_caution": {
        "en": "Caution: some details look unusual. Do not pay money or share personal information until you confirm with the official source yourself.",
        "hi": "सावधानी: कुछ बातें अजीब लग रही हैं। अपने स्वयं आधिकारिक स्रोत से पुष्टि करने तक पैसे या व्यक्तिगत जानकारी न भेजें।",
    },
    "verdict_do_not_use": {
        "en": "Warning: this looks like a scam or harmful item. Do not open it, do not press any links, delete the message. If you already sent money, call your bank right away.",
        "hi": "चेतावनी: यह धोखाधड़ी या हानिकारक लग रहा है। इसे खोलना न करें, किसी लिंक पर दबाएँ न, संदेश हटा दें। अगर पहले ही पैसे भेज चुके हैं तो तुरंत अपने बैंक को फोन कीजिए।",
    },
    "verdict_unable": {
        "en": "I could not fully verify this right now. Please check it directly on the official website or app before doing anything.",
        "hi": "अभी इसका पूरा सत्यापन नहीं कर सका। किसी भी काम से पहले आधिकारिक वेबसाइट या ऐप पर स्वयं जाँच कर लें।",
    },
    "confidence_line": {
        "en": "My confidence: %(conf)s. This tool helps, but a human double-check is always safer.",
        "hi": "मेरी विश्वसनीयता: %(conf)s। यह टूल मदद करता है, लेकिन इंसानी डबल-चेक हमेशा बेहतर है।",
    },
    "advice_avoid_links": {
        "en": "Tip: never click links from unknown numbers. Official institutions never ask for passwords on WhatsApp links.",
        "hi": "सलाह: अज्ञात नंबरों से लिंक खोलना न करें। सरकारी संस्थाएं व्हाट्सऐप लिंक पर कभी पासवर्ड नहीं माँगतीं।",
    },
    "progress_file": {
        "en": "Scanning the file (%(name)s)…",
        "hi": "फ़ाइल की स्कैनिंग जारी है (%(name)s)…",
    },
    "progress_url": {
        "en": "Analysing the link…",
        "hi": "लिंक का विश्लेषण हो रहा है…",
    },
    "progress_media": {
        "en": "Studying the video/audio in detail. This takes a little longer…",
        "hi": "वीडियो/ऑडियो का विस्तृत अध्ययन हो रहा है। यह थोड़ा समय लेगा…",
    },
    "evidence_missing": {
        "en": "Some checks were skipped because a required service was unavailable; the verdict above reflects only what I could actually test.",
        "hi": "कुछ जाँचें छोड़नी पड़ीं क्योंकि आवश्यक सेवा उपलब्ध नहीं थी; ऊपर वाला नतीजा सिर्फ इतने पर आधारित है जितना मैंने वास्तव में जाँचा।",
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
