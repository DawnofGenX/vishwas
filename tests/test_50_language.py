"""Language-selection feature (2026-08-27).

Users can set their reply language explicitly instead of it being re-detected
from every message, and a language request is intercepted BEFORE the scam-check
pipeline so it is never treated as content to verify.
"""
from vishwas.i18n import parse_language_request as parse, language_menu_text
from vishwas.channels import MessageProcessor


# ------------------------------------------------------------ parser tests ---
def test_bare_language_name_switches():
    assert parse("hindi") == ("set", "hi")
    assert parse("తెలుగు") == ("set", "te")
    assert parse("bangla") == ("set", "bn")


def test_trigger_word_alone_offers_menu():
    assert parse("language") == ("menu",)
    assert parse("भाषा") == ("menu",)


def test_trigger_with_language_switches():
    assert parse("language hindi") == ("set", "hi")
    assert parse("change language to tamil") == ("set", "ta")
    assert parse("set language bn") == ("set", "bn")


def test_menu_reply_number_when_awaiting():
    assert parse("2", awaiting=True) == ("set", "hi")   # 2nd in menu order
    assert parse("1", awaiting=True) == ("set", "en")


def test_bare_number_ignored_when_not_awaiting():
    assert parse("2", awaiting=False) is None


def test_greeting_and_content_not_hijacked():
    assert parse("hi") is None          # greeting, not "set English"
    assert parse("hello") is None
    assert parse("England is nice") is None
    assert parse("is this link safe? http://x.com/pay") is None
    assert parse("a" * 60) is None      # long content is never a command


def test_menu_lists_all_languages():
    menu = language_menu_text("en")
    for name in ("English", "हिंदी", "தமிழ்", "বাংলা"):
        assert name in menu


# ------------------------------------------------- MessageProcessor tests ----
class _FakeOutcome:
    def __init__(self, lang):
        self.language = lang
        self.user_message = f"[analysis in {lang}]"
    def to_dict(self):
        return {"language": self.language}


class _SpyOrch:
    """Records the sender_lang the pipeline was invoked with."""
    def __init__(self):
        self.last_lang = None
    def handle_incoming(self, msg, followup_sender=None):
        self.last_lang = msg.get("sender_lang")
        return _FakeOutcome(self.last_lang or "auto")


def _mp(tmp_path):
    return MessageProcessor(_SpyOrch(), openwa=None, workdir=tmp_path,
                            persist_outcomes=False)


def test_language_command_skips_pipeline(tmp_path):
    mp = _mp(tmp_path)
    r = mp.process({"id": "u1", "session_key": "u1", "text": "language hindi"})
    assert r.get("language_command") is True
    assert r["outcome"] is None                 # pipeline NOT run
    assert mp.orch.last_lang is None            # orchestrator never called
    assert mp.session_state["u1"]["lang"] == "hi"


def test_menu_then_number_sets_language(tmp_path):
    mp = _mp(tmp_path)
    r1 = mp.process({"id": "u2", "session_key": "u2", "text": "language"})
    assert mp.session_state["u2"]["awaiting_lang"] is True
    assert "1. English" in r1["reply"]
    r2 = mp.process({"id": "u2", "session_key": "u2", "text": "3"})  # Tamil
    assert r2.get("language_command") is True
    assert mp.session_state["u2"]["lang"] == "ta"
    assert not mp.session_state["u2"].get("awaiting_lang")


def test_sticky_language_applied_to_analysis(tmp_path):
    mp = _mp(tmp_path)
    mp.process({"id": "u3", "session_key": "u3", "text": "language hindi"})
    # a following real message (a URL) must be analysed with the chosen language
    mp.process({"id": "u3", "session_key": "u3", "text": "http://example.com"})
    assert mp.orch.last_lang == "hi"


def test_content_message_still_analysed(tmp_path):
    mp = _mp(tmp_path)
    r = mp.process({"id": "u4", "session_key": "u4",
                    "text": "please check http://example.com/pay"})
    assert r.get("language_command") is not True
    assert r["outcome"] is not None             # pipeline DID run
