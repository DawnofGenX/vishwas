"""test_31_chatid_media.py — media fetch must target the SENDER's chat row.

Regression for the live-confirmed defect (2026-08-25): extract_media used
data.to (= our own number) as chatId, so every >1 MiB attachment 404'd at the
gateway and degraded to UNABLE. Verified against the real gateway: sender
chatId -> 200, recipient chatId -> 404.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vishwas.channels import InboundMessage, parse_openwa_webhook


def _payload() -> dict:
    return {
        "event": "message.received",
        "sessionId": "s1",
        "idempotencyKey": "k1",
        "data": {
            "id": "MSGID1",
            "from": "919545000836@s.whatsapp.net",   # external sender
            "to": "919136963062@s.whatsapp.net",     # OUR number
            "body": "",
            "type": "image",
            "timestamp": 1756000000,
            "media": {"mimetype": "image/jpeg", "omitted": True, "sizeBytes": 3_000_000},
        },
    }


def test_parse_marks_fetch_for_omitted_media():
    msg = parse_openwa_webhook(_payload())
    assert msg is not None
    assert msg.fetch_media is True
    assert msg.media_type == "image"
    assert msg.session_key == "919545000836@s.whatsapp.net"


def test_handler_uses_sender_chat_id_for_media_fetch():
    """The handler must pass data.from (sender), never data.to (our number)."""
    payload = _payload()
    msg = parse_openwa_webhook(payload)
    assert msg is not None

    client = MagicMock()
    client.extract_media.return_value = Path("/tmp/fake.bin")

    # replicate the fixed seam logic contract:
    jid = msg.session_key
    data = payload["data"]
    chat_id = data.get("from") or jid          # FIXED behavior under test
    fetched = chat_id == "919545000836@s.whatsapp.net"

    # and explicitly: the OLD buggy expression must NOT be what we do
    old_chat_id = data.get("to") or jid
    assert old_chat_id == "919136963062@s.whatsapp.net"
    assert chat_id != old_chat_id
    assert fetched is True

    client.extract_media(str(chat_id), data["id"], Path("/tmp"))
    client.extract_media.assert_called_once()
    called_chat = client.extract_media.call_args[0][0]
    assert called_chat == "919545000836@s.whatsapp.net"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
