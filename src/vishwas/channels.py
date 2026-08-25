"""Channel layer: OpenWA (WhatsApp) transport + in-process CLI simulator.

Target: rmyndharis/OpenWA v0.21.0+ (MIT, NestJS, Node 22). Verified against the
project's published openapi.json (2026-08-19), NOT against older community forks:

  * REST base   : /api/sessions/{sessionId}/...      (port 2785 default)
  * auth        : X-API-Key header (role OPERATOR or higher)
  * reply       : POST /api/sessions/{s}/messages/send-text  {"chatId":"E.164@c.us","text"}
  * inbound media: GET /api/sessions/{s}/messages/{chatId}/{messageId}/media
                   (application/octet-stream; webhook payloads OMIT blobs >1MiB and
                    replace them with a marker {mimetype,filename,omitted,sizeBytes})
  * webhook     : per-session registration POST /api/sessions/{s}/webhooks
                   envelope {"event","timestamp","sessionId","idempotencyKey",
                             "deliveryId","data":{...}}
                   HMAC header X-OpenWA-Signature: sha256=<hex> over the RAW body
                   dedupe header X-OpenWA-Idempotency-Key (stable across retries)

Replies go out as plain text by design (short sentences for non-technical users);
results reference what was checked, never raw forensics dumps. No credentials are
ever written into logs or reports.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("vishwas.channel")

# mime prefix -> canonical media_type used by the router
_MIME_KIND = {"image": "image", "video": "video", "audio": "audio",
              "application": "document", "text": "document", "font": "document",
              "model": "document", "octet-stream": "document", "sticker": "image"}


def _kind_for(mime_or_type: str) -> str:
    """Map a mimetype OR OpenWA message type ('image','video',...) to our 4 media types."""
    s = (mime_or_type or "").strip().lstrip(".").lower()
    if "/" in s:  # real mime like image/jpeg
        return _MIME_KIND.get(s.split("/", 1)[0], "document")
    ext = s.rsplit("=", 1)[-1]
    if ext in {"png", "jpg", "jpeg", "webp", "gif", "heic"}:
        return "image"
    if ext in {"mp4", "mov", "mkv", "webm", "3gp", "avi"}:
        return "video"
    if ext in {"mp3", "wav", "ogg", "opus", "m4a", "aac", "flac", "amr"}:
        return "audio"
    if ext in ("image", "sticker"):
        return "image"
    if ext in ("video", "audio"):
        return ext
    return "document"


@dataclass(slots=True)
class InboundMessage:
    id: str                      # stable dedupe id (OpenWA idempotencyKey or generated)
    session_key: str            # sender jid (E.164@c.us or <num>@g.us)
    text: str
    media_path: str | None      # extracted media file path, or None when a fetch is required
    media_type: str             # image|video|audio|document|'' (no media)
    sender_lang: str | None
    fetch_media: bool = False   # True => OpenWAClient.extract_media() must run

    def as_msg(self) -> dict:
        d: dict[str, Any] = {"id": self.id, "session_key": self.session_key,
                             "text": self.text, "media_path": self.media_path,
                             "media_type": self.media_type, "sender_lang": self.sender_lang}
        return {k: v for k, v in d.items() if v is not None}


class ChannelError(RuntimeError):
    pass


class OpenWAClient:
    """Client for the real OpenWA v0.21.x REST surface.

    All settings come from env; nothing secret is logged. Methods raise
    ChannelError on transport failure so callers can degrade gracefully.
    """

    def __init__(self, base_url: str | None = None, api_token: str | None = None,
                 session_id: str | None = None, timeout_s: int = 30,
                 webhook_secret: str | None = None):
        self.base_url = (base_url or os.environ.get("OPENWA_BASE_URL", "http://localhost:2785")).rstrip("/")
        self.api_token = api_token or os.environ.get("OPENWA_API_KEY", "")
        self.session_id = session_id or os.environ.get("OPENWA_SESSION_ID", "main")
        self.webhook_secret = webhook_secret or os.environ.get("OPENWA_WEBHOOK_SECRET", "")
        self.timeout_s = timeout_s
        self._dedupe: dict[str, float] = {}

    # ------------------------------------------------------------- plumbing --
    def _req(self, method: str, path: str, body: Any = None, raw: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        if self.api_token:
            headers["X-API-Key"] = self.api_token
        data = json.dumps(body).encode() if body is not None else None
        if data:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                blob = r.read()
        except Exception as e:  # noqa: BLE001 — channel failures must not crash analysis
            log.warning("openwa %s %s failed: %s", method, path, e.__class__.__name__)
            raise ChannelError(f"openwa {method} {path}: {e.__class__.__name__}") from e
        if raw:
            return blob
        txt = blob.decode(errors="ignore")
        return json.loads(txt) if txt.strip() else {}

    def _seen(self, key: str) -> bool:
        """At-least-once webhooks => consumer-side idempotency (docs §delivery)."""
        now = time.time()
        self._dedupe = {k: t for k, t in self._dedupe.items() if now - t < 3600}
        if key in self._dedupe:
            return True
        self._dedupe[key] = now
        return False

    # -------------------------------------------------------------- outbound --
    def send_text(self, chat_id: str, text: str) -> bool:
        """POST /api/sessions/{s}/messages/send-text (text capped at 4096 upstream)."""
        if not chat_id or not text:
            return False
        ok = True
        for i in range(0, max(1, len(text)), 4000):  # stay under the 4096 cap
            chunk = text[i:i + 4000]
            try:
                self._req("POST", f"/api/sessions/{self.session_id}/messages/send-text",
                          {"chatId": chat_id, "text": chunk, "linkPreview": False})
            except ChannelError:
                ok = False
                break
        if not ok:
            log.error("send_text to %s rejected after retry", chat_id)
        return ok

    # --------------------------------------------------------------- inbound --
    def extract_media(self, chat_id: str, msg_id: str, outdir: Path,
                      max_bytes: int = 64_000_000) -> Path | None:
        """GET /api/sessions/{s}/messages/{chatId}/{msgId}/media -> quarantine area."""
        path = (f"/api/sessions/{self.session_id}/messages/"
                f"{urllib.parse.quote(str(chat_id), safe='')}/"
                f"{urllib.parse.quote(str(msg_id), safe='')}/media")
        try:
            blob = self._req("GET", path, raw=True)
        except ChannelError:
            return None
        if not blob:
            return None
        fp = outdir / f"whatsapp_{str(msg_id)[:16]}.bin"
        fp.write_bytes(blob[:max_bytes])
        return fp

    def poll_inbound(self, chat_id: str, since_ts: float) -> list[dict]:
        """Dev fallback: history endpoint (webhooks being unconfigured)."""
        try:
            r = self._req("GET",
                          f"/api/sessions/{self.session_id}/messages/{urllib.parse.quote(chat_id, safe='')}/history")
        except ChannelError:
            return []
        msgs = r if isinstance(r, list) else (r.get("messages") or [])
        self_jid = os.environ.get("OPENWA_SELF_JID", "")  # own-number jid for echo filter
        out = []
        for m in msgs:
            if m.get("fromMe"):
                continue  # own echoes never act
            if self_jid and str(m.get("to")) == self_jid:
                continue  # sent-from-us message (own send echo without fromMe flag)
            ts = m.get("timestamp") or 0
            if ts and ts <= since_ts:
                continue
            out.append(m)
        return out


# ------------------------------------------------------- webhook verification --

def verify_openwa_signature(raw_body: bytes, signature_header: str | None,
                            secret: str) -> bool:
    """Verify X-OpenWA-Signature ('sha256=<hex>' over the RAW request bytes).

    Returns False (reject) when the secret is configured but the header is missing
    or mismatched. Constant-time compare; recomputed over exact bytes, never over
    a re-serialized parse (OpenWA docs §HMAC).
    """
    if not secret:
        return True  # unsigned webhook => accepted (operator choice)
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header.strip())


def parse_openwa_webhook(payload: dict) -> InboundMessage | None:
    """Map an OpenWA v0.21 'message.received' envelope to our canonical message.

    Envelope: {event, timestamp, sessionId, idempotencyKey, deliveryId, data}
    data: {id, from, to, body, type, timestamp(epoch-sec), isGroup, kind,
           hasMedia, contact?, media?} — media either an inline base64 blob (<=1MiB
           by gateway cap) or the omitted-marker {mimetype, filename?, omitted, sizeBytes}.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get("event") not in ("message.received", "MESSAGE_RECEIVED"):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    jid = str(data.get("from") or "unknown")
    dedupe = (payload.get("idempotencyKey")
              or data.get("id")
              or f"{jid}:{(data.get('timestamp') or 0)}:{(data.get('body') or '')[:12]}")
    text = data.get("body") or ""
    md = data.get("media") if isinstance(data.get("media"), dict) else None
    media_path: str | None = None
    fetch_media = False
    mtype = ""
    if md is not None:
        if md.get("omitted"):
            fetch_media = True  # >1MiB marker — pull bytes via the media GET route
        else:
            b64 = md.get("data")
            if isinstance(b64, str) and b64:
                media_path = f"inline-b64:{b64}"  # caller decodes into quarantine
        src = str(md.get("mimetype") or data.get("type") or "")
        fname = md.get("filename") or ""
        if "/" in src:
            mtype = _kind_for(src)
        else:
            mtype = {"sticker": "image", "image": "image", "video": "video",
                     "audio": "audio", "document": "document"}.get(src,
                            _kind_for(fname.rsplit(".", 1)[-1]) if "." in fname else "document")
    elif data.get("type") in ("image", "video", "audio", "document", "sticker"):
        mtype = "image" if data.get("type") == "sticker" else data.get("type")
        fetch_media = True
    return InboundMessage(id=str(dedupe), session_key=jid, text=text,
                          media_path=media_path, media_type=mtype or "",
                          sender_lang=None, fetch_media=fetch_media)


# ------------------------------------------------------------------ runner --

class JobCounters:
    """Thread-safe monotone job counters surfaced by GET /health.

    In-memory only: counters reset to zero when the process restarts.
    Durable per-job evidence lives in outcomes.jsonl (see _persist), not here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._ok = 0
        self._failed = 0

    def record_started(self) -> None:
        with self._lock:
            self._total += 1

    def record_ok(self) -> None:
        with self._lock:
            self._ok += 1

    def record_failed(self) -> None:
        with self._lock:
            self._failed += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"jobs_total": self._total, "jobs_ok": self._ok,
                    "jobs_failed": self._failed}


class MessageProcessor:
    """Binds orchestrator + channels + greeting/session state together."""

    def __init__(self, orchestrator, openwa: OpenWAClient | None = None,
                 workdir: Path | None = None, persist_outcomes: bool = True):
        self.orch = orchestrator
        self.openwa = openwa
        self.workdir = workdir or Path(os.environ.get("VISHWAS_WORKDIR", "/tmp/vishwas-work"))
        self.session_state: dict[str, dict] = {}
        self.persist_outcomes = persist_outcomes
        self.outcome_log: Path | None = None
        # 2.3 ops: /health job counters (ok/failed classification below).
        self.counters = JobCounters()
        # 2.1: sessions explicitly closed while a heavy stage is still running;
        # follow-ups addressed to them are dropped silently (logged, no retry).
        self.closed_sessions: set[str] = set()
        # every composed follow-up that reached the sender (for CLI/tests)
        self.followups: list[dict] = []

    def end_session(self, sid: str) -> None:
        """Mark a chat session closed: later heavy-stage follow-ups drop."""
        self.closed_sessions.add(sid)

    def process(self, msg_dict: dict) -> dict:
        """Returns the delivery payload {jid, reply, verdict...} for this message."""
        from .orchestrator import maybe_greet
        greeting = maybe_greet(msg_dict, self.session_state)
        sid = msg_dict.get("session_key") or msg_dict.get("id") or "anon"
        st = self.session_state.setdefault(sid, {})
        qwork = self.workdir / sid[:24]
        qwork.mkdir(parents=True, exist_ok=True)

        def _followup_sender(chat_id: str, text: str) -> bool:
            if chat_id in self.closed_sessions:
                print(f"[vishwas] session {chat_id} ended before its deep check "
                      f"finished; heavy follow-up dropped silently", flush=True)
                return False
            self.followups.append({"jid": chat_id, "reply": text,
                                   "ts_mono": time.monotonic()})
            return self.deliver({"jid": chat_id, "reply": text})

        # 2.3 ops: every pipeline run is one job. ok = orchestrator returned an
        # outcome; failed = it raised (the webhook turns that into a generic
        # apology reply). Counters are in-memory and reset on restart.
        self.counters.record_started()
        try:
            outcome = self.orch.handle_incoming({**msg_dict, "_qroot_override": str(qwork)},
                                                followup_sender=_followup_sender)
        except Exception:
            self.counters.record_failed()
            raise
        self.counters.record_ok()
        try:
            replies: list[str] = []
            if greeting:
                replies.append(greeting)
            replies.append(outcome.user_message)
            reply = "\n".join(replies)
            st["last_replied_ts"] = time.time()
            st["last_activity_mono"] = time.monotonic()
            self._persist(outcome)
            return {"jid": sid, "reply": reply, "outcome": outcome.to_dict()}
        finally:
            # zero-retention hygiene (P10): remove per-session staging shells
            # that are now EMPTY after the job purge. Never removes
            # non-empty directories, so the outcomes.jsonl evidence log and
            # any in-flight concurrent job stay intact.
            for d in (qwork, self.workdir):
                try:
                    if d.is_dir() and not any(d.iterdir()):
                        d.rmdir()
                except OSError:
                    pass

    def deliver(self, delivered: dict) -> bool:
        """Push the reply through OpenWA when available; else store locally."""
        if self.openwa:
            try:
                return self.openwa.send_text(delivered["jid"], delivered["reply"])
            except ChannelError:
                log.exception("delivery failed; storing locally only")
        if self.persist_outcomes:
            if self.outcome_log is None:
                self.workdir.mkdir(parents=True, exist_ok=True)
            self.outcome_log = self.workdir / "outcomes.jsonl"
            rec = {"ts": time.time(), "jid": delivered["jid"], "reply": delivered["reply"]}
            with self.outcome_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return False

    def _persist(self, outcome) -> None:
        """Append a REDUCED outcome record (privacy: retention policy 2026-08-25).

        Zero-retention posture: we keep only what ops-troubleshooting needs —
        timestamps, verdict, per-check (name, status) and small scalar signals.
        Dropped: sender JID (hashed), full reply text, page titles / redirect
        chains / clam signature strings / any list-valued signal. Records past
        VISHWAS_OUTCOMES_TTL_S (default 7 d) are pruned opportunistically here.
        """
        if not self.persist_outcomes:
            return
        if self.outcome_log is None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        self.outcome_log = self.workdir / "outcomes.jsonl"
        self._prune_outcomes()
        import hashlib
        full = outcome.to_dict()
        jid = getattr(outcome, "jid", "") or ""
        jid_hash = hashlib.sha256(jid.encode()).hexdigest()[:12] if jid else ""
        reduced_checks = []
        for c in full.get("checks", []):
            sig_small = {k: v for k, v in (c.get("signals") or {}).items()
                         if isinstance(v, (int, float, bool))}
            reduced_checks.append({"name": c.get("name"), "status": c.get("status"),
                                   "signals": sig_small})
        rec = {"ts": time.time(), "jid_h": jid_hash,
               "target": full.get("target"), "verdict": full.get("verdict"),
               "confidence": full.get("confidence"), "checks": reduced_checks}
        with self.outcome_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _prune_outcomes(self) -> None:
        """Drop outcome records older than the TTL (default 7 days)."""
        try:
            ttl = int(os.environ.get("VISHWAS_OUTCOMES_TTL_S", 7 * 86400))
            if not self.outcome_log.exists() or ttl <= 0:
                return
            cutoff = time.time() - ttl
            kept, changed = [], False
            for line in self.outcome_log.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    if float(rec.get("ts", 0)) >= cutoff:
                        kept.append(line)
                    else:
                        changed = True
                except Exception:
                    kept.append(line)
            if changed:
                self.outcome_log.write_text("\n".join(kept) + ("\n" if kept else ""),
                                            encoding="utf-8")
        except Exception:  # noqa: BLE001 — pruning must never break persistence
            log.exception("outcomes prune failed; continuing")


class CLISimulator:
    """Development stand-in for WhatsApp so the full flow runs headless."""

    @staticmethod
    def run_once(orchestrator, msg_dict: dict) -> dict:
        proc = MessageProcessor(orchestrator, openwa=None, persist_outcomes=False)
        res = proc.process(msg_dict)
        # 2.1: if a heavy stage is still running, wait and print its
        # deterministic template follow-up inline (no-op transport works the
        # same way — composition is independent of delivery).
        waiter = getattr(orchestrator, "wait_for_pending_followups", None)
        if waiter is not None:
            waiter(timeout_s=float(os.environ.get("VISHWAS_FOLLOWUP_WAIT_S", "120")))
        if proc.followups:
            for fu in proc.followups:
                print(fu["reply"])
            res["followups"] = [{"jid": f["jid"], "reply": f["reply"]}
                                for f in proc.followups]
        return res
