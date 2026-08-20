"""VeriSafe application entrypoints.

Run modes:
  cli      — send one ad-hoc item through the pipeline (dev/QA smoke)
             verisafe-cli --text "https://example.com" 
             verisafe-cli --file /path/to/sample.apk [--lang hi]
             verisafe-cli --greet                (exercise greeting path only)
  webhook  — HTTP server speaking the OpenWA webhook contract on :PORT
             POST /health        -> {status:"ok"}
             POST /webhook/inbound -> parse event, run pipeline, reply via client
All heavy deps are environment-gated; nothing here requires more than the
Python stdlib to start (missing tools are reported per-check as unavailable).
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verisafe.capabilities import default_capabilities
from verisafe.channels import (MessageProcessor, OpenWAClient, parse_openwa_webhook,
                               verify_openwa_signature)
from verisafe.events import InputType
from verisafe.fusion import FusionEngine, ReliabilityGate
from verisafe.report import ReportBuilder

log = logging.getLogger("verisafe.app")


def detect_available_deps() -> set[str]:
    """Honest capability inventory for the availability gate (P8 input)."""
    deps: set[str] = set()
    env = os.environ

    def has_bin(name: str) -> bool:
        import shutil
        return shutil.which(env.get(f"VERISAFE_{name.upper()}_BIN", name)) is not None

    if env.get("VERISAFE_VT_API_KEY"):
        deps.add("vt")
    if env.get("OPENAI_API_KEY") or env.get("VERISAFE_LLM_BASE_URL"):
        deps.add("llm")
    if has_bin("ffmpeg") or env.get("VERISAFE_FFMPEG_BIN"):
        deps.add("media-tools")
    if has_bin("tesseract"):
        deps.add("ocr")
    if has_bin("clamscan"):
        deps.add("clamav")
    if has_bin("strace"):
        deps.add("strace")
    try:
        import importlib.util
        for mod, dep in (("playwright", "browser"), ("cv2", "cv2"), ("docling", "docling"),
                         ("yara_x", "yara"), ("pefile", "pe-static"), ("lief", "pe-lief")):
            if importlib.util.find_spec(mod) is not None:
                deps.add(dep)
        if (importlib.util.find_spec("asn1crypto") is not None
                and importlib.util.find_spec("cryptography") is not None):
            deps.add("pades")
    except Exception:
        pass
    # RAG template cache: present when the index file exists AND parses.
    # Retrieval-cache signal only; absence is a silent feature-OFF, never an error.
    try:
        from verisafe import rag_cache as _rc
        if _rc.available():
            deps.add("rag-cache")
    except Exception:
        pass
    for wenv in ("VERISAFE_EFFORT_WEIGHTS",
                 "VERISAFE_DEMAMBA_WEIGHTS", "VERISAFE_FAKEMAMBA_WEIGHTS",
                 "VERISAFE_AASIST_WEIGHTS", "VERISAFE_SSL_AUDIO_WEIGHTS",
                 "VERISAFE_HAVIC_WEIGHTS",
                 "VERISAFE_IMAGE_FACE_WEIGHTS"):
        if env.get(wenv) and Path(env[wenv]).exists():
            deps.add("model-weights")
            break
    if env.get("VERISAFE_CAPE_CMD") or has_bin("firejail"):
        deps.add("dynamic-sandbox")
    return deps


def build_orchestrator(deps: set[str] | None = None):
    deps = deps if deps is not None else detect_available_deps()
    log.info("available deps: %s", sorted(deps) or "(none beyond stdlib)")
    caps = default_capabilities(deps)
    fusion = FusionEngine()
    # Optional provisioned stacker checkpoints (fusion_train --synthetic/--dataset).
    # Default OFF so un-provisioned deployments keep the explicit-weight path and
    # behaviour is byte-identical; set VERISAFE_FUSION_USE_TRAINED=1 to enable.
    if os.environ.get("VERISAFE_FUSION_USE_TRAINED", "").strip().lower() \
            in ("1", "true", "yes"):
        _fdir = Path(os.environ.get("VERISAFE_FUSION_DIR")
                     or Path(__file__).resolve().parent.parent / "fusion")
        _n = fusion.load_trained(_fdir / "training")
        if _n:
            log.info("loaded %d trained fusion stack(s) from %s", _n, _fdir)
    reliability = ReliabilityGate()
    reporter = ReportBuilder()
    from verisafe.orchestrator import Orchestrator
    return Orchestrator(capabilities_by_target=caps, fusion=fusion,
                        reliability=reliability, reporter=reporter,
                        hard_budget_s=float(os.environ.get("VERISAFE_BUDGET_S", "300")),
                        available_deps=deps)


# --------------------------------------------------------------------- CLI --

def main_cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verisafe-cli", description=__doc__)
    ap.add_argument("--text", help="message text (may contain a URL)")
    ap.add_argument("--file", help="path to a media/document file to analyze")
    ap.add_argument("--media-type", choices=["image", "video", "audio", "document"], default=None)
    ap.add_argument("--url", action="store_true", help="treat --text strictly as a URL")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--greet", action="store_true", help="only exercise the greeting path")
    ap.add_argument("--deps", default="", help="comma-separated forced deps (override auto-detect)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.environ.get("VERISAFE_LOG_LEVEL", "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.greet:
        orch = build_orchestrator()
        msg = {"id": "greet-test", "session_key": "greet-test",
               "text": "hello?", "sender_lang": args.lang}
        print(MessageProcessor(orch, persist_outcomes=False).process(msg)["reply"])
        return 0

    deps = set(d.strip() for d in args.deps.split(",") if d.strip()) if args.deps else detect_available_deps()
    orch = build_orchestrator(deps)

    if args.file:
        fp = Path(args.file)
        if not fp.exists():
            print(f"error: file not found: {fp}")
            return 2
        ext = fp.suffix.lstrip(".").lower()
        mtype = args.media_type or ({".mp4", ".mov", ".mkv", ".webm"} & {fp.suffix} and "video"
                                    or {".mp3", ".wav", ".ogg", ".opus", ".m4a"} & {fp.suffix} and "audio"
                                    or {".png", ".jpg", ".jpeg", ".webp"} & {fp.suffix} and "image"
                                    or "document")
        msg = {"id": f"cli-{fp.name}", "session_key": f"cli-{fp.name}",
               "text": "", "media_path": str(fp), "media_type": mtype,
               "sender_lang": args.lang}
    else:
        if not args.text:
            ap.error("one of --text/--file/--greet is required")
        itype = "url" if args.url else ("text" if not _looks_like_url(args.text) else "url")
        msg = {"id": "cli-text", "session_key": "cli-text",
               "text": args.text, "input_type": itype, "sender_lang": args.lang}

    out = MessageProcessor(orch, persist_outcomes=False).process(msg)
    print(out["reply"])
    print("--- debug ---")
    od = out["outcome"]
    print(json.dumps({k: od[k] for k in ("job_id", "verdict", "confidence", "language", "wall_s", "purged")}, indent=2))
    print("checks:")
    for c in od["checks"]:
        print(f"  [{c['cost']:>5}] {c['name']:<32} {c['status']:<12} {json.dumps(c['signals'])[:160]}")
    print("fusion:", json.dumps(od["fusion_trace"]))
    return 0


def _looks_like_url(s: str) -> bool:
    import re
    s = (s or "").strip()
    return bool(re.match(r"^(?:https?://|www\.)[^\s]+", s, re.I))


# --------------------------------------------------------------- webhook ----

class WebhookHandler(BaseHTTPRequestHandler):
    processor: "MessageProcessor"   # injected by factory

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            body = json.dumps({"status": "ok", "deps": sorted(getattr(self, "_deps", []))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(min(length, 4 * 1024 * 1024))
        proc: MessageProcessor = self.processor
        secret = getattr(proc.openwa, "webhook_secret", "") if proc.openwa else \
            os.environ.get("OPENWA_WEBHOOK_SECRET", "")
        ok_sig = verify_openwa_signature(raw, self.headers.get("X-OpenWA-Signature"), secret) \
            if secret else True
        try:
            payload = json.loads(raw.decode(errors="ignore"))
        except json.JSONDecodeError:
            self._reply(400, {"error": "bad json"})
            return
        if not self.path.startswith("/webhook/inbound"):
            self._reply(404, {"error": "not found"})
            return
        if not ok_sig:
            log.warning("openwa webhook signature mismatch; rejecting")
            self._reply(401, {"error": "signature mismatch"})
            return
        msg = parse_openwa_webhook(payload)
        if msg is None:
            self._reply(200, {"ignored": True})
            return
        jid = msg.session_key
        # consumer-side idempotency: OpenWA delivery is at-least-once
        client = proc.openwa
        if client is not None and client._seen(f"{payload.get('idempotencyKey') or msg.id}"):
            self._reply(200, {"ignored": "duplicate"})
            return
        reply_text = ""
        try:
            workdir = proc.workdir / jid.replace("=", "").replace("@", "-")[:32]
            workdir.mkdir(parents=True, exist_ok=True)
            md = msg.as_msg()
            # materialize media: inline base64 (<=1MiB gateway cap) or marker-fetch via API
            if str(msg.media_path or "").startswith("inline-b64:"):
                blob = base64.b64decode(str(msg.media_path)[len("inline-b64:"):],
                                        validate=False)
                fp = workdir / f"whatsapp_{msg.id[:24]}.bin"
                fp.write_bytes(blob)
                md["media_path"] = str(fp)
            elif msg.fetch_media and client is not None:
                chat_id = payload.get("data", {}).get("to") or jid
                got = client.extract_media(str(chat_id), str(payload.get("data", {}).get("id") or msg.id), workdir)
                if got:
                    md["media_path"] = str(got)
            delivered = proc.process(md)
            reply_text = delivered["reply"]
        except Exception as e:  # noqa: BLE001 — never let a job kill the webhook
            log.exception("pipeline crash")
            reply_text = "Sorry, something went wrong while checking that. Please try again."
        sent = False
        if client is not None:
            try:
                sent = client.send_text(jid, reply_text)
            except Exception:
                log.warning("openwa reply failed; returning text in response body instead")
        self._reply(200, {"status": "ok", "reply_delivered": sent, "reply": reply_text})

    def _reply(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep stderr quiet; use logger
        log.debug("http: " + fmt, *args)


def main_webhook(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="OpenWA-compatible webhook server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VERISAFE_WEBHOOK_PORT", "8899")))
    ap.add_argument("--host", default=os.environ.get("VERISAFE_WEBHOOK_HOST", "0.0.0.0"))
    ap.add_argument("--openwa-url", default=None)
    ap.add_argument("--budget-s", type=float, default=300.0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.environ.get("VERISAFE_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    deps = detect_available_deps()
    openwa = OpenWAClient(base_url=args.openwa_url) if args.openwa_url else None
    orch = build_orchestrator(deps)
    # P10 fix: do not force a root-only workdir here. Omitting `workdir` lets
    # MessageProcessor use its portable /tmp default (or $VERISAFE_WORKDIR if
    # the operator set one), so the webhook runs under any user account.
    proc = MessageProcessor(orch, openwa=openwa)
    Handler = type("BoundWebhookHandler", (WebhookHandler,), {
        "processor": proc, "_deps": deps,
    })
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    log.info("verisafe webhook listening on %s:%s (deps=%s)", args.host, args.port, sorted(deps) or "[]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "cli":
        return main_cli(rest)
    if cmd == "webhook":
        return main_webhook(rest)
    print(f"unknown command: {cmd} (use 'cli' or 'webhook')")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
