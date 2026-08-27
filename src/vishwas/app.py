"""Vishwas application entrypoints.

Run modes:
  cli      — send one ad-hoc item through the pipeline (dev/QA smoke)
             vishwas-cli --text "https://example.com" 
             vishwas-cli --file /path/to/sample.apk [--lang hi]
             vishwas-cli --greet                (exercise greeting path only)
             vishwas-cli --doctor               (report provisioned detectors + how to enable the rest)
  webhook  — HTTP server speaking the OpenWA webhook contract on :PORT
             GET  /health          -> rich JSON ops snapshot (see health_snapshot)
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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vishwas.capabilities import default_capabilities
from vishwas.channels import (MessageProcessor, OpenWAClient, parse_openwa_webhook,
                               verify_openwa_signature)
from vishwas.events import InputType
from vishwas.fusion import FusionEngine, ReliabilityGate
from vishwas.report import ReportBuilder

log = logging.getLogger("vishwas.app")


def detect_available_deps() -> set[str]:
    """Honest capability inventory for the availability gate (P8 input)."""
    deps: set[str] = set()
    env = os.environ

    def has_bin(name: str) -> bool:
        import shutil
        return shutil.which(env.get(f"VISHWAS_{name.upper()}_BIN", name)) is not None

    if env.get("VISHWAS_VT_API_KEY"):
        deps.add("vt")
    if env.get("OPENAI_API_KEY") or env.get("VISHWAS_LLM_BASE_URL"):
        deps.add("llm")
    if has_bin("ffmpeg") or env.get("VISHWAS_FFMPEG_BIN"):
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
        from vishwas import rag_cache as _rc
        if _rc.available():
            deps.add("rag-cache")
    except Exception:
        pass
    for wenv in ("VISHWAS_EFFORT_WEIGHTS",
                 "VISHWAS_DEMAMBA_WEIGHTS", "VISHWAS_FAKEMAMBA_WEIGHTS",
                 "VISHWAS_AASIST_WEIGHTS", "VISHWAS_SSL_AUDIO_WEIGHTS",
                 "VISHWAS_XLSRMAMBA_WEIGHTS",  # was missing: xlsr-mamba alone
                 "VISHWAS_HAVIC_WEIGHTS",      # would not flip the model-weights gate
                 "VISHWAS_IMAGE_FACE_WEIGHTS"):
        if env.get(wenv) and Path(env[wenv]).exists():
            deps.add("model-weights")
            # Finding A fix (2026-08-21): weights present but torch broken
            # (missing __init__ / _prims_common — e.g. the half-installed user
            # site-packages tree) means every learned tier fast-fails while
            # this dep claims ready. Probe that torch actually WORKS (import
            # torch.nn, not just the namespace) before claiming readiness.
            try:
                import importlib.util as _ilu
                if _ilu.find_spec("torch") is not None:
                    try:
                        import torch.nn  # noqa: F401 — real usability probe
                    except Exception:
                        log.warning(
                            "model-weights env vars are set but torch is BROKEN "
                            "(namespace imports, torch.nn fails — likely a stale "
                            "~/.local site-packages). Learned tiers will fast-fail "
                            "to 'unavailable'. Fix: remove/move the stale "
                            "~/.local/.../torch dir or launch via "
                            "scripts/run_vishwas.sh with docling-python first on "
                            "PYTHONPATH.")
                        deps.discard("model-weights")
                else:
                    log.warning(
                        "model-weights env vars are set but torch is NOT importable — "
                        "learned tiers will fast-fail to 'unavailable'. Launch via "
                        "scripts/run_vishwas.sh (wires pylibs/docling-python paths) "
                        "or fix PYTHONPATH.")
                    deps.discard("model-weights")
            except Exception:
                pass
            break
    if env.get("VISHWAS_CAPE_CMD") or has_bin("firejail"):
        deps.add("dynamic-sandbox")
    return deps


def build_orchestrator(deps: set[str] | None = None):
    deps = deps if deps is not None else detect_available_deps()
    log.info("available deps: %s", sorted(deps) or "(none beyond stdlib)")
    caps = default_capabilities(deps)
    fusion = FusionEngine()
    # Optional provisioned stacker checkpoints (fusion_train --synthetic/--dataset).
    # Default OFF so un-provisioned deployments keep the explicit-weight path and
    # behaviour is byte-identical; set VISHWAS_FUSION_USE_TRAINED=1 to enable.
    if os.environ.get("VISHWAS_FUSION_USE_TRAINED", "").strip().lower() \
            in ("1", "true", "yes"):
        _fdir = Path(os.environ.get("VISHWAS_FUSION_DIR")
                     or Path(__file__).resolve().parent.parent / "fusion")
        _n = fusion.load_trained(_fdir / "training")
        if _n:
            log.info("loaded %d trained fusion stack(s) from %s", _n, _fdir)
    reliability = ReliabilityGate()
    reporter = ReportBuilder()
    from vishwas.orchestrator import Orchestrator
    return Orchestrator(capabilities_by_target=caps, fusion=fusion,
                        reliability=reliability, reporter=reporter,
                        hard_budget_s=float(os.environ.get("VISHWAS_BUDGET_S", "300")),
                        available_deps=deps)


# ------------------------------------------------------------------ doctor --

# Detector -> (what it unlocks, how to provision it). Drives `vishwas-cli
# --doctor`, which explains WHY verdicts come back UNVERIFIED / MEDIUM: with no
# detectors provisioned, learned/AV tiers all report 'unavailable', so media &
# URLs honestly read UNVERIFIED and files fall back to static heuristics only.
_DEP_GUIDE: list[tuple[str, str, str]] = [
    ("clamav", "File/malware AV signatures (executables, APKs, PDFs)",
     "Install ClamAV + signature DB: `sudo apt install clamav && sudo freshclam` "
     "(or set VISHWAS_CLAMSCAN_BIN / VISHWAS_CLAMD_DB)."),
    ("yara", "YARA-X rule matching (packers, malware families)",
     "`pip install yara-x` (rules ship in assets/yara_rules; override with "
     "VISHWAS_YARA_RULES)."),
    ("vt", "VirusTotal hash reputation",
     "Set VISHWAS_VT_API_KEY=<your VirusTotal API key>."),
    ("model-weights", "Deepfake video/audio + image-forensics learned detectors",
     "Install torch (working torch.nn) and point the weight env vars at real "
     "checkpoints: VISHWAS_EFFORT_WEIGHTS, VISHWAS_FAKEMAMBA_WEIGHTS, "
     "VISHWAS_AASIST_WEIGHTS, VISHWAS_SSL_AUDIO_WEIGHTS, VISHWAS_HAVIC_WEIGHTS, "
     "VISHWAS_IMAGE_FACE_WEIGHTS. Launch via scripts/run_vishwas.sh."),
    ("media-tools", "Frame/stream extraction for video & audio",
     "Install ffmpeg (`sudo apt install ffmpeg`) or set VISHWAS_FFMPEG_BIN."),
    ("ocr", "Document text extraction (Aadhaar/certificate OCR)",
     "Install Tesseract (`sudo apt install tesseract-ocr`)."),
    ("pades", "PDF digital-signature (PAdES) verification for gov documents",
     "`pip install asn1crypto cryptography`."),
    ("llm", "Optional plain-language narration layer (advisory only)",
     "Set OPENAI_API_KEY or VISHWAS_LLM_BASE_URL (never a decision-maker)."),
    ("dynamic-sandbox", "Dynamic behavioral escalation for confirmed-suspicious files",
     "Install firejail or set VISHWAS_CAPE_CMD."),
]


def run_doctor() -> int:
    """Print a provisioning report: what each detector unlocks, whether it is
    available on THIS machine, and the exact command to enable a missing one."""
    deps = detect_available_deps()
    have = "✅"
    miss = "❌"
    print("Vishwas detector inventory")
    print("=" * 60)
    print(f"Detected: {', '.join(sorted(deps)) or '(none beyond the Python stdlib)'}\n")
    missing_any = False
    for dep, unlocks, fix in _DEP_GUIDE:
        ok = dep in deps
        missing_any = missing_any or not ok
        print(f"{have if ok else miss} {dep:<16} {unlocks}")
        if not ok:
            print(f"    fix: {fix}")
    print("=" * 60)
    # Per-model learned-weight status — the biggest lever for image/video/audio
    # accuracy. Each row: is the weight env var set AND does the file exist?
    print("\nLearned detector weights (image / video / audio):")
    _WEIGHT_ENVS = [
        ("VISHWAS_IMAGE_FACE_WEIGHTS", "image  · SPAI spectral AI-image detector"),
        ("VISHWAS_EFFORT_WEIGHTS", "video  · EFFORT face-forensics (strongest video model)"),
        ("VISHWAS_HAVIC_WEIGHTS", "video  · HAVIC cross-modal A/V"),
        ("VISHWAS_FAKEMAMBA_WEIGHTS", "audio  · FakeMamba (RawBMamba)"),
        ("VISHWAS_AASIST_WEIGHTS", "audio  · AASIST3 anti-spoof"),
        ("VISHWAS_XLSRMAMBA_WEIGHTS", "audio  · XLSR-Mamba"),
        ("VISHWAS_SSL_AUDIO_WEIGHTS", "audio  · SSL wav2vec audio detector"),
    ]
    torch_ok = False
    try:
        import importlib.util as _ilu
        if _ilu.find_spec("torch") is not None:
            import torch.nn  # noqa: F401
            torch_ok = True
    except Exception:
        torch_ok = False
    for env_name, label in _WEIGHT_ENVS:
        val = os.environ.get(env_name)
        if val and Path(val).exists():
            print(f"  ✅ {label}")
        elif val:
            print(f"  ⚠️  {label}  ({env_name} set but file not found: {val})")
        else:
            print(f"  ❌ {label}  (set {env_name}=/path/to/checkpoint)")
    print(f"\n  torch (required to run any learned detector): {'✅ working' if torch_ok else '❌ missing/broken — learned detectors cannot run'}")
    print("=" * 60)
    if missing_any:
        print("Missing detectors report 'unavailable' at runtime. With NONE of the\n"
              "above provisioned, media/URL checks honestly return UNVERIFIED and\n"
              "files fall back to static heuristics only (often MEDIUM). Install the\n"
              "detectors above to get real LOW / MEDIUM / HIGH verdicts.\n"
              "Image/video accuracy in particular is bounded by the learned weights\n"
              "above — see scripts/fetch_model_weights.sh for what is fetchable.")
    else:
        print("All detector families provisioned. Verdicts run at full coverage.")
    return 0


# --------------------------------------------------------------------- CLI --

def main_cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="vishwas-cli", description=__doc__)
    ap.add_argument("--text", help="message text (may contain a URL)")
    ap.add_argument("--file", help="path to a media/document file to analyze")
    ap.add_argument("--media-type", choices=["image", "video", "audio", "document"], default=None)
    ap.add_argument("--url", action="store_true", help="treat --text strictly as a URL")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--greet", action="store_true", help="only exercise the greeting path")
    ap.add_argument("--doctor", action="store_true",
                    help="report which detectors are provisioned and how to enable the rest")
    ap.add_argument("--deps", default="", help="comma-separated forced deps (override auto-detect)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.environ.get("VISHWAS_LOG_LEVEL", "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.doctor:
        return run_doctor()

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

# Process-uptime anchor for /health (monotonic clock, survives NTP jumps).
# Captured at module import; in practice that is process start for the
# webhook entrypoint.
_PROCESS_START_MONO = time.monotonic()


def health_snapshot(processor, deps) -> dict:
    """Build the rich GET /health payload (Task 2.3).

    Schema:
      status           "ok" while the process can serve at all
      uptime_s         int seconds since process start (>= 0)
      jobs_total       jobs started through MessageProcessor.process()
      jobs_ok          jobs whose orchestrator run returned an outcome
      jobs_failed      jobs whose orchestrator run raised
      quarantines_open job dirs still present under the quarantine root
      deps             {"available": [...], "count": N} — detect_available_deps()
      deps_available   flat sorted list (backward-compat pre-2.3 field)

    NOTE: jobs_* counters are in-memory and RESET TO ZERO on restart;
    durable per-job evidence lives in outcomes.jsonl.
    """
    from vishwas.quarantine import count_open_quarantines  # lazy: env-gated paths
    snap = processor.counters.snapshot() if processor is not None \
        else {"jobs_total": 0, "jobs_ok": 0, "jobs_failed": 0}
    dep_list = sorted(deps) if deps else []
    from vishwas.device import resolve_device  # lazy: torch import cost
    return {
        "status": "ok",
        "uptime_s": max(0, int(time.monotonic() - _PROCESS_START_MONO)),
        **snap,
        "quarantines_open": count_open_quarantines(),
        "device": resolve_device(),
        "deps": {"available": dep_list, "count": len(dep_list)},
        # backward compat: pre-2.3 /health returned this exact flat list
        "deps_available": dep_list,
    }


class WebhookHandler(BaseHTTPRequestHandler):
    processor: "MessageProcessor"   # injected by factory

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/health"):
            body = json.dumps(health_snapshot(getattr(self, "processor", None),
                                              getattr(self, "_deps", []))).encode()
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
                # Media lives in the SENDER's chat row — data.to is OUR number
                # (the message was sent TO us), which never matches the archive
                # key. Prefer data.from; fall back to the session jid.
                chat_id = payload.get("data", {}).get("from") or jid
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
    ap.add_argument("--port", type=int, default=int(os.environ.get("VISHWAS_WEBHOOK_PORT", "8899")))
    ap.add_argument("--host", default=os.environ.get("VISHWAS_WEBHOOK_HOST", "0.0.0.0"))
    ap.add_argument("--openwa-url", default=None)
    ap.add_argument("--budget-s", type=float, default=300.0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=os.environ.get("VISHWAS_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    deps = detect_available_deps()
    openwa = OpenWAClient(base_url=args.openwa_url) if args.openwa_url else None
    orch = build_orchestrator(deps)
    # P10 fix: do not force a root-only workdir here. Omitting `workdir` lets
    # MessageProcessor use its portable /tmp default (or $VISHWAS_WORKDIR if
    # the operator set one), so the webhook runs under any user account.
    proc = MessageProcessor(orch, openwa=openwa)
    Handler = type("BoundWebhookHandler", (WebhookHandler,), {
        "processor": proc, "_deps": deps,
    })
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    log.info("vishwas webhook listening on %s:%s (deps=%s)", args.host, args.port, sorted(deps) or "[]")
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
