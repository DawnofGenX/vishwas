"""Orchestrator: deterministic routing, progressive tiering, failure matrix.

Flow per message:
  route -> validate -> T0 cheap checks (parallel) -> decide conclusive?
   -> T1 mid checks (independent ones parallel under a semaphore)
   -> T2 heavy stages ONLY if still inconclusive AND budget OK
  -> fuse calibrated scores -> reliability gate -> verdict
  -> report build (i18n) -> channel send -> quarantine purge
Nothing here trusts an LLM output as a decision; models only feed signals.
Every stage wraps exceptions into CheckResult(status=failed); a dead tool
never kills the job — worst case the job ends UNABLE_TO_VERIFY with reasons.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FTimeout
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from .capabilities.base import Capability, CheckResult
from .events import Artifact, InputType, JobContext, Verdict, new_job_id
from .file_validator import FileValidator
from .fusion import FusionEngine, ReliabilityGate
from .i18n import t
from .quarantine import JobQuarantine
from .report import ReportBuilder
from .router import Router, RouteDecision

log = logging.getLogger("vishwas.orchestrator")

_HEAVY_POOL = 2                    # thermal-safe concurrency for heavy stages
_HARD_BUDGET_MULT = 2.0


def _default_heavy_stage_budget() -> float:
    return float(os.environ.get("VISHWAS_HEAVY_STAGE_BUDGET_S", "30"))


@dataclass(slots=True)
class JobOutcome:
    job_id: str
    verdict: Verdict
    confidence: float             # 0..1 after calibration & reliability scaling
    verdict_reason_keys: list[str]
    user_message: str
    language: str
    checks: list[CheckResult]
    fusion_trace: dict[str, Any]
    wall_s: float
    purged: bool

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(type(self))}
        d["verdict"] = self.verdict.value
        d["checks"] = [
            {"name": c.name, "status": c.status, "signals": c.signals, "cost": c.cost}
            for c in self.checks
        ]
        return d


# P8: capability classes that keep running after a decisive finding settles
# the user answer — they are light analysis / reporting families whose extra
# seconds of wall time are cheap relative to the heavy model runs.
_LIGHT_CAPS = frozenset({"GovDocumentCapability", "UrlPhishingCapability"})


def _has_confirmed_danger(batch: list[CheckResult]) -> bool:
    """Conservative early-stop trigger. Fires only on *confirmed* positives
    from independent check families (never on a single heuristic alone):

      - AV signature match (clamscan detected)          -> malware confirmed
      - YARA hits > 0.6 AND any of quark/packed/statics -> malware confirmed
      - vt reputation >= 0.5                           -> known-bad confirmed
      - phish score > 0.7 with corroborating dom/ssrf   -> phishing confirmed

    The short-circuit then skips remaining heavy/unknown stages. This is a
    performance gate, NOT an accuracy claim: every evidence source that ran
    still appears in the report, and the verdict is computed over all checks.
    """
    by_name = {c.name: c for c in batch}

    def sig(name: str, key: str, default=None):
        c = by_name.get(name)
        if c is None or not c.usable():
            return default
        return c.signals.get(key, default)

    if sig("clamscan", "detected") is True:
        return True
    if (sig("yara_x", "hits_norm", 0.0) or 0.0) > 0.6 and any(
            (by_name.get(n) is not None and by_name[n].usable() and
             ((by_name[n].signals.get(k) or 0) > t))
            for n, k, t in (("quark_engine", "score_norm", 0.4),
                            ("pe_statics", "packed", False))):
        return True
    if (sig("vt_reputation", "prob_malicious", 0.0) or 0.0) >= 0.5:
        return True
    p = sig("phish_heuristics", "score_norm", 0.0) or 0.0
    if p > 0.7 and (
            (sig("ssrf_guard", "degraded", False) is True) or
            (sig("url_redirects", "suspicious_hops", 0) or 0) > 0):
        return True
    return False


class Orchestrator:
    """Wires router + validator + capabilities + fusion + reporting + cleanup."""

    def __init__(self,
                 capabilities_by_target: dict[str, list[Capability]],
                 fusion: FusionEngine,
                 reliability: ReliabilityGate,
                 reporter: ReportBuilder,
                 i18n_lang_default: str = "en",
                 hard_budget_s: float = 300.0,
                 greeting_enabled: bool = True,
                 available_deps: set[str] | None = None,
                 heavy_stage_budget_s: float | None = None,
                 followup_sender: Callable[[str, str], bool] | None = None):
        self.capabilities = capabilities_by_target     # target-name -> ordered capabilities
        self.fusion = fusion
        self.reliability = reliability
        self.reporter = reporter
        self.router = Router()
        self.validator = FileValidator()
        self.i18n_default = i18n_lang_default
        self.hard_budget_s = hard_budget_s
        self.greeting_enabled = greeting_enabled
        self.available_deps = available_deps or set()  # empty => everything gated off except cheap
        self._pool = ThreadPoolExecutor(max_workers=_HEAVY_POOL + 2)
        # 2.1 non-blocking heavy stages: ONE worker — learned inference is
        # never parallelized (CPU-only laptop, thermal constraint). A stage
        # that outlives its per-stage budget keeps running here while the
        # fast verdict ships; its follow-up is composed on completion.
        self.heavy_stage_budget_s = (heavy_stage_budget_s
                                     if heavy_stage_budget_s is not None
                                     else _default_heavy_stage_budget())
        self.default_followup_sender = followup_sender
        self._heavy_pool = ThreadPoolExecutor(max_workers=1,
                                              thread_name_prefix="vishwas-heavy-seq")
        self._pending_lock = threading.Lock()
        self._pending_futs: list[Future] = []

    # ------------------------------------------------------------ public --
    def handle_incoming(self, msg: dict,
                        followup_sender: Callable[[str, str], bool] | None = None) -> JobOutcome:
        """msg: {id, text?, media_path?, media_type?, sender_lang?}.

        followup_sender(chat_id, text) is called (from the background pool)
        when a heavy stage that outlived its budget finally finishes; it may
        be None, in which case a composed follow-up is only logged.
        """
        job_id = new_job_id()
        qroot = Path(msg.pop("_qroot_override", None) or _default_qroot())
        qroot.mkdir(parents=True, exist_ok=True)
        with JobQuarantine(job_id, root=qroot) as q:
            workdir = q.make_subdir("work")
            decision = self.router.classify(msg)
            lang = msg.get("sender_lang") or (decision.text and _quick_lang(decision.text)) or self.i18n_default
            artifact = self._materialize(q, workdir, decision, msg, job_id)
            ctx = JobContext(job_id=job_id, artifact=artifact, quarantine_root=q.job_dir,
                             deadline_mono=time.monotonic() + self.hard_budget_s,
                             wall_budget_s=self.hard_budget_s,
                             model_weights_available="model-weights" in self.available_deps,
                             dynamic_sandbox_available="dynamic-sandbox" in self.available_deps,
                             browser_available="browser" in self.available_deps,
                             vt_api_key=(os.environ.get("VISHWAS_VT_API_KEY") if "vt" in self.available_deps else None),
                             llm_available="llm" in self.available_deps,
                             pades_available="pades" in self.available_deps,
                             rag_cache_available="rag-cache" in self.available_deps,
                             chat_id=str(msg.get("chat_id") or msg.get("session_key")
                                         or msg.get("id") or "") or None)
            outcome = self._run(artifact, ctx, decision, lang, q,
                                followup_sender=followup_sender)
        outcome.purged = True
        return outcome

    # ---------------------------------------------------------- internals --
    def _materialize(self, q: JobQuarantine, workdir: Path, decision: RouteDecision,
                     msg: dict, job_id: str) -> Artifact:
        from .file_validator import make_artifact
        kind_input = decision.input_type
        media_path = msg.get("media_path")
        if media_path:
            mp = Path(media_path)
            if not mp.exists():
                raise FileNotFoundError(f"media path missing: {mp}")
            dest = q.track(workdir / mp.name)[0]
            dest.write_bytes(mp.read_bytes())
            art = Artifact(path=dest, original_filename=mp.name, declared_type=kind_input)
        elif decision.text and decision.is_url:
            art = make_artifact(workdir, "url.txt", InputType.URL, data=(decision.url or "").encode())
        else:
            art = make_artifact(workdir, "message.txt", kind_input, data=(decision.text or "").encode())
        return art

    def _run(self, art: Artifact, ctx: JobContext, decision: RouteDecision, lang: str,
             q: JobQuarantine, followup_sender: Callable[[str, str], bool] | None = None) -> JobOutcome:
        t0 = time.monotonic()
        target = self.router.target_for(decision, art)
        caps = self.capabilities.get(target, [])
        # validate first — magic bytes are part of the evidence
        kind, mismatch = self.validator.validate(art, ctx)
        ctx.extra["verified_kind"] = kind.value
        ctx.extra["ext_mismatch"] = mismatch
        if decision.urls_in_text:
            ctx.extra["urls_in_text"] = decision.urls_in_text
        if decision.url:
            ctx.extra.setdefault("primary_url", decision.url)

        results: list[CheckResult] = []
        if mismatch:
            results.append(CheckResult(name="ext_mismatch_flag", cost="cheap", status="ok",
                                       signals={"declared": art.declared_type.value, "verified": kind.value},
                                       notes="filename extension does not match actual content type"))
        else:
            # No mismatch: emit the explicit negative so fusion's const_true signal
            # records a known_gap (N/A) instead of absent (missing evidence).
            results.append(CheckResult(name="ext_mismatch_flag", cost="cheap", status="ok",
                                       signals={}, notes="filename extension matches content type"))

        stage_timings: dict[str, float] = {}
        short_circuited: str | None = None
        pending_heavy: list[dict[str, Any]] = []
        for cap in caps:
            cap_name = cap.__class__.__name__
            needed = [d for d in getattr(cap, "requires", ()) if d not in self.available_deps]
            if needed:
                results.append(CheckResult(name=f"{cap_name}.gated", cost="cheap",
                                           status="unavailable", signals={"missing_dependencies": needed},
                                           notes="capability dependencies not provisioned; recorded as evidence gap"))
                continue
            if ctx.expired():
                results.append(CheckResult(name=f"{cap_name}.timeout", cost="cheap",
                                           status="failed", signals={}, notes="wall budget exhausted before this stage"))
                continue
            # P8 budget floor: a stage started with <10s left cannot yield a
            # finished check in time — record it as a gap, do not burn CPU.
            if ctx.remaining_s() < 10.0:
                results.append(CheckResult(name=f"{cap_name}.timeout", cost="cheap",
                                           status="skipped", signals={},
                                           notes="less than 10s of budget left; stage not started"))
                continue
            # P8 conservative short-circuit: once a *confirmed* positive finding
            # has settled the user-facing answer (do_not_use), only the light
            # analysis families keep running — heavy/unknown stages are skipped
            # because they buy CPU heat, not additional safety.
            if short_circuited is not None and cap_name not in _LIGHT_CAPS:
                results.append(CheckResult(name=f"{cap_name}.skip_early_stop", cost="cheap",
                                           status="skipped", signals={},
                                           notes=f"earlier stage '{short_circuited}' produced decisive evidence; heavy/unknown stages skipped"))
                continue
            last_batch: list[CheckResult] = []
            t_stage = time.monotonic()
            pending_fut: Future | None = None
            try:
                if getattr(cap, "stage_cost", "") == "heavy":
                    # 2.1: heavy learned stages run under a per-stage budget.
                    # Over budget -> stop WAITING (never stop the work): the
                    # stage keeps running in the sequential background pool
                    # and the fast verdict ships with pending_heavy evidence.
                    pending_fut = self._heavy_pool.submit(cap.analyze, art, ctx)
                    last_batch = pending_fut.result(timeout=self.heavy_stage_budget_s)
                else:
                    last_batch = cap.analyze(art, ctx)
                results.extend(last_batch)
            except FTimeout:
                stage_timings[cap_name] = round(time.monotonic() - t_stage, 2)
                pending_heavy.append({"cap": cap_name,
                                      "expected_s": max(1, int(round(self.heavy_stage_budget_s)))})
                self._arm_heavy_followup(cap_name, pending_fut, ctx, lang,
                                         target, followup_sender)
                continue
            except Exception as e:  # noqa: BLE001 — isolation by contract
                log.exception("capability %s crashed", cap)
                results.append(CheckResult(name=f"{cap_name}.crash", cost="mid",
                                           status="failed",
                                           signals={"exception": f"{e.__class__.__name__}:{str(e)[:80]}"},
                                           notes="stage crashed and was isolated; other stages continue"))
            stage_timings[cap_name] = round(time.monotonic() - t_stage, 2)
            if short_circuited is None and _has_confirmed_danger(last_batch):
                short_circuited = cap_name
        # dedupe: keep highest-quality result per check name
        dedup: dict[str, CheckResult] = {}
        rank = {"ok": 4, "degraded": 3, "failed": 2, "unavailable": 1, "skipped": 0}
        for r in sorted(results, key=lambda x: rank.get(x.status, 0)):
            dedup[r.name] = r
        results = sorted(dedup.values(), key=lambda r: -rank.get(r.status, 0))

        fused = self.fusion.decide(target, results)
        reliable_ok, gate_notes = self.reliability.evaluate(fused, results, ctx)
        if not reliable_ok:
            fused.score = 0.0
            fused.verdict = Verdict.UNABLE_TO_VERIFY
            fused.reasons.append("reliability_gate:" + ";".join(gate_notes[:4]))
            fused.confidence = min(fused.confidence, 0.25)

        report = self.reporter.build(target=decision.target_hint or target,
                                     verdict=fused.verdict,
                                     confidence=fused.confidence,
                                     reasons=fused.reasons,
                                     checks=results,
                                     lang=lang,
                                     artifact_name=art.original_filename)
        if pending_heavy:
            # plain-language promise that a deeper result will arrive later
            report.text = report.text + "\n\n" + t("heavy_pending_notice", lang)
        outcome = JobOutcome(job_id=ctx.job_id,
                            verdict=fused.verdict,
                            confidence=round(fused.confidence, 3),
                            verdict_reason_keys=fused.reasons[:8],
                            user_message=report.text,
                            language=lang,
                            checks=results,
                            fusion_trace={
                                "target": target,
                                "raw_score": round(fused.raw, 4),
                                "calibrated": round(fused.score, 4),
                                "disagreement": fused.disagreement,
                                "gate": {"ok": reliable_ok, "notes": gate_notes},
                                "usable_checks": sum(1 for r in results if r.usable()),
                                "unavailable": [r.name for r in results if r.status == "unavailable"],
                                # P8 ops visibility: per-stage wall time + early-stop marker
                                "stage_timings_s": stage_timings,
                                "short_circuited_at": short_circuited,
                                # 2.1: heavy stages still running in the background
                                "pending_heavy": pending_heavy,
                            },
                            wall_s=round(time.monotonic() - t0, 2),
                            purged=False)
        ctx.note(f"verdict={fused.verdict.value} conf={outcome.confidence} reasons={len(fused.reasons)}")
        return outcome

    # ------------------------------------------- background heavy follow-up --
    def _arm_heavy_followup(self, cap_name: str, fut: Future | None, ctx: JobContext,
                            lang: str, target: str,
                            sender: Callable[[str, str], bool] | None) -> None:
        """Compose + deliver a DETERMINISTIC template follow-up when a heavy
        stage that outlived its budget finally finishes. Template-only: i18n
        wording + fused confidence number; never an LLM. Any exception in this
        path is logged with a CheckResult-style failure record — it must never
        raise out of process()."""
        if fut is None:
            return
        with self._pending_lock:
            self._pending_futs.append(fut)

        def _done(f: Future) -> None:
            try:
                batch = f.result()
                fused = self.fusion.decide(target, batch)
                text = t("heavy_followup", lang,
                         cap=cap_name,
                         verdict=t(ReportBuilder.VERDICT_KEY[fused.verdict], lang),
                         conf=f"{int(round(fused.confidence * 100))}%")
                send = sender or self.default_followup_sender
                if ctx.chat_id and send is not None:
                    try:
                        send(ctx.chat_id, text)
                    except Exception:  # noqa: BLE001 — delivery is best-effort
                        log.exception("follow-up delivery failed (job %s)", ctx.job_id)
                else:
                    log.info("job %s: heavy follow-up composed but no transport/chat_id; dropped",
                             ctx.job_id)
            except Exception as e:  # noqa: BLE001 — never raise from a callback
                log.exception("late heavy stage %s follow-up path failed", cap_name)
                rec = CheckResult(name=f"{cap_name}.followup_failed", cost="mid",
                                  status="failed",
                                  signals={"exception": f"{e.__class__.__name__}:{str(e)[:80]}"},
                                  notes="background stage follow-up failed; recorded, never raised")
                print(f"[vishwas] followup failure record job={ctx.job_id} "
                      f"check={rec.name} status={rec.status} signals={rec.signals}",
                      flush=True)
            finally:
                with self._pending_lock:
                    try:
                        self._pending_futs.remove(f)
                    except ValueError:
                        pass

        fut.add_done_callback(_done)

    def wait_for_pending_followups(self, timeout_s: float | None = None) -> bool:
        """Block until every armed background heavy stage has settled.

        Dev/CLI/test helper — the production message loop never waits.
        Returns True when nothing is pending, False on timeout."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            with self._pending_lock:
                pending = list(self._pending_futs)
            if not pending:
                return True
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)


def _quick_lang(text: str) -> str:
    from .i18n import detect_language
    return detect_language(text or "")


def _default_qroot() -> Path:
    from .quarantine import QUARANTINE_ROOT
    return QUARANTINE_ROOT


GREETING_TRIGGER = ("help?", "hello", "hi ", "hii", "hey", "start", "कैसे", "मदद")


def maybe_greet(msg: dict, session_state: dict) -> str | None:
    """Return localized 'Hi, how can I help you?' when appropriate, else None.

    Appropriate on: first substantive contact in session, OR the message is a
    short ambiguous opener. Suppressed after any analysis reply in-session.
    """
    sid = msg.get("session_key") or msg.get("id") or "anon"
    st = session_state.setdefault(sid, {"greeted": False, "last_replied_ts": 0})
    idle = time.monotonic() - st.get("last_activity_mono", time.monotonic())
    text = (msg.get("text") or "").strip().lower()
    is_opener = (not text) or len(text) <= 12 or any(g in text for g in GREETING_TRIGGER)
    has_media = bool(msg.get("media_path")) or bool(msg.get("url"))
    now_active = time.monotonic()
    if not has_media and not msg.get("url") and is_opener and not st["greeted"]:
        st["greeted"] = True
        from .i18n import t
        lang = msg.get("sender_lang") or _quick_lang(text) or "en"
        return t("greeting", lang)
    if has_media or (text and len(text) > 12):
        st["greeted"] = True
    st["last_activity_mono"] = now_active
    return None
