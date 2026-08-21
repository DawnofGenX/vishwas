# VeriSafe — Architecture

**WhatsApp-based deepfake / government-document / malicious-content verification
platform.** Purpose: give users (especially older, non-technical people) one simple
channel to check whether something they were sent is fake before they trust it, pay
from it, submit it, or run it. This document describes the *built* system as it exists
on disk (`src/verisafe/`, 23 modules) — every claim here is grounded in that code, not
in aspiration.

---

## 1. Design principles (non-negotiable)

1. **Deterministic routing.** Input type (MIME + magic bytes) selects the capability
   pipeline. The LLM is never allowed to route, gate, or issue a verdict. All control
   flow is pure functions over typed inputs. This makes behaviour auditable and
   reproducible — a hard requirement for a safety product.
2. **Zero retention.** Every job runs inside its own isolated quarantine directory.
   Original + *all* derived artifacts (frames, audio, OCR text, sandbox dirs, dumps)
   are deleted on success, on exception, on timeout, and by a stale scanner. A JSONL
   audit line is written *outside* the tree. We prove this, we don't assert it: every
   purge records `residual_paths` and an e2e smoke shows `[]`.
3. **No fabricated evidence.** If a sub-detector can't actually measure something
   (missing weights, near-silence audio, unparseable media), it returns a **degraded
   abstention**, never a made-up score. The fusion layer treats missing signals as
   *less information*, not as *benign*.
4. **Independent detectors, fused deterministically.** Multiple independent models
   per modality are combined by logistic-regression stacking on out-of-fold (OOF)
   predictions plus calibration and selective prediction — *never* by averaging raw
   scores. Qwen (the configured local model) only *interprets structured evidence*
   into plain language; it cannot change the verdict.
5. **Graceful degradation behind availability gates.** Heavy dependencies (weights,
   YARA, ClamAV, Playwright, CAPE, tesseract/docling) are probed at runtime. If absent,
   the affected stage reports `unavailable` and the pipeline degrades instead of
   crashing; the reply always says what could *not* be tested.
6. **Security-by-default on every byte.** Files, fetched pages, OCR text, code, and
   messages are all treated as adversarial input. SSRF-safe fetch with IP pinning and a
   redirect allow-list; prompt-injection guard around any LLM call; sandboxes used only
   when a capability requires dynamic analysis.

---

## 2. Component map (actual modules)

```
                 ┌─────────────── inbound ───────────────┐
   WhatsApp ───► │  channels.py  (OpenWAClient / CLI sim) │
                 │  webhook envelope · media GET · HMAC   │
                 └───────────────┬───────────────────────┘
                                 ▼  msg dict {text, media_path, media_type}
                     ┌──────────────────────────┐
                     │      router.py           │  pure classify() → RouteDecision
                     │  input-type + gov-hints  │  (URL / video / audio / image /
                     └────────────┬─────────────┘   doc / executable / package)
                                  ▼  target name (url_phishing | deepfake_* | …)
              ┌──────────────────────────────────────────────────────┐
              │                  orchestrator.py                      │
              │  _materialize → file_validator.validate (magic bytes)│
              │  → run capability (T0 cheap → T1 mid → T2 heavy)     │
              │  → Collect CheckResults                               │
              │  → fusion.FusionEngine.score (per-target stack)       │
              │  → ReliabilityGate.evaluate → (ok, notes)             │
              │  → report.py plain-language reply                    │
              │  → quarantine.purge (always)                         │
              └──────────────────────────────────────────────────────┘
```

### Core plumbing
| Module | Responsibility | Key guarantees |
|---|---|---|
| `events.py` | Typed dataclasses: `InputType`, `MediaKind`, `CheckResult`, `Artifact`, `JobOutcome` | slots-only; enum-driven; single source of truth |
| `file_validator.py` | Extension hint **+ magic-byte / MIME** confirmation; mismatch flag | Never trusts the filename alone |
| `url_guard.py` | URL normalisation, tracking-param strip, **SSRF guard** (IP-category blocklist incl. CGNAT + IPv4-mapped), redirect allow-list, typosquat + host-string phishing heuristics | Blocks loopback/private/link-local/CGNAT pre-connect |
| `quarantine.py` | Per-job dir + manifest + purge-on-every-exit + stale scan + out-of-tree audit | Proven zero retention |
| `llm_guard.py` | Prompt-injection firewall; wraps every LLM prompt; LLM reads structured evidence only | LLM can interpret, never verdict |
| `i18n.py` | en primary, best-effort hi (+ extensible); every user-facing string templated | Non-technical, multi-language UX |
| `media_utils.py` | ffprobe probe, frame extraction, robustness transform matrix (codec/bitrate/resize/crop/fps/screen-record) for deepfake tests | CPU-friendly; deterministic transforms |
| `fusion.py` | Per-target LR stacking on OOF features, Platt/temp scaling calibration, PR-AUC/ECE, **selective prediction**, `ReliabilityGate`, `load_trained()` | No raw-score averaging; honest confidence |
| `fusion_train.py` | OOF CV, feature layout shared with serving (`feature_vector`), artifact save/load; synthetic labelled demo sets | train–serving consistency (identical feature order/value+gap layout) |
| `router.py` | Pure classification into a capability target | No I/O, no LLM |
| `orchestrator.py` | Glue above: materialise → validate → run tiers → fuse → gate → report → purge | Single entry point `handle_incoming(msg)` |
| `report.py` | Two-sentence plain-language reply: result + confidence + what-to-do (trust/open/use/avoid) | Human-readable; never exposes internals |
| `channels.py` | OpenWA transport (real v0.21 API) + in-process CLI simulator | Same code path for both transports |
| `app.py` | CLI ad-hoc mode + threaded webhook server (`python -m verisafe.app`) | Entry points |

### Capabilities (progressive T0→T1→T2, each a `CapabilityContract`)
| Capability | Target | What it actually does today | Gated stages |
|---|---|---|---|
| `url_phishing.py` | `url_phishing` | Normalise → SSRF → (VT reputation) → host-string typosquat/phish scorer (always computable) → DOM kit (forms/password/external-post/brand-match) → PhishLLM if available → download files re-enter file pipeline | VT, browser, PhishLLM, LLM |
| `malware_file.py` | `malicious_file` | APK: VT → MobSF → JADX/APKTool → Quark-Engine → YARA-X; PE/ELF: static imports/heuristics → YARA-X → CAPE (dynamic) when present | clamav, yara, mobsf/jadx, dynamic-sandbox |
| `gov_document.py` | `gov_document` | Docling/tesseract OCR → doc-type + authority detection → DigiLocker / digital-sig / QR / API Setu / official-API verification → Playwright-on-official-site fallback → versioned RAG cache (retrieval only, **not** source of truth) | ocr, docling, browser, gov APIs |
| `deepfake_video.py` | `deepfake_video` | ffprobe → frames → Effort (spatial) + VB+StA (temporal) + DeMamba (general/degraded) heavy detactors **when weights present**; offline heuristic baseline otherwise; robustness-transform battery | model-weights, cv2, ffmpeg |
| `deepfake_audio.py` | `deepfake_audio` | Probe → offline spectral/MFCC baseline with **near-silence abstention gate** → Fake-Mamba / AASIST / SSL detector (multi-crop) when weights present | model-weights, ffmpeg |
| `cross_modal.py` | companion to video-with-audio | HAVIC-class AV-consistency forensics (lip/audio timing, energy correlation) | weights |
| `image_facecheck.py` | `image_facecheck` | Still-frame face-manipulation check (frame path of the video detector) | weights, cv2 |

Every heavy stage self-reports `missing_dependency` when its dependency isn't installed,
so a fresh box runs the whole pipeline in *degraded-but-honest* mode (proven in P10).

---

## 3. Transport: OpenWA (verified, not assumed)

VeriSafe speaks **only** OpenWA for WhatsApp I/O. Verified identity and surface
(see `docs/research/VERIFY_SECURITY_STACK.md`):

* Repo `rmyndharis/OpenWA`, MIT, NestJS, **Node 22**, default port **2785**, SQLite.
* Session-scoped REST routes, e.g.:
  * `POST /api/sessions/:id/messages/send-text`
  * `GET  /api/sessions/:id/messages/:chat/:msgId/media`  (inbound media bytes)
  * list/chats/sessions/status endpoints.
* Webhook: OpenWA posts a signed event envelope on `message.received`; signature header
  `X-OpenWA-Signature` (HMAC). `channels.OpenWAClient` implements send, media fetch, and
  webhook parsing; `app.py` verifies the HMAC and dedupes by idempotency key.

The in-process **CLI simulator** (`CliChannel`) exercises the *identical* `MessageProcessor`
path, so development/QA needs no live WhatsApp number while production uses OpenWA
verbatim. Deployment (compose + webhook wiring) is documented in `docs/DEPLOYMENT.md`
(P9) and intentionally defers the exact image tag to the upstream repo to avoid drift.

---

## 4. Fusion & reliability semantics

* **Per-target stack.** One LR stacker per capability target (`url_phishing`,
  `deepfake_video`, `deepfake_audio`, …). Training features use the exact
  value+gap-flag pair layout emitted by `FusionEngine.feature_vector()` at inference,
  so trained weights are directly loadable (P5-proven for `url_phishing` and
  `deepfake_video`).
* **Calibration.** Temperature/Platt scaling gives a calibrated probability
  (`calibrated`) alongside the raw score. ECE reported in training.
* **Selective prediction.** When detector disagreement is high or too few usable
  checks exist, the engine **abstains** (`unable_to_verify`) rather than guessing — the
  right answer for a safety product where a false "safe" is worse than "I couldn't tell."
* **ReliabilityGate.** Returns `(ok: bool, notes: [])`. A failed gate downgrades the
  verdict and appends plain-language "could not test X" notes to the reply.

### 4.1 Non-blocking heavy stages (`pending_heavy` follow-ups)

* T2 learned stages (`stage_cost = "heavy"`: AASIST audio, EFFORT video, HAVIC
  cross-modal) get a per-stage budget (`VERISAFE_HEAVY_STAGE_BUDGET_S`, default 30s).
* Over budget, the orchestrator stops *waiting*, not the work: the stage keeps running in a
  dedicated **sequential** background pool (`max_workers=1` — heavy inference is never
  parallelized on the CPU-only laptop), and the fast verdict ships immediately with
  `pending_heavy: [{cap, expected_s}]` evidence plus a plain-language "update coming" notice.
* On completion the follow-up is composed from the `heavy_followup` i18n template ONLY
  (deterministic wording + fused confidence number — never LLM-generated) and delivered via
  `MessageProcessor.deliver`; the CLI simulator prints it inline.
* If the user session ended first (`MessageProcessor.end_session`), the follow-up is dropped
  silently with a stdout log — no retry queue. Any exception in the follow-up path becomes a
  logged CheckResult-style failure record; it never raises out of `process()`.

---

## 5. Threat model summary (full table in P9)

| Threat | Mitigation in code |
|---|---|
| SSRF / DNS-rebinding via inbound URLs | IP-category blocklist pre-connect, redirect allow-list, size+time caps |
| Prompt injection via page/doc/message text reaching an LLM | `llm_guard` firewall; LLM sees structured evidence only, cannot set verdict |
| Untrusted binary/media execution | static-first; dynamic only in CAPE/FireJail sandbox when enabled |
| Filename spoofing (`.txt` really an `.exe`) | magic-byte/MIME confirm, mismatch flagged to malware pipeline |
| Retention / privacy leak | per-job quarantine + purge-all-on-exit + out-of-tree audit |
| False "safe" on weak signal | selective prediction abstention; degraded abstention, never fabricated score |
| Model supply-chain risk | heavy models gated behind explicit weight paths; core is stdlib-only |

---

## 6. Multi-agent development system (how this was built)

Built under a coordinated multi-agent process, orchestrated by a lead agent:
**Research, Architecture, Backend, AI/ML, Deepfake, Doc-Verification, Malware/Security,
URL/Phishing, Frontend/UI, WhatsApp/OpenWA, QA, Security/Red-Team, Performance,
Documentation.** Rules enforced throughout:

* Agents emit **structured deliverables with evidence** (source URLs, arXiv IDs, repo
  URLs), never bare claims.
* The orchestrator **does not blindly accept output** — it re-verifies against primary
  sources before integration (this caught wrong arXiv IDs and a fictional API shape).
* Cross-review + red-team challenge before a change lands; iterative refine.

Because the delegation infra hit rate-cap failures during P1, the orchestrator performed
that verification **directly** against arXiv/GitHub/PyPI primary sources and wrote the
resulting evidence notes itself (`docs/research/*.md`) rather than trusting or re-trying
the failed fan-out.
