# Vishwas — Workflow: Decision Tree & Failure Matrix

Companion to `docs/ARCHITECTURE.md`. This page is the *operational* reference: given a
message on WhatsApp, exactly which code runs in what order, and what happens when each
stage degrades or fails. Everything below maps to real functions in `src/vishwas/`.

---

## 1. End-to-end flow (one job)

```
message.received (OpenWA webhook)            OR   CLI ad-hoc message
        │  HMAC verify + idempotency dedupe              │
        ▼                                                ▼
channels.MessageProcessor.process(msg)
        │
        ├── inbound media? ── GET /sessions/:id/messages/:chat/:msgId/media
        │       (≤1 MiB inline base64 decoded; >1 MiB fetched by marker)
        ▼
orchestrator.handle_incoming(msg_dict)
        │
        │  t0 = now
        ▼
[1] router.Router.classify()          ── pure; yields RouteDecision(input_type, target_hint)
[2] orchestrator._materialize()       ── copy artifact into per-job quarantine (JobQuarantine.track)
[3] file_validator.validate()         ── extension hint + magic-byte/MIME confirm → verified_kind
[4] router.target_for(art)            ── final capability target after validation
[5] capabilities.<target>.analyze()   ── progressive tiers: T0 cheap → T1 mid → T2 heavy
        │                                   each stage either emits a CheckResult or marks
        │                                   itself unavailable/skipped (never raises out)
        ▼
[6] fusion.FusionEngine.score(target, checks)   ── LR stack on OOF feature layout
        │                                          ── calibrated probability + disagreement
        ▼
[7] fusion.ReliabilityGate.evaluate() ── (ok, notes); low usable-checks ⇒ abstain
        ▼
[8] report.build_reply()              ── verdict + confidence + "what to do" plain language
        ▼
[9] quarantine.purge()                ── delete original + ALL derived, audit line outside tree
        ▼
channel.send_text(reply)              ── OpenWA send-text   (or stdout for CLI)
```

The whole thing is **deterministic**: the LLM (Qwen), when present, is called only inside
`report`/`llm_guard` to turn structured evidence into friendly wording. It cannot change
the verdict, route, or gate outcome.

---

## 2. Input-type decision tree (router.py `_hint`)

```
incoming msg
 ├─ text has URL (regex) . . . . . . . . . . . . . ─► target: url_phishing
 ├─ media_path present:
 │    ├─ kind ∈ {PE, ELF}                      ─► malicious_file   (executable)
 │    ├─ kind ∈ {APK, JAR, ZIP, RAR, 7Z, GZIP} ─► malicious_file   (package)
 │    ├─ input VIDEO  / kind video container   ─► deepfake_video
 │    ├─ input AUDIO  / kind audio codec       ─► deepfake_audio
 │    ├─ input IMAGE                              ─► image_facecheck
 │    ├─ kind office-doc/pdf/text/html/json/csv:
 │    │     ├─ gov-hint regex hits payload/file ─► gov_document
 │    │     └─ otherwise                        ─► document_generic
 │    └─ EMPTY / UNKNOWN / other                 ─► unclassified → unable_to_verify
 └─ no URL, no media, just text                 ─► free-text question → light advice reply
      (untrusted text; never auto-acts)
```

**Tie-breaking rule:** security-relevant targets (malware/phishing) outrank detection
targets when an item could match both — higher-harm wins. A `.mp4` that magic-bytes say is
really a PE binary is routed to `malicious_file`, not `deepfake_video`.

---

## 3. Progressive tier model (within a capability)

Each capability runs three tiers, stopping early on hard verdicts where safe:

| Tier | Name | Cost | Examples | Gate behaviour |
|---|---|---|---|---|
| T0 | cheap | ms | probes, magic bytes, hash lookups, host-string heuristics | always run |
| T1 | mid | seconds | DOM kit, offline spectral baseline, OCR + type detect, static imports | run unless dep missing |
| T2 | heavy | minutes / gated | Effort/VB+StA/DeMamba, Fake-Mamba/AASIST, MobSF/JADX, CAPE, Playwright scrape, PhishLLM | run only if dependency/weights present, else mark unavailable |

Short-circuits (perf, P8):
* URL: confirmed-evil VT reputation **and** malicious host-string score → skip browser.
* File: clean ClamAV + clean YARA-X + no exec/package signals → skip dynamic sandbox.
* Media: probe failure (unparseable) → stop, return `unable_to_verify`, do **not** invent.

---

## 4. Failure & degradation matrix

"degraded" = returns a `CheckResult(status=degraded/unavailable)` with an honest note;
the pipeline continues. "abstain" = `ReliabilityGate` refuses a verdict.

| # | Failure | Detected at | Behaviour | User sees |
|---|---|---|---|---|
| F1 | No VirusTotal key | `url_phishing._mid` / `malware_file` | stage marked `unavailable (no API key)` | "I couldn't check this link's reputation" |
| F2 | Host unreachable / DNS fail | `url_guard.fetch_url` | DOM stage skipped; host-string score still computed | lower confidence, still a verdict if string signals strong |
| F3 | Private/link-local IP (SSRF) | `url_guard.ssrf_guard` | connection blocked pre-resolve; SSRF flag set | flagged suspicious, not fetched |
| F4 | Redirect off allow-list | `url_guard` fetch loop | redirect aborted | "this page redirected somewhere unexpected" |
| F5 | Model weights absent | every T2 detector | `unavailable (missing_dependency=model-weights)`; offline baseline used instead | "advanced AI checks were skipped" |
| F6 | cv2 / tesseract / docling absent | facecheck, govd OCR | OCR fallback chain degrades; stage degraded | reduced confidence on document reading |
| F7 | Near-silence audio | `deepfake_audio._extract_mfcc_stats` | energy gate → `usable_audio=false`, prob=None | "audio too quiet to analyse", abstains |
| F8 | Unparseable media (bad MP4) | `media_utils.probe` | probe failed → zero usable signals | "couldn't read this video" → unable_to_verify |
| F9 | Too few usable checks / high disagreement | `ReliabilityGate` | selective-prediction abstention | "I wasn't sure enough to call it one way or the other" |
| F10 | Sandbox/CAPE not installed | `malware_file` dynamic step | dynamic analysis skipped | "I can only check the file's contents, not how it behaves" |
| F11 | Gov API/portal down or cred-gated | `gov_document` | fallback to official-site Playwright, then RAG cache (retrieval only) | "official verification endpoint unreachable; using cached reference" |
| F12 | Prompt-injection in content | `llm_guard` | injection stripped/quarantined; LLM prompt firewalled | (internal) injection reported as evidence, never obeyed |
| F13 | Any exception mid-job | orchestrator `try/except` | caught → purge(reason="exception") → error reply | "something went wrong; nothing was kept" |
| F14 | Job timeout | orchestrator watchdog | kill + purge(reason="timeout") | "took too long; stopped safely, nothing kept" |
| F15 | Webhook signature mismatch | `app.py` | request rejected, logged | (attacker) no processing |
| F16 | Duplicate webhook (retry) | idempotency key | second delivery dropped silently | single user-visible answer |

**Invariant across all rows:** zero retention holds. Purge runs on success, exception,
and timeout alike; the audit log records `residual_paths` so any leak is observable.

---

## 5. Verdict → user-action mapping (report.py)

| Verdict | Meaning | Plain-language action shown |
|---|---|---|
| `trust` / `open` / `use` | low-risk, sufficient usable evidence | "This looks safe to open." |
| `caution` | some weak/suspicious signal but insufficient to condemn | "Something looks off — be careful, don't enter passwords here." |
| `do_not_use` | positive malicious/deepfake indicators | "Do not use / open / pay from this." |
| `unable_to_verify` | abstained (no usable signals / gated-out / degraded) | "I couldn't tell for sure. Here's how you can double-check yourself." |

Confidence is always surfaced numerically **and** narrated (high/low) because the target
user is non-technical.
