# Threat Model

Adversary-facing design rationale for VeriSafe. Every control below exists because of a concrete attack class; the P7 red-team battery (`tests/test_09_redteam.py`) is the executable form of this document.

## Assets & threat actors

| Asset | Actors |
|---|---|
| Correctness of each verdict (safety-critical: "do_not_use" must not be a false negative that users rely on) | Fraudsters (phishers, malware distributors, deepfake operators), nation-state-scale content farms, drive-by uploaders testing the system itself |
| Privacy of user uploads (zero-retention contract) | Any party with post-compromise access to the host or its backups |
| The service's integrity (it is a *detector* — compromising it lets an adversary silently flip verdicts) | Anyone able to reach the webhook port, supply poisoned model weights / RAG cache, or craft inputs that steer the LLM narration layer |
| Host hardware (laptop, thermal-trip history) | Sustained-load DoS via huge/malformed media, fork-bomb-style subprocess patterns |

## Attack surfaces → controls

### A1. Inbound channel forgery
- **Webhook**: every delivery HMAC-verified over raw bytes (`X-OpenWA-Signature`), retried deliveries deduped by `X-OpenWA-Idempotency-Key`. Forged/unsigned deliveries rejected 401 (or accepted only when the operator has explicitly run unsigned mode, which is documented as LAN-only).
- **REST to OpenWA**: Bearer-keyed, pinned base URL, bounded timeouts so a hostile/hung gateway can't tie up worker threads.

### A2. Untrusted input (everything uploaded is adversarial)
- **Filename lies** (double extensions `x.pdf.exe`, case tricks, trailing dots/spaces/tabs): magic bytes are ground truth; mismatches raise `ext_mismatch` which feeds the fusion. P7 proved PE/SOURCE_CODE/PDF/ZIP/MP4 all route correctly regardless of declared name.
- **Shebang-as-prose**: a `#!/bin/sh` payload named `.txt` is now classified as `SOURCE_CODE`, never plain text (previously a real gap found in P7).
- **Control-character smuggling in URLs** (CRLF/LF/NUL for header injection at fetch time): stripped in `normalize_url()` before any network use; P7 asserts no char < 0x20 or 0x7F survives.
- **SSRF via IP literals** (`http://169.254.169.254/latest/meta-data/…` — cloud metadata, internal hosts): previously only *resolved* names were checked; P7 added direct classification of literal-IP hosts before any DNS call. All non-public categories blocked pre-fetch.
- **Homograph / lookalike TLDs**: normalized to punycode *for reporting* but never collapsed onto a brand string — a Cyrillic-а apple stays distinct from apple.com.
- **Redirect chains**: bounded hop count, per-hop timeout, suspicious hops recorded as signals rather than trusted destinations.
- **Oversized/malformed media**: fixed sniff window (256KB) for magic-byte work, per-subprocess timeouts, quarantine-bound output paths; a hung codec dies at its timeout and the job continues.

### A3. Prompt injection against the LLM layer
The LLM narrates evidence; it does not decide. Defense-in-depth in `llm_guard.py`:
- User-sourced text enters prompts ONLY inside an `UNTRUSTED_CONTENT_START/END` fenced block marked untrusted; the system role is template-built and never contains user text.
- Detector families (P7 expanded this): classic "ignore previous instructions", role-jailbreak ("you are now DAN"), developer/system/admin impersonation directives, skip-checks commands, mark-everything-safe commands, ChatML token smuggling, **and zero-width-space-wrapped tokens**.
- Detected payloads are quarantined *inside* the fence and flagged; they never leak into the system prompt. Verified property: attacker strings appear in the untrusted block only, never in the instruction span (P7 parametrised 8 variants, all flagged).

### A4. Poisoning the models/caches
- **Model weights** are loaded from explicit env-pointed paths (`VERISAFE_*_WEIGHTS`); nothing is downloaded at runtime. Updating = operator-controlled file placement + restart.
- **RAG template cache** (gov-document templates) is a *retrieval cache, not source of truth*: template similarity is one signal among many; a forged doc that matches a cached layout still fails field-level + signature + API checks, and disagreement drops reliability.
- **Fusion checkpoints** (`VERISAFE_FUSION_DIR`) are retrained via the audited `fusion_train.py` flow (OOF CV + calibration); hand-edits break the schema check at load.

### A5. Availability / thermal
- Hard wall budget per job (default 300s), 10s stage floor, conservative short-circuit after confirmed positives — all three bound worst-case CPU per message (P8).
- Bounded thread caps (`VERISAFE_FFMPEG_THREADS=2`, heavy pool 2 workers) sized against this machine's documented thermal trip.
- Subprocess isolation per capability: one crashing/killing stage can't take down the job or the server.

### A6. Retention leakage
Zero retention is enforced in code, not policy: purge on completion AND failure paths, stale sweep with TTL, audit log carries hashes + verdicts only — never content. Backups of the quarantine mount would violate the contract; operators exclude it (documented in DEPLOYMENT.md disk-hygiene note).

## Known residual risks (honest)
- Heuristic phish scoring can be evaded by fresh, clean-looking infrastructure; confidence bands are calibrated to say "low confidence" in that regime rather than guess.
- Deepfake detectors degrade on heavily compressed social-media encodes; the transform battery measures that degradation and reports reduced confidence instead of a false clean.
- A fully offline deployment (no VT key) loses external reputation signals; coverage is reported transparently via `unavailable` records, never hidden.
- Single-host trust root: if the host is compromised at OS level, app-layer controls do not apply (out of scope for this document; standard host hardening applies).
