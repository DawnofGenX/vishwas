# Capability Gaps & Enablement Matrix

**Roadmap:** see `.hermes/plans/2026-08-20_222933-verisafe-full-roadmap.md` — full-scale execution plan (P0 hygiene → P1 learned detectors → P2 delivery/ops → P3 languages → P4 enablement/deploy → P5 re-baseline). This inventory is updated as each phase lands.

Honest, current-state inventory. Every row states what runs *right now* on this host, what is gated off, exactly what enables it, and the effect on verdict confidence when absent. No feature here is silently assumed working — unprovisioned stages record themselves as `unavailable` evidence so reports can say "I could not fully check X" instead of guessing.

## How to read this

- **dep id** = string in the availability gate (`detect_available_deps()` in `app.py`).
- **Now?** = state observed on this i5-8250U host at last probe (2026-08-19).
- **Enables via** = env var / binary / file you provide.
- **If absent** = what the user-facing result says.

## Base tooling

| dep | Now? | Enables via | If absent |
|---|---|---|---|
| `media-tools` (ffmpeg/ffprobe) | ✅ present | `VERISAFE_FFMPEG_BIN`/`FFPROBE_BIN` point at binaries; threads capped by `VERISAFE_FFMPEG_THREADS` (default 2, thermal-safe) | Video/audio/image probes fail → deepfake targets degrade to `unable_to_verify` with explicit reason |
| `strace` | ✅ present | binary in PATH (or `VERISAFE_CAPE_CMD`) | Dynamic sandbox path limited; static analysis still runs |
| `browser` (playwright) | ✅ installed | pip package | DOM-based phishing heuristics fall back to host-string-only scoring (lower confidence band) |
| `gpg` (binary) | ✅ present + keyring wired (2026-08-20) | `VERISAFE_GPG_TRUSTSTORE` = dir of trusted signer public-key blobs (.asc/.gpg, ownertrust=FULL); optional `VERISAFE_GPG_UNTRUSTED` = dir of known-but-unvouched keys. Zero-network union-keyring model (`src/verisafe/gpg_check.py`): both rings imported into one ephemeral GNUPGHOME before `gpg --verify`; layout-agnostic VALIDSIG/KEY_CONSIDERED parsing; test_15 hermetic (8 tests, real gpg) | `digital_signature` check now runs real verification: trusted→ok, valid-untrusted→degraded+flagged, tampered→failed, no keyring→unavailable |
| `ocr` (tesseract) | ✅ present | in PATH (verified); optionally pin `VERISAFE_TESSERACT_BIN` | scanned-doc OCR works |
| `clamav` | ✅ present | ClamAV 1.5.3 with fresh DB in `/var/lib/clamav` (last refresh 2026-08-18); run `freshclam` periodically | local AV signature signal active |
| `yara` (yara_x) | ✅ present (via `scripts/run_verisafe.sh` / system `.pth` shim) | yara_x lives in `/home/hermes/pylibs`; set `VERISAFE_YARA_RULES` to add custom rules | YARA hit family active (default bundled rules) |
| `pe-static` / `pe-lief` (pefile, lief) | ✅ present (same `.pth`/launcher path as yara) | both importable from `/home/hermes/pylibs` | full PE/LIEF structural analysis active |
| `docling` | ✅ active when started via **`scripts/run_verisafe.sh`** (isolated ~5.5 GB tree in `/home/hermes/docling-python`, kept OUT of site-packages by design) | the launcher prepends it to PYTHONPATH; one-off use: `export PYTHONPATH=/home/hermes/docling-python:$PYTHONPATH` | structured-layout / scanned-doc extraction on the best tier |
| `cv2` (opencv) | ✅ present (same `.pth`/launcher path) | imports from `/home/hermes/pylibs` | frame-level image ops active; face forensics still pending weights (D5) |
| `vt` (VirusTotal) | ⚠️ wired, no key set — **BLOCKED ON CREDENTIALS (roadmap T4.1, re-checked 2026-08-21: no `VERISAFE_VT_API_KEY` in env/profile)** | `VERISAFE_VT_API_KEY` — free key from virustotal.com; once set, run the T4.1 live probe (1 known-clean + 1 known-malicious hash/domain, capture into `docs/research/VT_LIVE_EVIDENCE.md`) | Robust client in place (`src/verisafe/vt_client.py`: auth, 429 Retry-After + exponential backoff, injectable transport, verdict mapping; url_phishing + malicious_file routed through it, test_17 hermetic). Live probe 2026-08-20: **no anonymous tier** — `/api/v3/files/*` → HTTP 401 "X-Apikey header is missing", `/api/v3/intelligence/domains/*` → 404. Until a key is set: external reputation unavailable; strong local signals still work but confidence drops and report says "external lookup unavailable" |
| `llm` (narration) | ❌ no base URL/key | `OPENAI_API_KEY` or `VERISAFE_LLM_BASE_URL`+`API_KEY` (+`LLM_MODEL`) | Optional plain-language narration layer off; structured verdict still sent (template-first) |
| `model-weights` (deepfake/detection models) | ⚠️ **weights provisioned 2026-08-20** (3 of 3 public gates; 9.26 GB on disk) — **arch-vendor status: AASIST ✅ vendored 2026-08-21** (learned audio tier LIVE); **EFFORT ✅ vendored 2026-08-21** (learned video tier LIVE); **HAVIC ✅ vendored 2026-08-21** (learned cross-modal tier LIVE) | inference **adapter registry** in place (`src/verisafe/model_adapters.py`, 7 families keyed by `VERISAFE_{EFFORT,DEMAMBA,FAKEMAMBA,AASIST,SSL_AUDIO,HAVIC,IMAGE_FACE}_WEIGHTS`; deepfake_video/audio + image_facecheck routed through it, test_14 hermetic). On disk under `/opt/verisafe/models/`: **aasist** `best_model.pth` 3,791,651,367 B (sha `febfc126a079…`, torch.load → dict{model_state,optimizer_state,epoch,…}), **effort** 3× checkpoints 1,213,769,519 B each (chameleon/ffpp/genimage, sha `2fc1b970…`/`8d86711f…`/`7c32ceb4…`, OrderedDict 681 tensors), **havic** `best_ft_model.pth` 858,837,738 B (sha `7a0e3ddc…`) + `pt_model.200.pth` 972,770,538 B (sha `a8c44dd5…`). All byte-sizes match manifest ±0.01%; sha256 sidecars written; manifest updated in place. Enable: `source scripts/provision_weight_env.sh` (or `--quiet` for exports only). Skipped-by-design: demamba/fakemamba (no public checkpoint), ssl_audio (requires HF token), image_face (cv2-data path). **AASIST:** architecture class vendored (`model_archs/aasist.py`, WavLM-Large + HtrgGAT); real-weight verify 2026-08-21 → load 15.4 s, inference 13.7 s, `is_usable_model()` True, spoof posterior in range, thermal 49 °C (normal tier) — see `docs/research/ARCH_VENDOR_EVIDENCE_2026-08-21.md`. **EFFORT:** architecture class vendored (`model_archs/effort.py`, CLIP-style ViT-L/14 303M + OrthAlign rank-1 residuals); real-weight verify 2026-08-21 → 100% key coverage (681/681), inference ~1.7 s/image, label order confirmed [real,fake] on known-real photos (posterior 0.29–0.35), thermal 55 °C (normal tier) — same evidence file. **Caveat (havic):** none remaining — architecture class vendored (`model_archs/havic.py` + `_havic/` backend, MIT), wiring live through the adapter seam; real-weight smoke 2026-08-21 → load 8.6 s, inference 5.1 s per clip (NOT SLOW), posterior in range, thermal 48 °C (normal tier) — same evidence file. `pt200` chain entry is a headless pretrain payload that cannot score (documented honestly in the module docstring). |
| `dynamic-sandbox` (Cape/firejail) | ✅ firejail 0.9.72 installed 2026-08-20 (strace remains as fallback) | in PATH; Cape still needs `VERISAFE_CAPE_CMD` if desired | behavioural (sandboxed execution) analysis active |
| `digilocker` / `setu` (gov live APIs) | ❌ no e-KYC consumer creds (discovery-only surface IS reachable — see `research/INDIA_GOV_VERIFICATION.md` §1–§2) | `VERISAFE_DIGILOCKER_URL`+`KEY` (partnership-gated OAuth), `VERISAFE_APISETU_BASE`+`TOKEN` (partners.apisetu.gov.in client-credentials) — the API Setu *directory* search works anonymous (single-use bearer per call); actual `/certificate/v3/…` e-KYC calls need the registered consumer identity | Live-authority verification of govt docs unavailable; format/tampering/template checks + user-assisted QR verify instructions still run |
| Fusion checkpoints | ✅ trainer + collector live (2026-08-20) | `python3 -m verisafe.fusion_train --synthetic N --features F --target <name>` or `--dataset corpus.jsonl` (operator-labelled, `{features,label}` rows) → numpy mini-batch-SGD LR head → `$VERISAFE_FUSION_DIR/training/stack_<target>.json` (JSON, same contract as the OOF path); `FusionEngine.load_trained()` wires it; enable at serve time with `VERISAFE_FUSION_USE_TRAINED=1` (default OFF = byte-identical explicit-weight behaviour). Collector: `src/verisafe/dataset_collector.py` (synthesize/load_jsonl/save_jsonl/split/to_dataset_dict, typed DatasetError). test_16 hermetic (7 tests) | Explicit-weight fusion remains the default; provisioned checkpoints add calibrated LR stacking per target once enabled |
| RAG template cache | ◑ seeded (v1, built 2026-08-20) — 10 derived templates, 65 issuer-trust entries (from `apisetu_catalog_digest_2026-08-19.json`), 6 QR schemes, 16 official-content baselines; digilocker+nic reachable, incometax/nsdl/epfo unreachable from this box | rebuild via `python3 scripts/build_rag_cache.py` (honors `VERISAFE_RAG_CACHE` + `VERISAFE_RAG_VERSION`) | Template-similarity signal now active for the 10 known doc classes; stale/absent entries degrade confidence silently, never block. Retrieval cache only, never source of truth. |
| `i18n` (reply languages) | ✅ **7/7 languages render own strings** (merged 2026-08-21, roadmap Phase 3 under autonomy default) — en authoritative; ta/te/ml/kn/bn **DRAFT**; hi best-effort | native review still PENDING per language: drafts live in `docs/i18n/*.draft.md`; corrections land via the documented `load_custom_strings()` JSON overlay or a follow-up merge; test_08 pins the gate (`REVIEWED_LANGUAGES` stays empty until sign-offs) | no key falls back silently anymore — every supported language renders its own verdict text; unverified lines flagged below, never presented as localized |

### Language review state (roadmap Phase 3 — merged 2026-08-21, native sign-off still pending)

The five Indian-language drafts (ta, te, ml, kn, bn) were merged into `i18n._DEFAULTS` under the user's standing autonomy directive (gates take stated defaults; each default noted here). **Status is DRAFT, not reviewed:** any line carrying `[?]` in its draft file was sub-90% machine confidence and must be double-checked by a native reviewer before that language can be called localized. The full flagged inventory (32 keys total):

| Lang | Flagged keys (see `docs/i18n/<lang>.draft.md` for line-level notes) |
|---|---|
| ta | confidence_line, evidence_missing, heavy_pending_notice, progress_media, verdict_caution, verdict_do_not_use |
| te | confidence_line, evidence_missing, heavy_pending_notice, verdict_caution, verdict_do_not_use |
| ml | confidence_line, evidence_missing, heavy_pending_notice, verdict_caution, verdict_do_not_use |
| kn | evidence_missing, greeting, heavy_pending_notice, progress_media, verdict_caution, verdict_do_not_use, verdict_unable |
| bn | confidence_line, evidence_missing, heavy_pending_notice, verdict_caution, verdict_do_not_use |
| hi | all 14 keys (existing 12 are pre-roadmap best-effort machine output; the 2 new heavy_* keys drafted this cycle) |

Review path for a native speaker: read one `docs/i18n/<lang>.draft.md`, correct/copy the lines, either write a small `i18n_extra.json` (auto-overlaid at import, zero code change) or request a module merge; when done, add the language to `REVIEWED_LANGUAGES` in `tests/test_08_i18n_report.py` (that arms the differ-from-English assertion) and re-run `scripts/i18n_export.py` to refresh `docs/i18n/en.md`.

## Current overall capability posture (this box)

| Target | Verdict quality today | Limiting factor |
|---|---|---|
| `url_phishing` | **Good** (stdlib heuristics + SSRF + host-string; DOM if browser) | VT reputation would add external confidence; PhishLLM would add ML |
| `malicious_file` | **Moderate** (magic bytes, entropy, PE headers, APK structure) | ClamAV/YARA/Quark/model weights all gated → signature-level detection absent |
| `gov_document` | **Good** (type ID, field validation, tamper checks; layout-tier extraction live via isolated docling install; **GPG signature verification live** via union keyring) | tesseract (scans) + DigitalLocker/SETU keys |
| `deepfake_video` | **Good** (learned EFFORT tier LIVE 2026-08-21 + heuristic frames + transform battery) | EFFORT arch vendored (CLIP-style ViT-L/14 303M + OrthAlign), real-weight verify 100% key coverage, ~1.7 s/frame, label order confirmed; HAVIC cross-modal now also LIVE (2026-08-21, see cross_modal row); DeMamba/Fake-Mamba have no public checkpoint |
| `deepfake_audio` | **Good** (learned AASIST tier LIVE 2026-08-21 + offline numpy features + degradation consistency; near-silence abstains) | AASIST arch vendored (WavLM-Large + HtrgGAT), real-weight verified 13.7 s/crop; final posture confirmed at Phase 5 re-baseline; SSL-audio still needs HF token (off by design) |
| `cross_modal` | **Good** (heuristic AV probe + learned HAVIC consistency LIVE 2026-08-21) | HAVIC arch vendored (`model_archs/havic.py` + `_havic/` backend, MIT), real-weight smoke load 8.6 s / infer 5.1 s per clip, NOT SLOW; numpy kaldi-fbank port validated vs torchaudio ≤5e-4; heuristic AV-correlation probe unchanged and still runs alongside |
| `image_facecheck` | **Reduced** (integrity heuristics) | cv2 present; IMAGE_FACE weights absent (no public small-weight URL in manifest) |

### Capability posture v2 — 2026-08-21 (roadmap P5 re-baseline; delta vs 2026-08-19)

| Target | Aug-19 baseline | Now (2026-08-21) | What moved |
|---|---|---|---|
| `deepfake_audio` | Reduced (heuristics only) | **Good** | AASIST learned tier vendored + live (E2E-confirmed in CLI runs, not just adapter-level) |
| `deepfake_video` | Reduced (heuristics only) | **Good** | EFFORT learned tier vendored + live; adversarial small-N check: STABLE under 6 codec/scale/fps transforms |
| `cross_modal` | Reduced (consistency heuristics) | **Good** | HAVIC learned tier vendored + live (MIT, NOT SLOW at ~5 s/clip); heuristic AV probe still runs alongside |
| `gov_document` | Moderate | **Good** | GPG union-keyring signature verify live; CA trust-store seeded (ISRG Root X1) + anchored-chain PAdES test |
| `malicious_file` | Moderate | **Moderate** (unchanged rating; +entropy/PE confirmed live E2E) | ClamAV present but guard-bug-gated (Finding B in ZERO_RETENTION_E2E doc); YARA needs a rules bundle (Finding C) — both documented, unfixed by design this cycle |
| `url_phishing` | Good | **Good** (unchanged) | VT reputation still cred-gated (T4.1 blocked-on-credentials) |
| `image_facecheck` | Reduced | **Reduced** (unchanged) | weights simply don't exist publicly |

**Delta paragraph (Aug-19 → Aug-21):** the headline move is the entire
deepfake stack: three learned detectors (AASIST/EFFORT/HAVIC) went from
weights-on-disk-but-unusable to vendored, real-weight-verified, wired through
the arch-aware adapter seam, and confirmed inside real CLI runs — with
adversarial sensitivity STABLE across codec/scale/framerate transforms.
Supporting infrastructure landed alongside: non-blocking heavy follow-ups
(deterministic templates, en+hi byte-exact), rich /health with job counters,
stale-quarantine cron sweep (every 15 min), RAG freshness gating (14 d TTL +
digest), 7-language i18n with placeholder-leak contract (5 langs DRAFT
pending native review), CA trust-store + anchored PAdES, and full deployment
artefacts (compose/webhook/systemd, `docker compose config` clean). Still
gated exactly as documented: VT key, DigiLocker/API Setu creds, image_face
weights, LLM narration, HF-token SSL-audio, no-public-checkpoint
demamba/fakemamba. Honest gaps found during P5 and recorded (not patched):
ClamAV guard bug, YARA rules-bundle absence, gov-doc CLI routing gap.

## Enabling priority (recommended order for this hardware)

0. **Expose the isolated docling install** — `export PYTHONPATH=/home/hermes/docling-python` at service start. Already fully downloaded + model-cached in-dir (zero-cloud); instantly puts gov-document extraction on the structured-layout tier through the existing `docling` gate.
1. **Install tesseract** — cheapest win, immediately upgrades scanned-document coverage. `apt-get install tesseract-ocr`.
2. **Set a VirusTotal key** (`VERISAFE_VT_API_KEY`) — external reputation for both files and URLs is the highest-value signal-per-effort and needs no local compute.
3. **ClamAV + fresh DB** — local AV for the malware target; keep scans on small/medium files given the thread-cap budget.
4. ~~**Download model weights**~~ ✅ **Done 2026-08-20** — aasist/effort×3/havic×2 on disk under `/opt/verisafe/models` (9.26 GB, sha256-verified). `source scripts/provision_weight_env.sh` to export env vars. **Arch-vendor status:** ALL THREE public-gate families are vendored and real-weight-verified: AASIST ✅ (2026-08-21, learned audio tier live), EFFORT ✅ (2026-08-21, learned video tier live), HAVIC ✅ (2026-08-21, learned cross-modal tier live — load 8.6 s / infer 5.1 s per clip, NOT SLOW).
5. ~~**Retrain fusion**~~ ✅ **Trainer live 2026-08-20** — `fusion_train --synthetic/--dataset` + `dataset_collector` + `VERISAFE_FUSION_USE_TRAINED=1` wiring. Provision a real labelled corpus when available; synthetic path is tested end-to-end.
6. ~~**Wire a GPG keyring**~~ ✅ **Done 2026-08-20** — union-keyring model (`gpg_check.py`), `VERISAFE_GPG_TRUSTSTORE` + `VERISAFE_GPG_UNTRUSTED`, test_15 hermetic. DigitalLocker/SETU credentials still pending (partnership-gated).

Each step is independent and additive; nothing in this list requires cloud access — everything stays on-box per the zero-cloud constraint.
