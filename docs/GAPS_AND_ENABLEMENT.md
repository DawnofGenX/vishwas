# Capability Gaps & Enablement Matrix

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
| `vt` (VirusTotal) | ⚠️ wired, no key set | `VERISAFE_VT_API_KEY` | Robust client in place (`src/verisafe/vt_client.py`: auth, 429 Retry-After + exponential backoff, injectable transport, verdict mapping; url_phishing + malicious_file routed through it, test_17 hermetic). Live probe 2026-08-20: **no anonymous tier** — `/api/v3/files/*` → HTTP 401 "X-Apikey header is missing", `/api/v3/intelligence/domains/*` → 404. Until a key is set: external reputation unavailable; strong local signals still work but confidence drops and report says "external lookup unavailable" |
| `llm` (narration) | ❌ no base URL/key | `OPENAI_API_KEY` or `VERISAFE_LLM_BASE_URL`+`API_KEY` (+`LLM_MODEL`) | Optional plain-language narration layer off; structured verdict still sent (template-first) |
| `model-weights` (deepfake/detection models) | ✅ **provisioned 2026-08-20** (3 of 3 public gates; 9.26 GB on disk) | inference **adapter registry** in place (`src/verisafe/model_adapters.py`, 7 families keyed by `VERISAFE_{EFFORT,DEMAMBA,FAKEMAMBA,AASIST,SSL_AUDIO,HAVIC,IMAGE_FACE}_WEIGHTS`; deepfake_video/audio + image_facecheck routed through it, test_14 hermetic). On disk under `/opt/verisafe/models/`: **aasist** `best_model.pth` 3,791,651,367 B (sha `febfc126a079…`, torch.load → dict{model_state,optimizer_state,epoch,…}), **effort** 3× checkpoints 1,213,769,519 B each (chameleon/ffpp/genimage, sha `2fc1b970…`/`8d86711f…`/`7c32ceb4…`, OrderedDict 681 tensors), **havic** `best_ft_model.pth` 858,837,738 B (sha `7a0e3ddc…`) + `pt_model.200.pth` 972,770,538 B (sha `a8c44dd5…`). All byte-sizes match manifest ±0.01%; sha256 sidecars written; manifest updated in place. Enable: `source scripts/provision_weight_env.sh` (or `--quiet` for exports only). Skipped-by-design: demamba/fakemamba (no public checkpoint), ssl_audio (requires HF token), image_face (cv2-data path). **Caveat:** these are raw state-dict/checkpoint artefacts without their architecture classes — `is_usable_model()` correctly reports them unusable until the corresponding model class is vendored, so named-model stages stay `unavailable` until architecture code lands; heuristic fallbacks remain the active path. |
| `dynamic-sandbox` (Cape/firejail) | ✅ firejail 0.9.72 installed 2026-08-20 (strace remains as fallback) | in PATH; Cape still needs `VERISAFE_CAPE_CMD` if desired | behavioural (sandboxed execution) analysis active |
| `digilocker` / `setu` (gov live APIs) | ❌ no e-KYC consumer creds (discovery-only surface IS reachable — see `research/INDIA_GOV_VERIFICATION.md` §1–§2) | `VERISAFE_DIGILOCKER_URL`+`KEY` (partnership-gated OAuth), `VERISAFE_APISETU_BASE`+`TOKEN` (partners.apisetu.gov.in client-credentials) — the API Setu *directory* search works anonymous (single-use bearer per call); actual `/certificate/v3/…` e-KYC calls need the registered consumer identity | Live-authority verification of govt docs unavailable; format/tampering/template checks + user-assisted QR verify instructions still run |
| Fusion checkpoints | ✅ trainer + collector live (2026-08-20) | `python3 -m verisafe.fusion_train --synthetic N --features F --target <name>` or `--dataset corpus.jsonl` (operator-labelled, `{features,label}` rows) → numpy mini-batch-SGD LR head → `$VERISAFE_FUSION_DIR/training/stack_<target>.json` (JSON, same contract as the OOF path); `FusionEngine.load_trained()` wires it; enable at serve time with `VERISAFE_FUSION_USE_TRAINED=1` (default OFF = byte-identical explicit-weight behaviour). Collector: `src/verisafe/dataset_collector.py` (synthesize/load_jsonl/save_jsonl/split/to_dataset_dict, typed DatasetError). test_16 hermetic (7 tests) | Explicit-weight fusion remains the default; provisioned checkpoints add calibrated LR stacking per target once enabled |
| RAG template cache | ◑ seeded (v1, built 2026-08-20) — 10 derived templates, 65 issuer-trust entries (from `apisetu_catalog_digest_2026-08-19.json`), 6 QR schemes, 16 official-content baselines; digilocker+nic reachable, incometax/nsdl/epfo unreachable from this box | rebuild via `python3 scripts/build_rag_cache.py` (honors `VERISAFE_RAG_CACHE` + `VERISAFE_RAG_VERSION`) | Template-similarity signal now active for the 10 known doc classes; stale/absent entries degrade confidence silently, never block. Retrieval cache only, never source of truth. |

## Current overall capability posture (this box)

| Target | Verdict quality today | Limiting factor |
|---|---|---|
| `url_phishing` | **Good** (stdlib heuristics + SSRF + host-string; DOM if browser) | VT reputation would add external confidence; PhishLLM would add ML |
| `malicious_file` | **Moderate** (magic bytes, entropy, PE headers, APK structure) | ClamAV/YARA/Quark/model weights all gated → signature-level detection absent |
| `gov_document` | **Good** (type ID, field validation, tamper checks; layout-tier extraction live via isolated docling install; **GPG signature verification live** via union keyring) | tesseract (scans) + DigitalLocker/SETU keys |
| `deepfake_video` | **Reduced** (heuristic frames + transform battery) | EFFORT/HAVIC weights on disk but **architecture classes not vendored** → `is_usable_model()` rejects raw state-dicts; DeMamba/Fake-Mamba have no public checkpoint |
| `deepfake_audio` | **Reduced** (offline numpy features + degradation consistency; near-silence abstains) | AASIST weight on disk but **architecture class not vendored**; SSL-audio needs HF token |
| `cross_modal` | **Reduced** (consistency heuristics) | HAVIC weight on disk but **architecture class not vendored** |
| `image_facecheck` | **Reduced** (integrity heuristics) | cv2 present; IMAGE_FACE weights absent (no public small-weight URL in manifest) |

## Enabling priority (recommended order for this hardware)

0. **Expose the isolated docling install** — `export PYTHONPATH=/home/hermes/docling-python` at service start. Already fully downloaded + model-cached in-dir (zero-cloud); instantly puts gov-document extraction on the structured-layout tier through the existing `docling` gate.
1. **Install tesseract** — cheapest win, immediately upgrades scanned-document coverage. `apt-get install tesseract-ocr`.
2. **Set a VirusTotal key** (`VERISAFE_VT_API_KEY`) — external reputation for both files and URLs is the highest-value signal-per-effort and needs no local compute.
3. **ClamAV + fresh DB** — local AV for the malware target; keep scans on small/medium files given the thread-cap budget.
4. ~~**Download model weights**~~ ✅ **Done 2026-08-20** — aasist/effort×3/havic×2 on disk under `/opt/verisafe/models` (9.26 GB, sha256-verified). `source scripts/provision_weight_env.sh` to export env vars. **Remaining gap:** vendor the architecture classes (EffortFaceForensics, AASIST head, HAVIC encoder) so `is_usable_model()` passes; until then adapters correctly report `unavailable`.
5. ~~**Retrain fusion**~~ ✅ **Trainer live 2026-08-20** — `fusion_train --synthetic/--dataset` + `dataset_collector` + `VERISAFE_FUSION_USE_TRAINED=1` wiring. Provision a real labelled corpus when available; synthetic path is tested end-to-end.
6. ~~**Wire a GPG keyring**~~ ✅ **Done 2026-08-20** — union-keyring model (`gpg_check.py`), `VERISAFE_GPG_TRUSTSTORE` + `VERISAFE_GPG_UNTRUSTED`, test_15 hermetic. DigitalLocker/SETU credentials still pending (partnership-gated).

Each step is independent and additive; nothing in this list requires cloud access — everything stays on-box per the zero-cloud constraint.
