# Vishwas Post-Migration Remediation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Restore Vishwas to full pre-migration capability — all learned-model weight gates live, ClamAV signature scanning operational, and the whole stack re-verified end-to-end on the new disk.

**Architecture:** Two provisioning tracks (model weights into `/opt/vishwas/models/`, ClamAV via apt) followed by a verification track (unit suite, gate smoke checks, E2E CLI verdict matrix). Weight salvage from the old SSD image is preferred over re-download (exact sha256-known files, no bandwidth); re-download is the fallback for AASIST/HAVIC. RawBMamba's 3 MB checkpoint comes from its upstream GitHub repo (in-repo `model/rawbmamba_bset.pt`).

**Tech Stack:** bash (rsync/wsl mount), apt (clamav), HuggingFace/GitHub downloads (curl), pytest (hermetic suite), vishwas CLI + webhook /health.

---

## Current context (verified 2026-08-24)

- Repo: `/home/hermes/vishwas`, clean at `b4a8c0d` (RawBMamba vendoring HEAD), suite collects 336 tests
- Services active: openwa :2785 (session main ready) + vishwas-webhook :2790 (/health: 7 deps available)
- Env wiring intact: `deploy/openwa-reply.env`, `deploy/vishwas-secrets.env` (VT key, YARA rules, FAKEMAMBA path)
- YARA bundle intact (80 files); stale-sweep cron OK; RAG cache fresh (2026-08-19 digest)

**Broken after migration:**

| Gap | Detail | Fix source |
|---|---|---|
| All 4 weight gates dead | `/opt/vishwas/models/` absent on this disk (lived in old root fs) | Salvage from SSD image (preferred) or re-download AASIST (3.7 GB) + HAVIC (859 MB) per `scripts/fetch_model_weights.sh`; EFFORT was always skipped (thermal caps); RawBMamba = 3 MB from upstream repo |
| clamscan missing | ClamAV never installed on new system | `apt install clamav clamav-daemon` + `freshclam` |

**Key facts the implementer must know:**
- Old SSD image mounts via Windows admin PowerShell: run `powershell.exe -Command "Start-Process powershell -Verb RunAs -ArgumentList '-File C:\path\to\attach_disk.ps1'"` — script exists at `~/transfer_logs/attach_disk.ps1`; previously mounted at `/dev/sde2 → /mnt/wsl/ubuntu-ssd`. Cross-disk I/O is VERY SLOW (~30 MB/s): run rsync in background with notify_on_complete, never foreground.
- `VISHWAS_FAKEMAMBA_WEIGHTS=/opt/vishwas/models/rawbmamba/rawbmamba_bset.pt` is already set in `deploy/vishwas-secrets.env` — do not change the path; create the directory it points to.
- RawBMamba checkpoint provenance: sha256 prefix `8536477b…89bd`, 2.97 MB. Source repo `cyjie429/RawBMamba`, file `model/rawbmamba_bset.pt`.
- AASIST checkpoint: 3791.7 MB expected size (fetcher verifies by size). HAVIC: 858.8 MB (`best_ft_model.pth` only; companion file intentionally skipped).
- Thermal protocol: check `/sys/class/thermal/thermal_zone*/temp < 70000` before heavy runs; sequential only.
- The loaders swallow exceptions and return None (silent). Always verify gates via `provision_weight_env.sh` + a direct adapter probe, never by "it didn't error".

---

## Task 1: Attempt weight salvage from old SSD image

**Objective:** Copy `/opt/vishwas/models/` off the old root fs if remounting is possible.

**Files:**
- Create: `/opt/vishwas/models/**` (populated)
- No repo changes

**Step 1:** Ask the operator to run (Windows admin PowerShell) or approve running:
```bash
powershell.exe -Command "Start-Process powershell -Verb RunAs -ArgumentList '-File C:\Users\<user>\transfer_logs\attach_disk.ps1'"
```
(The .ps1 is already on the Windows side at the path recorded in transfer_logs; wsl --mount PHYSICALDRIVE1.)

**Step 2:** Wait for mount to appear: `ls /mnt/wsl/ubuntu-ssd/opt/vishwas/models` (poll up to ~2 min).

**Step 3 (background, notify_on_complete=true):** If present:
```bash
sudo mkdir -p /opt/vishwas && sudo chown hermes:hermes /opt/vishwas
rsync -a --info=stats2 /mnt/wsl/ubuntu-ssd/opt/vishwas/models/ /opt/vishwas/models/
```
Expected: aasist/best_model.pth (3.79 GB), effort/, havic/best_ft/best_ft_model.pth, rawbmamba/rawbmamba_bset.pt + PROVENANCE.md.

**Step 4:** Verify checksum of the small file: `sha256sum /opt/vishwas/models/rawbmamba/rawbmamba_bset.pt | grep ^8536477b`.

**Step 5:** Unmount again exactly as done earlier: `cd ~ && sudo umount /mnt/wsl/ubuntu-ssd && sudo losetup -d /dev/loopN`. Skip Tasks 2–4 that are superseded; go to Task 5.

**If mount refused (disk detached on Windows side):** skip Task 1 and execute Tasks 2–4 instead.

---

## Task 2: Download AASIST weights (fallback track)

**Objective:** Fetch the 3.79 GB AASIST checkpoint from HuggingFace when salvage isn't possible.

**Files:**
- Create: `/opt/vishwas/models/aasist/best_model.pth`

**Step 1:** Prepare dir: `sudo mkdir -p /opt/vishwas/models/{aasist,havic,rawbmamba} && sudo chown -R hermes:hermes /opt/vishwas`

**Step 2 (background, notify_on_complete=true):** Real download via the project fetcher (it verifies sizes):
```bash
export VISHWAS_MODEL_DIR=/opt/vishwas/models VISHWAS_FETCH_DRY_RUN=0
bash ~/vishwas/scripts/fetch_model_weights.sh
```
Note: this also attempts havic — fine, Task 3 just verifies. Expect ~15–25 min at typical HF speeds; thermal-safe (network-bound).

**Step 3:** Verify size: `stat -c%s /opt/vishwas/models/aasist/best_model.pth` ≈ 3975566336 bytes (±fetcher tolerance).

---

## Task 3: Download HAVIC weights (fallback track)

**Objective:** Fetch the 859 MB HAVIC fine-tuned checkpoint.

**Files:**
- Create: `/opt/verisave/models/havic/best_ft/best_ft_model.pth` (NOTE: correct path is `/opt/vishwas/models/havic/best_ft/best_ft_model.pth`)

**Step 1:** Covered by Task 2 Step 2 (same FETCHABLE record). If run standalone:
```bash
mkdir -p /opt/vishwas/models/havic/best_ft
curl -L --fail -o /opt/vishwas/models/havic/best_ft/best_ft_model.pth \
  https://huggingface.co/JielunPeng/HAVIC/resolve/main/best_ft_model.pth
```

**Step 2:** Verify: size ≈ 900 MB; `python3 -c "import torch; torch.load('<path>', map_location='cpu')"` loads without exception.

**Gate decision:** EFFORT stays unprovisioned BY DESIGN (CC BY-NC + >1 GB × 3 checkpoints vs thermal caps). Do not download it.

---

## Task 4: Fetch RawBMamba checkpoint (small, license-caveated)

**Objective:** Restore the 3 MB fakemamba-slot checkpoint.

**Files:**
- Create: `/opt/vishwas/models/rawbmamba/rawbmamba_bset.pt`
- Create: `/opt/vishwas/models/rawbmamba/PROVENANCE.md`

**Step 1:**
```bash
curl -L --fail -o /opt/vishwas/models/rawbmamba/rawbmamba_bset.pt \
  https://github.com/cyjie429/RawBMamba/raw/main/model/rawbmamba_bset.pt
sha256sum /opt/vishwas/models/rawbmamba/rawbmamba_bset.pt   # expect 8536477b…89bd
```
(If the GitHub raw URL 404s, consult `~/.hermes/skills/devops/vishwas-operations/references/mamba-deepfake-weights-scouting.md` §RawBMamba for the current location.)

**Step 2:** Recreate PROVENANCE.md (license UNRESOLVED, evaluation-only; vendored commit b4a8c0d; sha256 above).

**Step 3:** Self-verify the vendored integration: `PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:/home/hermes/vishwas/src python3 ~/vishwas/scripts/verify_rawbmamba.py` — expects all-pass.

---

## Task 5: Install ClamAV

**Objective:** Restore the malicious-file signature-scan dep.

**Step 1 (needs approval — system package install):**
```bash
sudo apt-get install -y clamav clamav-daemon
```

**Step 2:** Update signatures: `sudo freshclam` (expect main.cvd/daily.cvd downloaded; may take several minutes).

**Step 3:** Smoke test: `bash ~/vishwas/scripts/run_vishwas.sh cli --file <(printf 'X5O!P%%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*') 2>/dev/null | grep -i clam` — expect detection evidence in fused output. Simpler canonical fixture: write EICAR string to /tmp/eicar.txt first (plain text routes away from MZ magic-byte routing; use an MZ-prefixed fixture per references/yara-rules-bundle.md if PE routing is needed).

**Step 4:** Restart webhook so the deps list picks up clamscan: `systemctl --user restart vishwas-webhook`.

**Step 5:** Verify `/health`: `curl -s http://127.0.0.1:2790/health | python3 -c "import json,sys; d=json.load(sys.stdin); print('clamav' in str(d['deps']))"` — expect True (dep count 7→8).

---

## Task 6: Gate verification — all four learned stages load

**Objective:** Prove each gate resolves to a usable model object (defeats the silent-None pitfall).

**Step 1:** Direct adapter probe (NOT through CLI):
```bash
cd ~/vishwas && PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:src \
python3 -c "
from vishwas.model_adapters import resolve, is_usable_model
for g in ['VISHWAS_AASIST_WEIGHTS','VISHWAS_EFFORT_WEIGHTS','VERISAVE_HAVIC_WEIGHTS'.replace('SAVE','SAFE'),'VISHWAS_FAKEMAMBA_WEIGHTS']:
    try:
        obj = resolve(g).load(resolve(g).path)
        print(g, type(obj).__name__, is_usable_model(obj))
    except Exception as e:
        print(g, 'ERROR', e)
"
```
Expected: AASIST/HAVIC/Fakemamba → `True`; EFFORT → honest unavailable (absent by design).

**Step 2:** Confirm provision script sees them:
`eval "$(bash scripts/provision_weight_env.sh --quiet | grep '^export')"; env | grep -c VISHWAS_.*WEIGHTS` — expect 3 set (+FAKEMAMBA from secrets env = 4 live gates).

**Thermal note:** model loads are ~30 s combined but AASIST alone is 16 s / 3.79 GB disk read — one process, sequential, zones were 24 °C at plan time.

---

## Task 7: Hermetic suite + E2E verdict matrix re-run

**Objective:** Full regression proof on the new disk.

**Step 1:** `cd ~/vishwas && PYTHONPATH=src python3 -m pytest tests/ -q` — expect **333+ passed, ≤3 skipped**, no new failures vs pre-migration baseline (was 333 passed / 3 skipped).

**Step 2:** CLI verdict matrix (sequential; thermal precheck first):
- Text URL light path: `bash scripts/run_vishwas.sh cli --text "check http://google.com"` — <1 s, calibrated low.
- Audio fixture: `bash scripts/run_vishwas.sh cli --file tests/fixtures/smoke_5s.wav` — ~24 s, AASIST stage now LIVE (not unavailable).
- Malicious file probe (EICAR in MZ wrapper per yara reference): expect clamscan FOUND + yara hit + VT evidence.

**Step 3:** Webhook round-trip still green: `curl -s http://127.0.0.1:2790/health | grep -o '"status": "ok"'`.

---

## Task 8: Update ops docs + skill (dated)

**Objective:** Record the migration-remediation facts durably.

**Files:**
- Modify: `~/vishwas/docs/OPERATIONS.md` — add dated subsection "2026-08-24 post-migration reprovision" (what was restored, sources, sha256s).
- Modify: `docs/PERFORMANCE.md` ONLY if stage timings shifted materially.
- Patch: skill `vishwas-operations` SKILL.md — replace "Live gates" wording with post-restoration state; note the salvage-vs-download provenance.

**Commit:** `git add docs/ && git commit -m "docs(ops): post-migration weight+clamav reprovision record"`.

---

## Risks / tradeoffs / open questions

1. **SSD remount needs operator action** (admin PowerShell on Windows side) — Task 1 may block on the user. Fallback (Tasks 2–4) is fully autonomous but pulls ~4.7 GB.
2. **AASIST re-download integrity**: fetcher verifies by size only; a corrupted-but-right-sized file would surface at Task 6 load time (strict load fails loudly there — acceptable backstop).
3. **EFFORT remains unprovisioned by design** — video deepfake stage will still report degraded vs its best possible posture. Documented tradeoff, not a defect.
4. **apt install requires approval** and network; freshclam DB download can be slow.
5. **Open question for operator**: salvage (remount SSD once more) vs clean re-download — recommend salvage since the SSD is likely still attached on the Windows side.
