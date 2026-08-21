# Zero-Retention E2E Proof — 2026-08-21 (roadmap Phase 5)

End-to-end runs through the REAL CLI (`python3 -m verisafe.app cli`) with all
three learned-model gates provisioned, one per input kind, verifying after
each run that (a) the job quarantine dir is gone, (b) the purge-audit line
shows `residual_paths == []`, and (c) no job-id-matching temp files remain.

Method: sequential heavy runs, `VERISAFE_FFMPEG_THREADS=1`, canonical env
`PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:<repo>/src`
(see Finding A — bare `PYTHONPATH=src` fast-fails learned tiers), weights via
`scripts/provision_weight_env.sh`. Drivers persisted at `/tmp/p5_e2e/`
(`run_e2e.py`, `capture_checks.py`, per-kind stdout) for reproduction; both
fixture dirs `tests/fixtures/govdoc/` and `tests/fixtures/malware/` were
EMPTY, so a text-layer PDF stub (`aadhaar.pdf`, 605 B) and a PE stub
(`stub.exe`, 66 B) were created in tempdir (deviation noted honestly).

## Results — 5/5 PASS

| kind | job_id | wall | purged | residual_paths | /tmp hits | verdict |
|---|---|---|---|---|---|---|
| url | `job_3251d28c7310` | 0.4s | true | [] | 0 | PASS |
| audio | `job_4f0d56725376` | 22.3s | true | [] | 0 | PASS |
| video | `job_252497028406` | 35.4s | true | [] | 0 | PASS |
| document | `job_5136fe4e9502` | 0.4s | true | [] | 0 | PASS |
| malware | `job_10f8c692ad65` | 22.6s | true | [] | 0 | PASS |

Raw audit lines (all `"reason": "completed", "failures": 0`):
```
{"job_dir": "/tmp/verisafe-work/cli-text/job_3251d28c7310", "ts": 1787310416, "reason": "completed", "artifacts_deleted": 2, "failures": 0, "residual_paths": []}
{"job_dir": "/tmp/verisafe-work/cli-smoke_5s.wav/job_4f0d56725376", "ts": 1787310438, ..., "residual_paths": []}
{"job_dir": "/tmp/verisafe-work/cli-clip_av.mp4/job_252497028406", "ts": 1787310473, ..., "residual_paths": []}
{"job_dir": "/tmp/verisafe-work/cli-aadhaar.pdf/job_5136fe4e9502", "ts": 1787310709, ..., "residual_paths": []}
{"job_dir": "/tmp/verisafe-work/cli-stub.exe/job_10f8c692ad65", "ts": 1787310732, ..., "residual_paths": []}
```
Post-run state: `/tmp/verisafe-work/` no longer exists at all (every session
shell removed).

Learned-tier proof in-run: audio → `[heavy] aasist_detector ok
{"prob_deepfake": 0.997, "n_crops_scored": 3}` + degradation battery ok;
video → `[heavy] effort_face_forensics ok {"prob_deepfake": 0.304,
"n_frames_scored": 8}` + `[heavy] havic_crossmodal_model ok
{"prob_inconsistent": 0.994}` + cross_modal_av ok; url → heuristics ok with
`vt_url_reputation unavailable` (gated as documented).

## Honest findings (verified during this proof; documented, not patched — out of scope)

- **A. PYTHONPATH fragility** — with bare `PYTHONPATH=src`, torch is
  unimportable → learned tiers fast-fail to `unavailable` in <1s while the
  deps list still claims `model-weights` present. Always launch via
  `scripts/run_verisafe.sh` (which wires the canonical paths).
- **B. ClamAV guard bug** (`malware_file.py:_clamav`) — `os.access("clamscan",
  os.X_OK)` on a bare name is cwd-relative → False, so clamscan reports
  unavailable unless `VERISAFE_CLAMSCAN_BIN` is set to an absolute path.
  GAPS row claiming local AV "active" is overstated until fixed.
- **C. YARA off in practice** — requires `VERISAFE_YARA_RULES`; no rules
  bundle ships in-repo → unavailable in every run.
- **D. gov_document unreachable via CLI file path** — router checks
  `_looks_gov_artifact(art_filename=target_hint)`, never the real filename;
  CLI `--file` forces empty hint → PDFs route `document_generic`, which
  emitted zero usable signals even for a text-layer PDF. Purge still clean;
  capability reachability gap recorded.
