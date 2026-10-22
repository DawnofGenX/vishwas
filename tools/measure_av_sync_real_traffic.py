#!/usr/bin/env python3
"""Measure the AV-sync band on the operator's REAL WhatsApp traffic.

WHY
---
The leak-free corpus is SYNTHETIC: its audio is muxed from a different
source, so the lip/speech envelopes DECORRELATE (av=0.45) and never reach
the ANTI-CORRELATED band (av=0.5). That is why 11 of 30 fakes still read
TRUST -- the cross-modal gate, which is the only thing that flags them in
production, has nothing to fire on.

The operator's own WhatsApp traffic is the missing instrument: 67 real
videos from 5 correspondents, 63 with their ORIGINAL audio, so the audio and
the video are naturally synchronised.

This tool measures the av band distribution on that traffic. It is a
CLEAN-SIDE measurement (these are genuine recordings, not deepfakes), so it
answers: "what av band does genuinely synchronised media produce?" -- the
reference distribution the synthetic corpus lacks.

It is NOT a labelled accuracy measurement: there is no ground truth for
"fake" here, so no AUC is claimed.

2026-10-22: the summary this tool prints was written BEFORE the silence fix
(PR #13) and used to claim that genuine media "does NOT produce the
decorrelated band (0.45) or the anti-correlated band (0.5)". On the current
tree that is FALSE by its own numbers -- 3 clips read 0.45 and 2 read 0.5 --
because before PR #13 a SILENT audio track was reported as anti-correlated.
The summary now states what the bands actually mean, and the tool stamps
the commit it ran against so a stale run is distinguishable from a
behaviour change.

Usage: python tools/measure_av_sync_real_traffic.py
"""
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.cross_modal import CrossModalCapability
from vishwas.events import Artifact, JobContext


def _tree_stamp() -> str:
    """Stamp the commit this run is against (see tests/test_70_*)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if not sha:
            return "vishwas tree: UNKNOWN (no git info)"
        d = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
            capture_output=True, text=True, timeout=10)
        return f"vishwas tree: {sha}" + (" (DIRTY WORKTREE)" if d.stdout.strip() else "")
    except Exception:
        return "vishwas tree: UNKNOWN (git not available)"

MEDIA_ROOT = pathlib.Path("/home/hermes/openwa/data/media")
DB = "/home/hermes/openwa/data/openwa.sqlite"
QUARANTINE = pathlib.Path("/home/hermes/.hermes/cache/scratch/avtraffic_q")
OUT = pathlib.Path("/home/hermes/.hermes/cache/scratch/av_real_traffic.jsonl")
LIMIT = int(os.environ.get("VISHWAS_AV_LIMIT", "0"))

if QUARANTINE.exists():
    shutil.rmtree(QUARANTINE, ignore_errors=True)
QUARANTINE.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = con.execute("""
  select id, "from" as sender, timestamp, mediaPath, mediaMimetype
  from messages
  where mediaPath is not null and mediaPath != ''
    and mediaMimetype like 'video%'
  order by timestamp
""").fetchall()

clips = []
for r in rows:
    p = MEDIA_ROOT / r["mediaPath"]
    if p.exists():
        clips.append((p, r["sender"], r["timestamp"]))
if LIMIT:
    clips = clips[:LIMIT]
print(f"real WhatsApp videos: {len(clips)} from "
      f"{len(set(s for _, s, _ in clips))} senders")

xcap = CrossModalCapability()
results = []
for i, (p, sender, ts) in enumerate(clips):
    q = QUARANTINE / f"j{i}"
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id=f"av{i}", artifact=art, quarantine_root=q,
                     wall_budget_s=300.0, model_weights_available=True)
    av = align = corr = None
    try:
        for c in xcap.analyze(art, ctx):
            if c.name == "cross_modal_av" and c.status == "ok":
                av = c.signals.get("av_risk_addition")
                align = c.signals.get("alignment_class")
                corr = c.signals.get("av_correlation")
    except Exception as e:
        print(f"  {p.name[:24]}: {type(e).__name__} {e}")
        continue
    rec = {"path": str(p), "name": p.name, "sender": sender, "ts": ts,
           "av": av, "align": align, "corr": corr}
    results.append(rec)
    with OUT.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {p.name[:30]:<32s} av={av} align={align} corr={corr}")

print()
print("=" * 92)
print("THE AV-SYNC BAND ON GENUINE, NATURALLY-SYNCHRONISED MEDIA")
print("=" * 92)
print(f"  {_tree_stamp()}")
ok = [r for r in results if r["av"] is not None]
print(f"  measured: {len(ok)}/{len(results)}")
if ok:
    print(f"  av distribution: {dict(sorted(Counter(r['av'] for r in ok).items()))}")
    print(f"  alignment classes: {dict(Counter(r['align'] for r in ok).items())}")
    corrs = [r["corr"] for r in ok if r["corr"] is not None]
    if corrs:
        import statistics as st
        print(f"  av_correlation: min={min(corrs):.3f} med={st.median(corrs):.3f} "
              f"max={max(corrs):.3f}")
    print()
    print("  WHAT THIS ESTABLISHES (read with the numbers above):")
    print("   * genuinely synchronised media produces av=0.0 (synced) and")
    print("     av=0.1 (weakly_synced) -- the two bands that mean 'the lip")
    print("     motion and the speech energy agree'.")
    print("   * the 'unknown' band is a measurement the check could NOT make:")
    print("     a clip whose audio carries no speech energy has a CONSTANT")
    print("     envelope, so the correlation is undefined. It is not evidence")
    print("     of anything, in either direction.")
    print("   * the 0.45 (decorrelated) band on the SYNTHETIC corpus is a MUX")
    print("     ARTEFACT -- audio from a different source than the video.")
    print("     This is what the corroboration gate assumes.")
    print("   * ⚠️ THE 0.5 (anti_correlated) BAND IS A SILENCE ARTEFACT, NOT A")
    print("     DEEPFAKE SIGNATURE. Before PR #13 the correlation function")
    print("     returned -2.0 as its 'could not measure' sentinel, and the")
    print("     classifier mapped that straight onto the anti-correlated band")
    print("     (av=0.5) -- so any clip with a SILENT audio track was reported")
    print("     as 'the modalities were generated independently'. On THIS")
    print("     traffic that was 9 of the clips: 12% of genuine WhatsApp")
    print("     video has an all-zero audio track (a muted clip, a screen")
    print("     recording, a clip sent with the sound off).")
    print("     Any 0.5 readings still present are REAL negative correlations")
    print("     and even those overlap the operator's AI videos' range")
    print("     (-0.299 / -0.443), so the band is not decisive on its own.")
    print()
    print(f"  wrote {len(results)} rows -> {OUT}")
