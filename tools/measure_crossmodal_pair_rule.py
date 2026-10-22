#!/usr/bin/env python3
"""Measure the CANDIDATE cross-modal-only rules on the operator's real traffic.

WHY THIS TOOL EXISTS
---------------------
After the silence fix (PR #13) the av=0.5 band means "no speech energy",
and HAVIC is saturated on genuine traffic (>= 0.95 on 42 of 63 genuine
clips, median 0.980). That combination makes ONE rule look attractive:

    "flag when av_risk_addition == 0.5 AND havic >= 0.9"

It is a CROSS-MODAL-ONLY rule — it needs no video-side signal — and it is
exactly the shape of the old (wrong) belief that "anti-correlated IS the
deepfake signature". So it WILL be proposed.

MEASURED RESULT (2026-10-22), on the operator's real WhatsApp traffic:
    the 2 genuine av=0.5 clips BOTH have high HAVIC
      c1dfce01  corr=-0.299  havic=0.983
      b10342bc  corr=-0.443  havic=0.980
    => the pair fires on 2 of 63 genuine clips (3% FALSE POSITIVES)
    and it catches 0 of the 3 operator AI videos, because their av reading
    is 0.0 (unknown — silent audio), so the pair never fires for them.

=> THE PAIR RULE COSTS GENUINE FALSE POSITIVES AND BUYS NOTHING.
   Rejected on measurement.

The figures ABOVE are the measurement this tool made on 2026-10-22; the
figures the tool PRINTS are re-derived on every run (the denominators are
counted, not hardcoded), so a reader can always tell which is which.
This tool re-measures it so the rejection can be re-checked rather than
taken on trust.

Usage: python tools/measure_crossmodal_pair_rule.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vishwas.capabilities.cross_modal import CrossModalCapability  # noqa: E402
from vishwas.events import Artifact, JobContext  # noqa: E402

TRAFFIC = pathlib.Path("/home/hermes/.hermes/cache/scratch/av_real_traffic.jsonl")
Q = pathlib.Path("/home/hermes/.hermes/cache/scratch/pairq")
AI = REPO.parent / "fusion_av" / "seeded"
HAVIC_T = 0.9
AV_T = 0.5


def _tree_stamp() -> str:
    """Stamp the commit this run is against.

    2026-10-22. The PR #18 lesson: a measurement with no provenance is
    indistinguishable from a stale run. This tool is the re-measurement of
    a REJECTED rule, so a reader must be able to tell WHICH tree produced
    the numbers that justify the rejection.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO), capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if not sha:
            return "vishwas tree: UNKNOWN (no git info)"
        d = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO), capture_output=True, text=True, timeout=10)
        return f"vishwas tree: {sha}" + (
            " (DIRTY WORKTREE)" if d.stdout.strip() else "")
    except Exception:
        return "vishwas tree: UNKNOWN (git not available)"


def _signals(p: pathlib.Path, tag: str):
    """Return (av_risk_addition, havic) for one clip, via the capability."""
    q = Q / f"{tag}_{p.stem[:8]}"
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="video",
                   size_bytes=p.stat().st_size)
    ctx = JobContext(job_id="h", artifact=art, quarantine_root=q,
                      wall_budget_s=300.0, model_weights_available=True)
    av = hav = None
    for c in CrossModalCapability().analyze(art, ctx):
        if c.name == "cross_modal_av" and c.status == "ok":
            av = c.signals.get("av_risk_addition")
        elif c.name == "havic_crossmodal_model" and c.status == "ok":
            hav = c.signals.get("prob_inconsistent")
    return av, hav


def _has_audio(p: pathlib.Path) -> bool:
    """True if the clip carries an audio STREAM.

    The false-positive rate is quoted against the clips that HAVE audio
    (63 of the 67 in the traffic set) — a clip with no audio track never
    reaches the av=0.5 band at all, so including it in the denominator
    would understate the rate.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip())
    except Exception:
        return False


def main() -> int:
    print(f"  {_tree_stamp()}")
    if not TRAFFIC.exists():
        print(f"cache missing: {TRAFFIC}")
        print("regenerate with tools/measure_av_sync_real_traffic.py")
        return 1
    rows = [json.loads(l) for l in TRAFFIC.read_text().splitlines() if l.strip()]
    band = [r for r in rows if r.get("av") == AV_T]

    # The denominator: the genuine clips that HAVE an audio stream, because
    # only those can reach the band at all. Counted, never hardcoded.
    with_audio = sum(1 for r in rows
                     if pathlib.Path(r["path"]).exists()
                     and _has_audio(pathlib.Path(r["path"])))
    print(f"genuine traffic rows: {len(rows)}; "
          f"with an audio stream: {with_audio}; "
          f"in the av={AV_T} band: {len(band)}")
    print()

    pair_hits = 0
    print("=== the pair rule on the genuine clips in the av band ===")
    for r in band:
        p = pathlib.Path(r["path"])
        if not p.exists():
            continue
        av, hav = _signals(p, "band")
        fires = av == AV_T and hav is not None and hav >= HAVIC_T
        pair_hits += 1 if fires else 0
        note = "PAIR FIRES (GENUINE FALSE POSITIVE)" if fires else "no fire"
        print(f"  {r['name'][:12]}  corr={r.get('corr'):+.3f}  av={av}  "
              f"havic={hav:.3f}  -> {note}")
    print(f"  => the pair rule flags {pair_hits} of {len(band)} GENUINE clips "
          f"in the band")
    print()

    ai_hits = 0
    ai_clips = sorted(AI.glob("ai_*.mp4")) if AI.exists() else []
    if ai_clips:
        print("=== the pair rule on the operator's AI videos ===")
        for p in ai_clips:
            av, hav = _signals(p, "ai")
            fires = av == AV_T and hav is not None and hav >= HAVIC_T
            ai_hits += 1 if fires else 0
            print(f"  {p.name:<18} av={av}  havic={hav:.3f}  "
                  f"-> {'PAIR FIRES' if fires else 'NO FIRE'}")
        # The denominator is DERIVED, never hardcoded (PR #17): if a clip
        # is added to or removed from the anchor dir, the printed fraction
        # must follow it, or the summary states a number the run did not
        # produce.
        print(f"  => the pair rule catches {ai_hits} of {len(ai_clips)} AI videos")
    print()
    print("=" * 78)
    print(f"  THE PAIR RULE: {pair_hits} GENUINE clips flagged, "
          f"{ai_hits} AI videos caught.")
    if pair_hits and not ai_hits:
        rate = (100.0 * pair_hits / with_audio) if with_audio else 0.0
        print()
        print(f"  => IT COSTS {rate:.0f}% GENUINE FALSE POSITIVES "
              f"({pair_hits} of {with_audio} clips with audio) AND BUYS "
              "NOTHING. Rejected.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
