#!/usr/bin/env python3
"""Measure a video-verdict RULE against three corpora, including the
operator's own AI anchor videos.

CORPORA
-------
1. leak-free synthetic (60 rows) - 30 real / 30 fake, production code paths,
   audio label independent of video label.
2. operator natural-audio reals (4 rows) - genuine media, all REAL.
3. operator AI anchors (seeded/*.mp4, 3 AI videos) - the operator's own
   generated deepfakes. These MUST reach DO_NOT_USE: a rule that never fires
   is as useless as one that always fires.

The rule under test: DO_NOT_USE requires EFFORT (the only video signal with
measured separation) to be elevated AND at least one INDEPENDENT cross-modal
family (av_risk decorrelated, or havic) to be elevated. A lone EFFORT reading
caps at CAUTION — the same posture the image channel already takes, and the
same lesson the 2026-08-26 chameleon overfit taught.

Usage: python tools/eval_video_rule.py
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine, _CALIBRATE

V2 = pathlib.Path("/home/hermes/.hermes/cache/scratch/video_signal_rows_v2.jsonl")
OP = pathlib.Path("/home/hermes/.hermes/cache/scratch/operator_reals_signals.jsonl")
ANCHOR = pathlib.Path("/home/hermes/.hermes/cache/scratch/operator_ai_anchors_signals.jsonl")

v2 = [json.loads(l) for l in V2.read_text().splitlines() if l.strip()] if V2.exists() else []
op = [json.loads(l) for l in OP.read_text().splitlines() if l.strip()] if OP.exists() else []
an = [json.loads(l) for l in ANCHOR.read_text().splitlines() if l.strip()] if ANCHOR.exists() else []


def cal(key, v):
    a, b = _CALIBRATE.get(key, (1.0, 0.0))
    return min(1.0, max(0.0, a * v + b))


def decide(r):
    fe = FusionEngine()
    checks = [
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": r["effort"]}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": r["frameheur"]}, "m"),
        C("cross_modal_av", "heavy", "ok",
          {"av_risk_addition": r["av"],
           "alignment_class": r.get("av_alignment", "weakly_synced"),
           "av_correlation": r.get("av_correlation", 0.0),
           "av_synced_clean": 0.0}, "m"),
        C("havic_crossmodal_model", "heavy", "ok",
          {"prob_inconsistent": r["havic"]}, "m"),
    ]
    return fe.decide("deepfake_video", checks)


def summary(rows):
    out = []
    for r in rows:
        d = decide(r)
        eff_c = cal("effort.prob", r["effort"])
        av_c = cal("av_risk_addition", r["av"])
        hav_c = cal("havic.prob_inconsistent", r["havic"])
        out.append({
            "name": r["path"].split("/")[-1], "label": r["label"],
            "verdict": d.verdict.value, "score": round(float(d.score), 4),
            "eff_c": eff_c, "av_c": av_c, "hav_c": hav_c,
            "eff_raw": r["effort"], "av_raw": r["av"], "hav_raw": r["havic"],
        })
    return out


if __name__ == "__main__":
    print(f"leak-free corpus: {len(v2)}  operator reals: {len(op)}  AI anchors: {len(an)}")
    for name, rows in (("LEAK-FREE", v2), ("OPERATOR REALS", op), ("AI ANCHORS", an)):
        if not rows:
            continue
        s = summary(rows)
        print(f"\n=== {name} ===")
        by = Counter(x["verdict"] for x in s)
        print(f"  verdicts: {dict(by)}")
        if name == "LEAK-FREE":
            for lab in (0, 1):
                sub = [x for x in s if x["label"] == lab]
                fp = sum(1 for x in sub if x["verdict"] == "do_not_use")
                fn = sum(1 for x in sub if x["verdict"] == "trust")
                dnu = sum(1 for x in sub if x["verdict"] == "do_not_use")
                print(f"  label={lab} n={len(sub)}: FP(DNU on real)={fp}  FN(TRUST on fake)={fn}  DNU={dnu}")
        else:
            for x in s:
                print(f"  {x['name'][:34]:<36s} eff_raw={x['eff_raw']:.3f} "
                      f"eff_cal={x['eff_c']:.3f} av={x['av_raw']} hav={x['hav_raw']:.3f} "
                      f"-> {x['verdict']} ({x['score']:.3f})")
        if name == "LEAK-FREE":
            print("  --- the DNU rows ---")
            for x in s:
                if x["verdict"] == "do_not_use":
                    print(f"    {x['name'][:20]:<22s} lab={x['label']} eff_cal={x['eff_c']:.3f} "
                          f"av={x['av_raw']} hav_cal={x['hav_c']:.3f} score={x['score']:.3f}")
