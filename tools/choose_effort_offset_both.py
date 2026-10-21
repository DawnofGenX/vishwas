#!/usr/bin/env python3
"""Choose the EFFORT offset so that it works for BOTH audio-bearing and
SILENT videos.

THE SILENT-VIDEO SENSITIVITY (2026-10-21)
------------------------------------------
`_extract` distinguishes three states:
    known_gap  the check was skipped/unavailable BY DESIGN (a gate)
    absent     no such check produced at all
    failed     the check ran and failed

The gap-neutral denominator only counts `known_gap` toward `total_w`. An
`absent` signal is excluded from the denominator entirely, so its weight
does not dilute the others.

For a SILENT video the cross-modal checks are `skipped` ("no paired
audio+video streams") -> that is `known_gap` -> they DO count toward
`total_w` at zero contribution. So a silent video's score is computed over
the FULL weight pool with the cross-modal terms contributing 0, which is
the honest "we could not check" posture.

But it means the effort term carries a much larger SHARE of a silent
video's score than of an audio-bearing one. With the new -0.30 offset a
silent real video with raw effort 0.637 (which is what the operator's own
phone video measures) lands at:

    effort_cal 0.337, frameheur_cal 0.105
    s = 1.5*0.337 + 1.0*0.105 = 0.6105, total_w = 4.8 (incl. the gaps)
    x = 0.1272 -> raw 0.602

...which is CAUTION, not DNU -- but only just. The point of this tool is
to SWEEP the offset over BOTH populations and find the value that keeps
the operator's real media on the safe side of the line in BOTH regimes.

Usage: python tools/choose_effort_offset_both.py
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult as C
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE

# The operator's own media, scored with the FIXED model (raw effort, and
# the cross-modal av value each clip actually produced).
OPERATOR = [
    # (kind, raw effort, av, alignment)
    ("REAL", 0.5172, 0.10, "weakly_synced"),
    ("REAL", 0.4578, 0.10, "weakly_synced"),
    ("REAL", 0.6332, 0.00, "synced"),
    ("REAL", 0.4380, 0.10, "weakly_synced"),
    ("AI", 0.3641, 0.50, "anti_correlated"),
    ("AI", 0.3715, 0.50, "anti_correlated"),
    ("AI", 0.3440, 0.50, "anti_correlated"),
]
# A SILENT real video: the same measured raw effort, but with no audio
# track, so the cross-modal checks are a known_gap.
SILENT_REAL = [("REAL-silent", 0.6332, None, None),
               ("REAL-silent", 0.5172, None, None),
               ("REAL-silent", 0.4578, None, None),
               ("REAL-silent", 0.4380, None, None)]


def checks_for(kind, eff, av, align, with_audio):
    out = [
        C("video_probe", "cheap", "ok", {"duration_s": 1.0}, "m"),
        C("usable_frames", "cheap", "ok", {"usable_frame_ratio": 1.0}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": 0.10}, "m"),
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": eff}, "m"),
    ]
    if with_audio:
        out.append(C("cross_modal_av", "heavy", "ok",
                     {"av_risk_addition": av, "alignment_class": align,
                      "av_correlation": 0.4,
                      "av_synced_clean": 0.4 if av == 0.0 else 0.0}, "m"))
        out.append(C("havic_crossmodal_model", "heavy", "ok",
                     {"prob_inconsistent": 0.99}, "m"))
    else:
        # A silent video: the cross-modal checks are SKIPPED -> known_gap.
        out.append(C("cross_modal_av", "heavy", "skipped", {},
                     "no paired audio+video streams"))
        out.append(C("havic_crossmodal_model", "heavy", "skipped", {},
                     "no audio track"))
    return out


def evaluate(offset):
    bw, bc = dict(WEIGHTS["deepfake_video"]), dict(_CALIBRATE)
    WEIGHTS["deepfake_video"] = {**bw, "effort.prob": 1.5}
    _CALIBRATE["effort.prob"] = (1.0, offset)
    try:
        fe = FusionEngine()
        res = []
        for kind, eff, av, align in OPERATOR:
            d = fe.decide("deepfake_video", checks_for(kind, eff, av, align, True))
            res.append((kind, d.verdict.value))
        for kind, eff, av, align in SILENT_REAL:
            d = fe.decide("deepfake_video", checks_for(kind, eff, av, align, False))
            res.append((kind, d.verdict.value))
    finally:
        WEIGHTS["deepfake_video"] = bw
        _CALIBRATE.clear()
        _CALIBRATE.update(bc)
    return res


print("=" * 100)
print("OFFSET SWEEP: the operator's media (audio-bearing) AND the same media as SILENT")
print("=" * 100)
print(f"{'offset':>8} | {'REAL audio (4)':<22} | {'AI (3)':<24} | {'REAL silent (4)':<22}")
print(f"{'':>8} | {'':.<22} | {'':.<24} | {'':.<22}")
for off in (-0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10):
    res = evaluate(off)
    ra = [v for k, v in res if k == "REAL"]
    ai = [v for k, v in res if k == "AI"]
    rs = [v for k, v in res if k == "REAL-silent"]
    ra_dnu = sum(1 for v in ra if v == "do_not_use")
    ai_dnu = sum(1 for v in ai if v == "do_not_use")
    rs_dnu = sum(1 for v in rs if v == "do_not_use")
    ok = "  <-- ALL SAFE" if (ra_dnu == 0 and rs_dnu == 0 and ai_dnu == 3) else ""
    print(f"{off:>8.2f} | {str(Counter(ra)):<22} | {str(Counter(ai)):<24} | "
          f"{str(Counter(rs)):<22}{ok}")
