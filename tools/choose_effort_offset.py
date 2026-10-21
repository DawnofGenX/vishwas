#!/usr/bin/env python3
"""Choose the EFFORT offset using the FULL corpus (each row's real
cross-modal signals), through the REAL FusionEngine.

WHY A SEPARATE TOOL
-------------------
`tools/derive_effort_calibration.py` sweeps the offset with SYNTHETIC
cross-modal signals (av=0.0, havic=0.5) to isolate the effort term. That
is the right instrument for reading the offset's effect, but it is NOT a
corpus evaluation: the real rows carry their own av/havic, and the
corroboration gate and the pattern layer both depend on them. The offset
must be chosen on the real signals, not on a stand-in.

This tool therefore joins the freshly-scored effort values with the
MEASURED cross-modal values from the leak-free corpus
(video_signal_rows_v2.jsonl) and evaluates each candidate offset through
the real FusionEngine on the joined rows.

Usage: python tools/choose_effort_offset.py
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult as C
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE

EFFORT_ROWS = pathlib.Path("/home/hermes/.hermes/cache/scratch/effort_cal_rows.jsonl")
XM_ROWS = pathlib.Path("/home/hermes/.hermes/cache/scratch/video_signal_rows_v2.jsonl")

eff = {}
for line in EFFORT_ROWS.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        eff[r["path"]] = r
xm = {}
for line in XM_ROWS.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        xm[r["path"]] = r

joined = []
for path, e in eff.items():
    x = xm.get(path)
    if not x:
        continue
    joined.append({
        "path": path, "label": e["label"],
        "effort": e["effort"],
        "frameheur": x["frameheur"], "av": x["av"], "havic": x["havic"],
    })
print(f"joined {len(joined)} rows (effort after the fix x measured cross-modal)")
reals = [r for r in joined if r["label"] == 0]
fakes = [r for r in joined if r["label"] == 1]
print(f"  real {len(reals)} / fake {len(fakes)}")


def checks_for(r):
    return [
        C("video_probe", "cheap", "ok", {"duration_s": 1.0}, "m"),
        C("usable_frames", "cheap", "ok", {"usable_frame_ratio": 1.0}, "m"),
        C("frame_heuristics", "mid", "ok", {"prob_deepfake": r["frameheur"]}, "m"),
        C("effort_face_forensics", "heavy", "ok", {"prob_deepfake": r["effort"]}, "m"),
        C("cross_modal_av", "heavy", "ok",
          {"av_risk_addition": r["av"],
           "alignment_class": "anti_correlated" if r["av"] >= 0.45 else "synced",
           "av_correlation": 0.4, "av_synced_clean": 0.4 if r["av"] == 0.0 else 0.0}, "m"),
        C("havic_crossmodal_model", "heavy", "ok",
          {"prob_inconsistent": r["havic"]}, "m"),
    ]


def evaluate(offset, weight=1.5):
    base_w = dict(WEIGHTS["deepfake_video"])
    base_c = dict(_CALIBRATE)
    WEIGHTS["deepfake_video"] = {**base_w, "effort.prob": weight}
    _CALIBRATE["effort.prob"] = (1.0, offset)
    try:
        fe = FusionEngine()
        res = [(r["label"], fe.decide("deepfake_video", checks_for(r)).verdict.value)
               for r in joined]
    finally:
        WEIGHTS["deepfake_video"] = base_w
        _CALIBRATE.clear()
        _CALIBRATE.update(base_c)
    real = [v for l, v in res if l == 0]
    fake = [v for l, v in res if l == 1]
    return {
        "fp": sum(1 for v in real if v == "do_not_use"),
        "fn": sum(1 for v in fake if v == "trust"),
        "dnu_fake": sum(1 for v in fake if v == "do_not_use"),
        "trust_real": sum(1 for v in real if v == "trust"),
        "n_real": len(real), "n_fake": len(fake),
        "real": Counter(real), "fake": Counter(fake),
    }


print()
print("=" * 96)
print("OFFSET SWEEP on the FULL joined corpus (real av/havic per row)")
print("=" * 96)
print(f"{'offset':>8} {'FP':>4} {'FN':>4} {'fakeDNU':>9} {'realTRUST':>10}   verdict")
best = []
for off in (-0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10,
            -0.05, 0.0, 0.05):
    m = evaluate(off)
    note = ""
    if m["fp"] == 0 and m["fn"] == 0:
        note = "  <-- no FP and no FN"
        best.append((off, m))
    print(f"{off:>8.2f} {m['fp']:>4} {m['fn']:>4} {m['dnu_fake']:>9} "
          f"{m['trust_real']:>10}   {note}")

print()
print("=== the candidates that are clean on BOTH axes ===")
for off, m in best:
    print(f"  offset {off:+.2f}: fakeDNU={m['dnu_fake']} realTRUST={m['trust_real']} "
          f"| real={dict(m['real'])} fake={dict(m['fake'])}")
if not best:
    print("  none")

print()
print("=== the CURRENT shipping offset (-0.55) for comparison ===")
m = evaluate(-0.55)
print(f"  FP={m['fp']} FN={m['fn']} fakeDNU={m['dnu_fake']} realTRUST={m['trust_real']}")
print(f"  real={dict(m['real'])}")
print(f"  fake={dict(m['fake'])}")
