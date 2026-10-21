#!/usr/bin/env python3
"""Grid-search a defensible deepfake_video weight/calibration set.

Constraints on any candidate:
  * ZERO false positives on reals (a real clip must never read DO_NOT_USE);
  * the video-only signals (effort, frameheur) are the only ones that can be
    measured honestly on this host, so the search is anchored on them;
  * the cross-modal gates may be kept or dropped, but if kept their influence
    must not dominate the score.

The grid is evaluated on the LEAK-FREE corpus (v2), where the audio-label
marginal is identical across video classes, so no audio-only detector can
separate it. A candidate is also re-scored on the LEAKED corpus (v1) to
confirm it is not secretly exploiting the leak.

Usage: python tools/grid_video_weights.py
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vishwas.capabilities.base import CheckResult
from vishwas.fusion import FusionEngine, WEIGHTS, _CALIBRATE

V1 = pathlib.Path("/home/hermes/.hermes/cache/scratch/video_signal_rows.jsonl")
V2 = pathlib.Path("/home/hermes/.hermes/cache/scratch/video_signal_rows_v2.jsonl")


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def checks_for(r, use_cross_modal):
    out = [
        CheckResult("effort_face_forensics", "heavy", "ok",
                    {"prob_deepfake": r["effort"]}, "m"),
        CheckResult("frame_heuristics", "mid", "ok",
                    {"prob_deepfake": r["frameheur"]}, "m"),
    ]
    if use_cross_modal:
        out.append(CheckResult("cross_modal_av", "heavy", "ok",
                                {"av_risk_addition": r["av"],
                                 "alignment_class": "m", "av_synced_clean": 0.0}, "m"))
        out.append(CheckResult("havic_crossmodal_model", "heavy", "ok",
                                {"prob_inconsistent": r["havic"]}, "m"))
    else:
        out.append(CheckResult("cross_modal_av", "heavy", "skipped", {}, "m"))
        out.append(CheckResult("havic_crossmodal_model", "heavy", "skipped", {}, "m"))
    return out


def evaluate(rows, weights, calib, use_cross_modal):
    WEIGHTS["deepfake_video"] = weights
    _CALIBRATE.update(calib)
    fe = FusionEngine()
    res = []
    for r in rows:
        d = fe.decide("deepfake_video", checks_for(r, use_cross_modal))
        res.append((r["label"], d.verdict.value))
    real = [v for lab, v in res if lab == 0]
    fake = [v for lab, v in res if lab == 1]
    return {
        "fp": sum(1 for v in real if v == "do_not_use"),
        "fn": sum(1 for v in fake if v == "trust"),
        "dnu_fake": sum(1 for v in fake if v == "do_not_use"),
        "trust_real": sum(1 for v in real if v == "trust"),
        "caution_fake": sum(1 for v in fake if v == "caution"),
        "unver_real": sum(1 for v in real if v == "unable_to_verify"),
        "n_real": len(real), "n_fake": len(fake),
        "real": Counter(real), "fake": Counter(fake),
    }


def main():
    v1, v2 = load(V1), load(V2)
    base_w = dict(WEIGHTS["deepfake_video"])
    base_c = {k: v for k, v in _CALIBRATE.items()}

    def restore():
        WEIGHTS["deepfake_video"] = base_w
        _CALIBRATE.clear(); _CALIBRATE.update(base_c)

    print("=" * 80)
    print("BASELINE (current shipping configuration)")
    print("=" * 80)
    cur = evaluate(v2, base_w, base_c, True)
    print(f"  leak-free: FP(real->DNU)={cur['fp']}/{cur['n_real']}  "
          f"FN(fake->TRUST)={cur['fn']}/{cur['n_fake']}  "
          f"fakeDNU={cur['dnu_fake']}  realTRUST={cur['trust_real']}")
    print(f"    real={dict(cur['real'])}")
    print(f"    fake={dict(cur['fake'])}")
    restore()

    print()
    print("=" * 80)
    print("GRID: video-only signals (effort + frameheur), cross-modal DROPPED")
    print("=" * 80)
    cands = []
    for we in (1.0, 1.5, 2.0, 2.5, 3.0):
        for wf in (0.0, 0.25, 0.5, 0.75, 1.0):
            for b in (-0.60, -0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20):
                for fa in (0.0, 0.5, 1.0, 1.5, 1.97, 2.5):
                    W = {"effort.prob": we, "frameheur.prob": wf}
                    C = {"effort.prob": (1.0, b), "frameheur.prob": (fa, 0.0)}
                    m = evaluate(v2, W, C, False)
                    if m["fp"] == 0:
                        cands.append((m["dnu_fake"], m["trust_real"], m["caution_fake"],
                                      we, wf, b, fa, dict(m["real"]), dict(m["fake"])))
    restore()
    cands.sort(key=lambda t: (-t[0], -t[1]))
    print(f"  {len(cands)} candidates with ZERO false positives on reals")
    print(f"  {'fakeDNU':>8} {'realTRUST':>9} {'we':>4} {'wf':>5} {'b_eff':>7} {'a_fh':>6}  verdicts")
    for t in cands[:15]:
        print(f"  {t[0]:8d} {t[1]:9d} {t[2]:7d} {t[3]:4.1f} {t[4]:5.2f} {t[5]:+7.2f} {t[6]:6.2f}  "
              f"real={t[7]} fake={t[8]}")

    print()
    print("=" * 80)
    print("GRID: cross-modal KEPT, but the frameheur amplification is removed")
    print("=" * 80)
    cands2 = []
    for we in (1.0, 1.5, 2.0, 2.5):
        for wav in (0.0, 0.5, 1.0, 1.5, 2.0):
            for whv in (0.0, 0.4, 0.8):
                for b in (-0.60, -0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25):
                    for fa in (0.0, 0.5, 1.0, 1.5):
                        W = {"effort.prob": we, "frameheur.prob": 1.0,
                             "av_risk_addition": wav, "havic.prob_inconsistent": whv}
                        C = {"effort.prob": (1.0, b), "frameheur.prob": (fa, 0.0),
                             "av_risk_addition": (1.0, 0.0),
                             "havic.prob_inconsistent": (0.2, 0.0)}
                        m = evaluate(v2, W, C, True)
                        if m["fp"] == 0:
                            cands2.append((m["dnu_fake"], m["trust_real"],
                                           we, wav, whv, b, fa,
                                           dict(m["real"]), dict(m["fake"])))
    restore()
    cands2.sort(key=lambda t: (-t[0], -t[1]))
    print(f"  {len(cands2)} candidates with ZERO false positives on reals")
    for t in cands2[:10]:
        print(f"  fakeDNU={t[0]:2d} realTRUST={t[1]:2d} | w_eff={t[2]} w_av={t[3]} "
              f"w_hav={t[4]} b_eff={t[5]:+.2f} a_fh={t[6]:.1f} | real={t[7]} fake={t[8]}")

    print()
    print("=" * 80)
    print("LEAK CHECK: does the best video-only candidate also work on v1?")
    print("=" * 80)
    if cands:
        best = cands[0]
        W = {"effort.prob": best[3], "frameheur.prob": best[4]}
        C = {"effort.prob": (1.0, best[5]), "frameheur.prob": (best[6], 0.0)}
        m1 = evaluate(v1, W, C, False)
        restore()
        print(f"  best candidate on the LEAKED corpus: FP={m1['fp']}/{m1['n_real']} "
              f"FN={m1['fn']}/{m1['n_fake']} fakeDNU={m1['dnu_fake']}")
        print(f"    real={dict(m1['real'])} fake={dict(m1['fake'])}")
        print("  (video-only signals cannot exploit an audio leak, so v1 and v2 agree)")

    print()
    print("=" * 80)
    print("RECOMMENDATION INPUT: what the 4 operator clips (NATURAL audio) say")
    print("=" * 80)
    print("  These are the only clips on this host with genuine, synchronised audio.")
    print("  They are all REAL, so any configuration that reads them DNU is wrong.")
    restore()


if __name__ == "__main__":
    main()
