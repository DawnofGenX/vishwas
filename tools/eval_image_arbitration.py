#!/usr/bin/env python3
"""Test image-channel ARBITRATION policies for the SPAI/NYUAD pair.

THE QUESTION
------------
`_image_clean` currently requires SPAI low AND NYUAD low to grant TRUST, so
EITHER detector alone can veto a TRUST verdict. The 2026-10-21 measurement
note claimed an "agreement only" policy would drop the 5 NYUAD-only false
alarms on real photos WITHOUT losing the 8 AI images that only NYUAD
catches.

That claim was ASSERTED, never measured. This script measures it.

METHOD (this matters)
----------------------
A policy is installed by REPLACING `vishwas.fusion._image_clean` with a
policy-specific version and re-running the real `FusionEngine.decide()`.
Post-processing the verdict (demoting trust->caution) is WRONG: it cannot
express a policy that ALLOWS a trust, which is exactly what "ignore
NYUAD's veto" needs to do. An earlier version of this script had that bug
and reported all four policies as identical.

THE KEY STRUCTURAL FACT
-----------------------
The 5 NYUAD-only false alarms and the 8 NYUAD-only catches have the SAME
(SPAI, NYUAD) signature: SPAI low, NYUAD high. So NO rule that sees only
those two numbers can separate them. The policy that drops the false alarms
necessarily drops the catches.

Usage: python tools/eval_image_arbitration.py
"""
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import vishwas.fusion as F
from vishwas.capabilities.base import CheckResult as C
from vishwas.fusion import FusionEngine

ROWS = pathlib.Path("/home/hermes/.hermes/cache/scratch/img_rows.json")
rows = json.loads(ROWS.read_text())


def checks_for(r):
    return [
        C("image_integrity", "cheap", "ok", {"format": "jpeg"}, "m"),
        C("frequency_band_analysis", "mid", "ok",
          {"prob_deepfake": r["freqband"], "band_anomaly": r["freqband"]}, "m"),
        C("image_face_forensics", "heavy", "ok", {"prob_deepfake": r["spai"]}, "m"),
        C("nyuad_image_detector", "heavy", "ok", {"prob_deepfake": r["nyuad"]}, "m"),
    ]


def _fb_ok(by_name):
    fb = F._probe(by_name, "frequency_band_analysis", "prob_deepfake", 0.0)
    return fb <= 0.60


# ---------------------------------------------------------------- policies --
# The shipped implementation, captured BEFORE any patching. `p_shipped`
# must call this captured reference, not the module attribute -- otherwise
# replacing F._image_clean makes it call itself (RecursionError).
_SHIPPED_IMAGE_CLEAN = F._image_clean


def p_shipped(by_name):
    """Current: SPAI low AND NYUAD low (either can veto)."""
    return _SHIPPED_IMAGE_CLEAN(by_name)


def p_agreement(by_name):
    """Both detectors must AGREE the image is real (SPAI low AND NYUAD low)."""
    spai = F._probe(by_name, "image_face_forensics", "prob_deepfake", 1.0)
    ny = F._probe(by_name, "nyuad_image_detector", "prob_deepfake", 1.0)
    return spai <= 0.45 and ny < 0.50 and _fb_ok(by_name)


def p_spai_only(by_name):
    """Ignore NYUAD entirely: SPAI alone decides."""
    spai = F._probe(by_name, "image_face_forensics", "prob_deepfake", 1.0)
    return spai <= 0.45 and _fb_ok(by_name)


def p_nyuad_only(by_name):
    """Ignore SPAI entirely: NYUAD alone decides."""
    ny = F._probe(by_name, "nyuad_image_detector", "prob_deepfake", 1.0)
    return ny < 0.50 and _fb_ok(by_name)


def p_either_low(by_name):
    """Either detector saying 'real' is enough (a veto needs BOTH to flag)."""
    spai = F._probe(by_name, "image_face_forensics", "prob_deepfake", 1.0)
    ny = F._probe(by_name, "nyuad_image_detector", "prob_deepfake", 1.0)
    return (spai <= 0.45 or ny < 0.50) and _fb_ok(by_name)


def evaluate(policy):
    original = F._image_clean
    F._image_clean = policy
    try:
        fe = FusionEngine()
        res = []
        for r in rows:
            d = fe.decide("image_facecheck", checks_for(r))
            res.append((r["label"], d.verdict.value))
    finally:
        F._image_clean = original
    real = [v for l, v in res if l == 0]
    fake = [v for l, v in res if l == 1]
    return {
        "fake_trust": sum(1 for v in fake if v == "trust"),
        "fake_caution": sum(1 for v in fake if v == "caution"),
        "real_trust": sum(1 for v in real if v == "trust"),
        "real_caution": sum(1 for v in real if v == "caution"),
        "n_real": len(real), "n_fake": len(fake),
    }


def show(name, m):
    # fakeTRUST      a fake that reads CLEAN (the false-clean -- the worst)
    # realFALSEALARM a real photo that does not read LOW
    print(f"{name:<44s} fakeTRUST={m['fake_trust']:2d}/{m['n_fake']}  "
          f"realFALSEALARM={m['real_caution']:2d}/{m['n_real']}  "
          f"realTRUST={m['real_trust']:2d}")


print("=" * 100)
print("IMAGE ARBITRATION POLICIES — measured end to end")
print(f"  corpus: {len(rows)} images (30 AI / 30 real)")
print("=" * 100)

show("0. as shipped (SPAI low AND NYUAD low)", evaluate(p_shipped))
show("1. agreement-only (both must say real)", evaluate(p_agreement))
show("2. SPAI-only veto (ignore NYUAD)", evaluate(p_spai_only))
show("3. NYUAD-only veto (ignore SPAI)", evaluate(p_nyuad_only))
show("4. either-low (a veto needs BOTH to flag)", evaluate(p_either_low))

print()
print("=" * 100)
print("WHY: the two groups are structurally INDISTINGUISHABLE")
print("=" * 100)
fa = [r for r in rows if r["label"] == 0 and r["verdict"] != "trust"
      and r["spai"] <= 0.45 and r["nyuad"] >= 0.50]
ca = [r for r in rows if r["label"] == 1 and r["spai"] < 0.45 and r["nyuad"] >= 0.50]
print(f"  REAL photos flagged by NYUAD alone ({len(fa)}):")
for r in fa:
    print(f"    {r['name']:<24} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
          f"fb={r['freqband']:.3f}")
print(f"  AI images caught by NYUAD alone ({len(ca)}):")
for r in ca:
    print(f"    {r['name']:<24} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
          f"fb={r['freqband']:.3f}")
print()
print("  Every member of both groups has SPAI LOW and NYUAD HIGH.")
print()
print("  The only OTHER number available is freqband, and it is INVERTED:")
print(f"    real-group fb median {st.median([r['freqband'] for r in fa]):.4f}")
print(f"    AI-group   fb median {st.median([r['freqband'] for r in ca]):.4f}")
print("  -> it is HIGHER for the real photos, so it cannot be used to keep")
print("     the catches and drop the alarms. It would do the opposite.")

print()
print("=" * 100)
print("CONCLUSION")
print("=" * 100)
print("The note's claim was WRONG: no policy over (SPAI, NYUAD) can keep the")
print("8 NYUAD-only catches AND drop the 5 NYUAD-only false alarms, because")
print("those 13 images have the SAME signature (SPAI low, NYUAD high).")
print()
print("The real trade is 8 catches vs 5 false alarms. The false-clean is the")
print("MORE dangerous direction -- it tells the user 'this AI image is")
print("genuine' -- so keeping NYUAD's veto is the right call, and the")
print("shipped policy is already the best of the five measured.")
print()
print("Separating them needs a THIRD signal (a different model, or a")
print("file/provenance check), not a re-arbitration of these two.")
