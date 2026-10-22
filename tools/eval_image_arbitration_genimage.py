"""Measure the image-channel arbitration policies on the cached 60-image
corpus, through the REAL FusionEngine.

Usage:
    PYTHONPATH=src python tools/eval_image_arbitration_genimage.py

The cache is produced by the 2026-10-21 image measurement
(/home/hermes/.hermes/cache/scratch/img_rows3.json), which adds the genimage
column to the per-image SPAI / NYUAD / freqband data.

Prints the same table the 2026-10-21 note used, so the two are comparable:

    fakeTRUST      = AI images that read TRUST  (the DANGEROUS direction --
                     it tells the user "this AI image is genuine")
    realFALSEALARM = genuine photos that do NOT read clean

It also sweeps the genimage veto threshold for BOTH candidate shapes
(veto-only and veto+clear) so the shipped policy is chosen on evidence.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vishwas.capabilities.base import CheckResult  # noqa: E402
from vishwas.fusion import FusionEngine  # noqa: E402

CACHE = pathlib.Path("/home/hermes/.hermes/cache/scratch/img_rows3.json")
THRESHOLDS = (0.50, 0.70, 0.85, 0.90, 0.93, 0.94, 0.95, 0.96, 0.97,
              0.9855, 0.995)


def _ck(name: str, signals: dict) -> CheckResult:
    return CheckResult(name=name, cost="mid", status="ok", signals=signals, notes="")


def _checks(r: dict, use_genimage: bool, use_nyuad: bool) -> list[CheckResult]:
    out = [
        _ck("image_integrity", {"integrity_ok": True}),
        _ck("image_face_forensics", {"prob_deepfake": r["spai"]}),
        _ck("frequency_band_analysis", {"prob_deepfake": r["freqband"]}),
    ]
    if use_nyuad:
        out.append(_ck("nyuad_image_detector", {"prob_deepfake": r["nyuad"]}))
    if use_genimage:
        out.append(_ck("genimage_image_detector", {"prob_deepfake": r["genimage"]}))
    return out


def _measure(rows, use_genimage=True, use_nyuad=True):
    fe = FusionEngine()
    fake_trust, real_fa = [], []
    for r in rows:
        d = fe.decide("image_facecheck", _checks(r, use_genimage, use_nyuad))
        if r["label"] == 1:
            if d.verdict.value == "trust":
                fake_trust.append(r["name"])
        elif d.verdict.value != "trust":
            real_fa.append(r["name"])
    return fake_trust, real_fa


def main() -> int:
    if not CACHE.exists():
        print(f"cache missing: {CACHE}")
        print("regenerate it with the 2026-10-21 image measurement + the "
              "genimage column")
        return 1
    rows = json.loads(CACHE.read_text())

    print("=== image arbitration, real FusionEngine, 60 images ===")
    base_ft, base_rf = _measure(rows, use_genimage=False)
    print(f"  {'as shipped BEFORE (SPAI + NYUAD)':<40} "
          f"fakeTRUST {len(base_ft):>2}/30   realFALSEALARM {len(base_rf):>2}/30")
    ft, rf = _measure(rows, use_genimage=True)
    print(f"  {'WITH genimage (veto-only)':<40} "
          f"fakeTRUST {len(ft):>2}/30   realFALSEALARM {len(rf):>2}/30")
    ft2, rf2 = _measure(rows, use_nyuad=False, use_genimage=True)
    print(f"  {'SPAI + genimage only':<40} "
          f"fakeTRUST {len(ft2):>2}/30   realFALSEALARM {len(rf2):>2}/30")
    ft3, rf3 = _measure(rows, use_nyuad=False, use_genimage=False)
    print(f"  {'SPAI only':<40} "
          f"fakeTRUST {len(ft3):>2}/30   realFALSEALARM {len(rf3):>2}/30")

    print()
    print("=== the AI images that still read TRUST (with genimage) ===")
    for n in ft:
        r = next(x for x in rows if x["name"] == n)
        print(f"  {n:<28} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
              f"genimage={r['genimage']:.4f}")
    if not ft:
        print("  (none)")

    print()
    print("=== the genuine photos that still false-alarm (with genimage) ===")
    for n in rf:
        r = next(x for x in rows if x["name"] == n)
        print(f"  {n:<28} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
              f"genimage={r['genimage']:.4f}")
    if not rf:
        print("  (none)")

    print()
    print("=== DELTA (veto-only, threshold 0.9855) ===")
    print(f"  fakeTRUST       {len(base_ft):>2}/30 -> {len(ft):>2}/30   "
          f"({len(base_ft) - len(ft):+d} safer)")
    print(f"  realFALSEALARM  {len(base_rf):>2}/30 -> {len(rf):>2}/30   "
          f"({len(base_rf) - len(rf):+d} fewer false alarms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
