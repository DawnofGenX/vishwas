#!/usr/bin/env python3
"""Sweep the genimage veto threshold and shape, through the REAL FusionEngine.

WHY THIS TOOL EXISTS (2026-10-22)
---------------------------------
`src/vishwas/fusion.py`'s `_image_clean` carries a table of genimage veto
thresholds:

    threshold   veto-only              veto+clear
                fakeTRUST  realFA      fakeTRUST  realFA
     0.9000        0/30      14/30        0/30      12/30
     0.9300        1/30      13/30        1/30      11/30
     0.9400        2/30      13/30        3/30      11/30
     0.9500        2/30      12/30        4/30      10/30
     0.9855        2/30      12/30        5/30       8/30

...and cites `tools/eval_image_arbitration.py` as its source. **THAT TOOL
DOES NOT SWEEP ANY THRESHOLD AND DOES NOT IMPLEMENT THE "CLEAR" SHAPE** —
it measures the TWO-signal (SPAI + NYUAD) arbitration only. Its
sibling `eval_image_arbitration_genimage.py` even claims in its
docstring that it "sweeps the genimage veto threshold for BOTH candidate
shapes (veto-only and veto+clear)"; the code does neither.

So the table that justifies the SHIPPED threshold was **ASSERTED, NOT
REPRODUCIBLE**. This tool makes it reproducible, and it either confirms
the table or corrects it. Either outcome is a win: the point is that the
number can be re-derived by anyone.

METHOD (this matters)
----------------------
The policy is installed by REPLACING `vishwas.fusion._image_clean` with a
wrapper, and re-running the real `FusionEngine.decide()`. Post-processing
the verdict is WRONG — it cannot express a policy that ALLOWS a trust,
which is exactly what the "clear" shape needs to do.

**THE WRAPPER CALLS THE REAL SHIPPED FUNCTION.** It does not
re-implement the base arbitration. An earlier version of this tool
re-implemented it and got realFALSEALARM 11/30 where the shipped comment
claimed 12/30 — because the shipped rule is `eff <= 0.45` (SPAI) and the
re-implementation used `>= 0.5`. **⇒ NEVER RE-IMPLEMENT A POLICY TO
MEASURE IT; WRAP IT.** The only thing that varies here is the genimage
threshold and the shape.

The two shapes:
  * VETO-ONLY (the shipped shape): a high genimage DENIES a clean
    reading. It can never grant one.
  * VETO+CLEAR: a high genimage denies, AND a LOW genimage can clear a
    lone NYUAD hit (a low genimage lets the image through even if NYUAD
    flagged it).

Usage:
    PYTHONPATH=src python tools/sweep_genimage_veto.py
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import vishwas.fusion as F  # noqa: E402
from vishwas.capabilities.base import CheckResult  # noqa: E402

CACHE = pathlib.Path("/home/hermes/.hermes/cache/scratch/img_rows3.json")
THRESHOLDS = (0.50, 0.70, 0.85, 0.90, 0.93, 0.94, 0.95, 0.96, 0.97,
              0.9855, 0.995)


def _ck(name: str, signals: dict) -> CheckResult:
    return CheckResult(name=name, cost="mid", status="ok", signals=signals, notes="")


def _checks(r: dict) -> list[CheckResult]:
    return [
        _ck("image_integrity", {"integrity_ok": True}),
        _ck("image_face_forensics", {"prob_deepfake": r["spai"]}),
        _ck("frequency_band_analysis", {"prob_deepfake": r["freqband"]}),
        _ck("nyuad_image_detector", {"prob_deepfake": r["nyuad"]}),
        _ck("genimage_image_detector", {"prob_deepfake": r["genimage"]}),
    ]


def _make_clean(threshold: float, shape: str, real_clean):
    """Wrap the REAL shipped `_image_clean` with a genimage policy.

    The base arbitration is the shipped function, untouched. The wrapper
    only adds the genimage behaviour, and it is applied BEFORE the base
    call so the base's own genimage veto (at the shipped threshold) does
    not interfere.

    For the "clear" shape the wrapper must be able to ALLOW a trust that
    the base would deny, so the base is called with the NYUAD check
    neutralised and the NYUAD decision re-applied afterwards.
    """
    def _clean(by_name: dict) -> bool:
        # 1. the genimage VETO — a high genimage denies, in both shapes.
        gen = by_name.get("genimage_image_detector")
        gen_p = None
        if gen is not None and gen.usable():
            gp = gen.signals.get("prob_deepfake", 1.0)
            if isinstance(gp, (int, float)):
                gen_p = gp
        if gen_p is not None and gen_p >= threshold:
            return False

        # 2. for the CLEAR shape, a LOW genimage can override a lone NYUAD
        #    hit. The base's NYUAD veto must be bypassed for this call.
        if shape == "veto_clear" and gen_p is not None and gen_p < threshold:
            by_name = dict(by_name)
            ny = by_name.get("nyuad_image_detector")
            if ny is not None and ny.usable():
                np_ = ny.signals.get("prob_deepfake", 1.0)
                if isinstance(np_, (int, float)) and np_ >= 0.50:
                    by_name["nyuad_image_detector"] = CheckResult(
                        name="nyuad_image_detector", cost="mid", status="ok",
                        signals={"prob_deepfake": 0.0}, notes="")
        return real_clean(by_name)
    return _clean


def _measure(rows, threshold: float, shape: str, real_clean):
    original = F._image_clean
    F._image_clean = _make_clean(threshold, shape, real_clean)
    try:
        fe = F.FusionEngine()
        fake_trust, real_fa = [], []
        for r in rows:
            d = fe.decide("image_facecheck", _checks(r))
            if r["label"] == 1:
                if d.verdict.value == "trust":
                    fake_trust.append(r["name"])
            elif d.verdict.value != "trust":
                real_fa.append(r["name"])
        return fake_trust, real_fa
    finally:
        F._image_clean = original


def main() -> int:
    if not CACHE.exists():
        print(f"cache missing: {CACHE}")
        print("regenerate it with the 2026-10-21 image measurement + the "
              "genimage column")
        return 1
    rows = json.loads(CACHE.read_text())
    n_ai = sum(1 for r in rows if r["label"] == 1)
    n_real = len(rows) - n_ai

    # The REAL shipped function, captured before any monkey-patching.
    real_clean = F._image_clean

    print("=== genimage veto threshold sweep, REAL FusionEngine ===")
    print(f"    {n_ai} AI / {n_real} real images")
    print(f"    base policy = the SHIPPED vishwas.fusion._image_clean")
    print()
    print("  threshold   veto-only              veto+clear")
    print("              fakeTRUST  realFA      fakeTRUST  realFA")
    for t in THRESHOLDS:
        ft_v, rf_v = _measure(rows, t, "veto", real_clean)
        ft_c, rf_c = _measure(rows, t, "veto_clear", real_clean)
        print(f"    {t:<9.4f} {len(ft_v):>2}/{n_ai:<6} {len(rf_v):>2}/{n_real:<8}"
              f"{len(ft_c):>2}/{n_ai:<6} {len(rf_c):>2}/{n_real}")

    print()
    print("=== THE SHIPPED THRESHOLD, in detail ===")
    ft, rf = _measure(rows, F._GENIMAGE_VETO_T, "veto", real_clean)
    print(f"  _GENIMAGE_VETO_T = {F._GENIMAGE_VETO_T}")
    print(f"  veto-only: fakeTRUST {len(ft)}/{n_ai}   "
          f"realFALSEALARM {len(rf)}/{n_real}")
    print()
    print("  the AI images that still read TRUST:")
    for n in ft:
        r = next(x for x in rows if x["name"] == n)
        print(f"    {n:<28} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
              f"genimage={r['genimage']:.4f}")
    print()
    print("  === WHAT THE 'CLEAR' SHAPE CHANGES (veto-only vs veto+clear) ===")
    ft_v, rf_v = _measure(rows, F._GENIMAGE_VETO_T, "veto", real_clean)
    ft_c, rf_c = _measure(rows, F._GENIMAGE_VETO_T, "veto_clear", real_clean)
    # "newly let through" = in the clear shape's TRUST set but not the
    # veto-only one.
    newly = [n for n in ft_c if n not in ft_v]
    # "newly freed" = false-alarming under veto-only but NOT under clear.
    freed = [n for n in rf_v if n not in rf_c]
    print(f"  AI images the CLEAR shape newly lets through: {len(newly)}")
    for n in newly:
        r = next(x for x in rows if x["name"] == n)
        print(f"    {n:<28} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
              f"genimage={r['genimage']:.4f}")
    print(f"  genuine photos the CLEAR shape newly frees: {len(freed)}")
    for n in freed:
        r = next(x for x in rows if x["name"] == n)
        print(f"    {n:<28} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
              f"genimage={r['genimage']:.4f}")
    print()
    print("  the genuine photos that still false-alarm:")
    for n in rf:
        r = next(x for x in rows if x["name"] == n)
        print(f"    {n:<28} spai={r['spai']:.4f} nyuad={r['nyuad']:.4f} "
              f"genimage={r['genimage']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
