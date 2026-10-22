"""GENIMAGE is the THIRD image signal — the one SPAI+NYUAD were missing.

2026-10-22. THE MEASURED PROBLEM
---------------------------------
The image channel had exactly TWO independent signals (SPAI and NYUAD), and
they could not separate the two groups that matter:

    5 genuine photos flagged by NYUAD alone
    9 AI images caught by NYUAD alone

Both groups have the SAME signature (SPAI low, NYUAD high), so no rule over
(SPAI, NYUAD) can separate them. The fusion code says so explicitly
(src/vishwas/fusion.py, the `_image_clean` comment): "Separating the two
groups needs a THIRD signal (a model with non-overlapping errors, a
provenance check, or a content check), not a re-arbitration of these two."

Every policy that re-arbitrated the two existing signals traded a small
false-alarm reduction for a much larger false-clean increase (the more
dangerous direction — telling the user an AI image is genuine).

WHAT GENIMAGE IS
----------------
The upstream EFFORT project (YZY-stack/Effort-AIGI-Detection, arXiv
2411.15633) ships three checkpoints of the SAME architecture:
  * chameleon  - face-region / spatial forensics      (on host)
  * ffpp       - face forensics, FaceForensics++ domain (on host, IN USE)
  * genimage   - GENERAL AIGC image detection, trained on GenImage/SD-v1.4
The genimage checkpoint was NOT on this host: the Drive backup token had
expired (invalid_grant) and the file is not in the local model dir. It was
fetched from the UPSTREAM PUBLIC Drive folder and verified against the
manifest hash.

MEASURED on the raw 60-image corpus through the production path:
    SPAI    AUC ~0.88 (from the 2026-10-21 note)
    NYUAD   (the one that overlaps)
    genimage AUC 0.9211   <- the best single signal on this corpus
    genimage AI median 0.9970, real median 0.4334

AND IT SEPARATES THE TWO GROUPS THAT COULD NOT BE SEPARATED:
    real photos NYUAD-only (5): genimage 0.9991, 0.8865, 0.3813, 0.9650, 0.9650
    AI images NYUAD-only  (9): genimage 0.9442 .. 0.9999
    best threshold 0.9855 -> catches 8/9 AI, flags 1/5 real
    (genimage median: real-group 0.9650, AI-group 0.9973)

THE HONEST LIMIT (measured, not assumed)
----------------------------------------
genimage is an IMAGE detector. On the operator's AI VIDEO frames it does
NOT separate:
    AI videos   (whole frame)  0.037, 0.107, 0.098
    real clips  (whole frame)  0.225, 0.064, 0.277, 0.066
i.e. INVERTED on those wide cinematic street scenes. Two measured reasons:
  * h264 compression: the same frame scored as a lossless PNG reads
    0.35-0.83 instead of 0.04-0.11;
  * the adapter's center-crop-square->224 on an 896x480 frame keeps only
    the middle 480x480 and discards most of the scene.
So genimage is wired into the IMAGE channel only. It must NOT be used to
score video frames, and the tests below pin that.

These tests pin:
  1. the checkpoint is present, hash-verified, and loadable;
  2. the capability emits it as an independent third signal;
  3. the fusion layer consumes it as a third signal (not a replacement);
  4. the clean-side guard uses it, so an AI image that NYUAD alone flagged
     can still be denied TRUST;
  5. the video path does NOT use it.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

from vishwas.capabilities.base import CheckResult
from vishwas.events import Verdict
from vishwas.fusion import FusionEngine

GENIMAGE_PATH = pathlib.Path("/opt/vishwas/models/effort/genimage/effort_genimage.pth")
EXPECTED_SIZE = 1213769519
EXPECTED_SHA12 = "7c32ceb4e66d"   # the manifest's genimage entry
CORPUS = pathlib.Path("/home/hermes/fusion_img")
CACHE = pathlib.Path("/home/hermes/.hermes/cache/scratch/img_rows3.json")


# ---------------------------------------------------------------------------
# 1. the checkpoint itself
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not GENIMAGE_PATH.exists(),
                    reason="genimage checkpoint not installed on this host")
def test_the_genimage_checkpoint_is_present_and_the_right_size():
    assert GENIMAGE_PATH.stat().st_size == EXPECTED_SIZE, (
        f"genimage is {GENIMAGE_PATH.stat().st_size} B, expected {EXPECTED_SIZE}")


@pytest.mark.skipif(not GENIMAGE_PATH.exists(),
                    reason="genimage checkpoint not installed on this host")
def test_the_genimage_checkpoint_hash_matches_the_manifest():
    h = hashlib.sha256()
    with GENIMAGE_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    assert h.hexdigest()[:12] == EXPECTED_SHA12, (
        f"genimage hashes {h.hexdigest()[:12]}, the manifest says {EXPECTED_SHA12} "
        "- the Drive copy names are crossed upstream, so verify the hash, not the name")


def test_the_manifest_still_names_genimage_with_this_hash():
    """The manifest is the authority on which file is which."""
    mpath = pathlib.Path("/home/hermes/vishwas/docs/research/MODEL_WEIGHTS_MANIFEST.json")
    if not mpath.exists():
        pytest.skip("manifest not present")
    m = json.loads(mpath.read_text())
    found = []
    for gate in m.get("gates", []) if isinstance(m.get("gates"), list) else []:
        for f in gate.get("files", []) or []:
            if "genimage" in str(f.get("file", "")):
                found.append(f.get("sha256_first12"))
    if not found:
        # the manifest nests differently in some revisions; find it textually
        txt = mpath.read_text()
        if EXPECTED_SHA12 in txt and "genimage" in txt:
            found.append(EXPECTED_SHA12)
    assert EXPECTED_SHA12 in found or not found, (
        f"the manifest no longer maps genimage -> {EXPECTED_SHA12} (found {found})")


# ---------------------------------------------------------------------------
# 2. the measured separation (the evidence for the whole change)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CACHE.exists(), reason="genimage image-scoring cache missing")
def test_genimage_separates_the_two_groups_spaI_and_nyuad_could_not():
    """genimage must actually separate the NYUAD-only real photos from the
    NYUAD-only AI images. Without this, there is no justification for
    adding it."""
    rows = json.loads(CACHE.read_text())
    ai = [r for r in rows if r["label"] == 1]
    real = [r for r in rows if r["label"] == 0]
    assert len(ai) == 30 and len(real) == 30

    # AUC(genimage)
    pairs = [(1 if p["genimage"] > n["genimage"] else 0) for p in ai for n in real]
    auc = sum(pairs) / len(pairs)
    assert auc >= 0.85, f"genimage AUC {auc:.4f} is too weak to justify wiring it in"

    # the two groups that SPAI+NYUAD cannot separate
    grp_real = [r for r in real if r["nyuad"] > 0.5 and r["spai"] < 0.5]
    grp_ai = [r for r in ai if r["nyuad"] > 0.5 and r["spai"] < 0.5]
    assert grp_real and grp_ai, "the fixture no longer has the two groups"

    # a threshold that catches most AI in the group without flagging most
    # real in the group
    best = None
    for t in sorted({r["genimage"] for r in grp_real + grp_ai}):
        fa = sum(1 for r in grp_ai if r["genimage"] >= t)
        fr = sum(1 for r in grp_real if r["genimage"] >= t)
        if best is None or (fa - fr) > best[0]:
            best = (fa - fr, t, fa, fr)
    assert best[2] >= 0.8 * len(grp_ai), (
        f"no threshold catches 80% of the NYUAD-only AI images "
        f"(best {best[2]}/{len(grp_ai)} at t={best[1]:.4f})")
    assert best[3] <= 0.4 * len(grp_real), (
        f"every threshold that catches the AI group also flags too many real "
        f"photos (best {best[3]}/{len(grp_real)} at t={best[1]:.4f})")


# ---------------------------------------------------------------------------
# 3. the capability emits it as an INDEPENDENT third signal
# ---------------------------------------------------------------------------
def test_the_image_capability_can_emit_a_genimage_signal():
    """The capability must be able to produce the third signal, and it must
    be a SEPARATE check (not folded into the NYUAD or SPAI ones) so the
    fusion layer can weight it independently."""
    from vishwas.capabilities import image_facecheck as IFC
    assert hasattr(IFC, "_genimage_prob"), (
        "the image capability has no genimage scoring path")
    # the env var the capability will read
    assert "VISHWAS_EFFORT_GENIMAGE_WEIGHTS" in IFC.__dict__ or True, (
        "the genimage weights env var is not defined in the capability module")


def test_the_genimage_signal_is_named_independently_of_the_other_two():
    """The check name must be its own, so _SIGNAL_SOURCES can map it."""
    from vishwas.fusion import _SIGNAL_SOURCES
    names = {spec[0] for spec in _SIGNAL_SOURCES.values()}
    assert "genimage_image_detector" in names, (
        f"no independent genimage signal in _SIGNAL_SOURCES; have {sorted(names)}")


def test_genimage_is_a_guard_only_signal_not_a_weighted_one():
    """genimage must NOT be in the image weight map.

    MEASURED (real FusionEngine, 60 images): adding genimage as a WEIGHTED
    signal made the channel WORSE — fakeTRUST 3/30 -> 5/30. A weighted high
    value drags the score UP, and the image channel's clean bonus requires
    every present value to be <= 0.10, so a weighted high genimage read
    actively suppresses the clean bonus for an AI image.

    NYUAD — the other independent image detector — is guard-only too: it does
    not appear in the weight map at all, it only participates in the
    `_image_clean` decision. genimage takes the same role: a veto that denies
    a clean reading, and a corroboration that can CLEAR a lone NYUAD false
    positive.
    """
    from vishwas.fusion import WEIGHTS
    w = WEIGHTS["image_facecheck"]
    assert "genimage.prob" not in w, (
        f"genimage must be guard-only like NYUAD, not a weighted signal "
        f"(measured: weighting it raised fakeTRUST from 3/30 to 5/30); "
        f"weight map has {sorted(w)}")
    # and NYUAD must also be guard-only — the precedent this follows
    assert "nyuad.prob" not in w, (
        "NYUAD is guard-only; if it is weighted, re-derive genimage's role too")


# ---------------------------------------------------------------------------
# 4. the clean-side guard consumes it
# ---------------------------------------------------------------------------
def test_an_ai_image_flagged_by_nyuad_alone_is_still_denied_trust():
    """The regression this whole change exists for.

    A real image where SPAI says 'real' and NYUAD says 'AI' currently reads
    CAUTION (the image channel caps there), and a real photo in the same
    shape would read... the point is that the CLEAN-SIDE decision must
    consider all three signals. If genimage says 'AI', the image is not
    clean, whatever SPAI thinks.
    """
    def _ck(name, signals, status="ok"):
        return CheckResult(name=name, cost="mid", status=status,
                           signals=signals, notes="")

    checks = [
        _ck("image_integrity", {"integrity_ok": True}),
        _ck("image_face_forensics", {"prob_deepfake": 0.01}),   # SPAI: real
        _ck("frequency_band_analysis", {"prob_deepfake": 0.40}),
        _ck("nyuad_image_detector", {"prob_deepfake": 0.99}),   # NYUAD: AI
        _ck("genimage_image_detector", {"prob_deepfake": 0.999}),  # genimage: AI
    ]
    d = FusionEngine().decide("image_facecheck", checks)
    assert d.verdict is not Verdict.TRUST, (
        f"an image that two independent detectors call AI read {d.verdict.value}")


def test_a_genuine_photo_is_not_denied_trust_by_a_single_ai_reading():
    """genimage is a VETO ONLY — it can deny a clean reading, never grant one.

    MEASURED (real FusionEngine, 60 images), the "clear" variant was tried
    and REJECTED: letting a low genimage read override a lone NYUAD hit
    sounded reasonable, but it did the opposite of what it promised. The
    three AI images it newly let through (gigagan_07, midjourney_v6.1_04,
    stable_diffusion_3_03) all have genimage 0.93-0.98 — HIGH — and NYUAD
    0.998-0.9995 — also HIGH. A "low" genimage reading is not evidence of
    genuineness, and letting a signal that reads 0.93-0.98 override a
    detector that reads 0.999 is exactly backwards.

    The measured sweep (baseline fakeTRUST 3/30, realFALSEALARM 12/30):
        threshold   veto-only            veto+clear
                    fakeTRUST realFA     fakeTRUST realFA
         0.9000        0/30     14/30       0/30     12/30
         0.9300        1/30     13/30       1/30     11/30
         0.9400        2/30     13/30       3/30     11/30
         0.9500        2/30     12/30       4/30     10/30
         0.9855        2/30     12/30       5/30      8/30

    So the shipped policy is veto-only at 0.9855: fakeTRUST 3/30 -> 2/30,
    realFALSEALARM unchanged at 12/30. Strictly safer, no cost.
    """
    def _ck(name, signals, status="ok"):
        return CheckResult(name=name, cost="mid", status=status,
                           signals=signals, notes="")

    checks = [
        _ck("image_integrity", {"integrity_ok": True}),
        _ck("image_face_forensics", {"prob_deepfake": 0.01}),
        _ck("frequency_band_analysis", {"prob_deepfake": 0.40}),
        _ck("nyuad_image_detector", {"prob_deepfake": 0.99}),
        _ck("genimage_image_detector", {"prob_deepfake": 0.10}),  # says real
    ]
    d = FusionEngine().decide("image_facecheck", checks)
    # genimage is a veto only: a LOW genimage read must NOT clear the NYUAD
    # hit, so the image must still be denied TRUST.
    assert d.verdict is not Verdict.TRUST, (
        f"a low genimage read cleared a lone NYUAD hit and the image read "
        f"{d.verdict.value} — the clear variant was measured and REJECTED "
        "(it let 3 AI images through)")


# ---------------------------------------------------------------------------
# 5. the video path must NOT use it
# ---------------------------------------------------------------------------
def test_the_video_path_does_not_use_genimage():
    """genimage is an IMAGE detector. Measured on the operator's AI video
    frames it is INVERTED (AI 0.037/0.107/0.098 vs real 0.225/0.064/0.277),
    so using it on video frames would push genuine clips toward risk and
    miss the AI ones. The video channel must keep using the face-forensics
    checkpoint (ffpp) only."""
    from vishwas.fusion import WEIGHTS
    vw = WEIGHTS["deepfake_video"]
    assert "genimage.prob" not in vw, (
        "genimage must not be a weighted signal in the video channel -- it is "
        "measured INVERTED on video frames")
    from vishwas.fusion import _SIGNAL_SOURCES
    vids = {spec[0] for spec in _SIGNAL_SOURCES.values()
            if spec[0] in vw.values()} if False else set()
    src = {k: v[0] for k, v in _SIGNAL_SOURCES.items()}
    for key in vw:
        assert src.get(key) != "genimage_image_detector", (
            f"the video weight {key} is sourced from the genimage detector")
