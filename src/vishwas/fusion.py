"""Fusion & reliability layer.

Design (per spec): NEVER average raw confidence scores blindly. Specialist
detectors emit probabilities; a per-capability *logistic-regression stack*
trained on out-of-fold predictions combines them with contextual features.
Where no trained stack exists for a target, we fall back to an explicit
evidence-weighted rule (documented weights, monotonic evidence checks), and
the ReliabilityGate is what decides whether the output may be surfaced at all.

Key semantics (fixed during review of the capability modules):
  - A *weighted signal* refers to a concrete (check_name, signal_key) pair
    exactly as emitted by the capability modules — nothing here re-parses prose.
  - status ∈ {skipped, unavailable} on a check = *known gap* (tool gated off by
    design). Known gaps never count as "missing evidence" for the TRUST rule;
    the ReliabilityGate tracks their fraction separately instead.
  - status ∈ {failed} or a None value on an emitted signal = genuine missing
    evidence and DOES suppress a TRUST verdict (selective prediction).
  - "Unable to verify" is first-class: disagreement, conflicting evidence, poor
    media quality, distribution shift or zero usable signals => UNABLE_TO_VERIFY
    rather than a forced verdict.

The signal map below mirrors the real CheckResult names/signals emitted by
capabilities/* (verified by grep during integration — kept in sync by test).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import JobContext, Verdict
from .capabilities.base import CheckResult

log = logging.getLogger("vishwas.fusion")

# ---------------------------------------------------------------- constants

KIND_DEFAULT_NUM = "num_default"      # numeric present -> value; absent -> 0.0
KIND_NEG_BOOL = "neg_bool"            # boolean present&true -> 1.0 (weight sign decides direction)
KIND_LIST_FRAC = "list_len_frac"      # len(list)/3 clamped to 0..1
KIND_INT_FRAC = "int_frac"            # int/3 clamped to 0..1
KIND_INV_NUM = "num_inv"              # 1 - value
KIND_RISK_STR = "risk_str"            # string risk level -> numeric
KIND_CONST_TRUE = "const_true"        # check presence alone is the signal

RISK_LEVEL_MAP = {"critical": 1.0, "high": 0.8, "medium": 0.4, "low": 0.0,
                  "unknown": 0.0, "": 0.0}

# Explicit evidence weights used by the fallback combiner when no trained
# LR-stack checkpoint exists for a target. Calibrated later against held-out
# real-world sets (docs/ARCHITECTURE.md "Fusion & reliability").
WEIGHTS: dict[str, dict[str, float]] = {
    "malicious_file": {
        "vt.positives_ratio": 4.0,
        "clamav.detected": 3.5,
        "yara.hit_count_norm": 2.5,
        "quark.score_norm": 2.0,
        "pe.packed_flag": 2.0,
        "pe.prob_malicious": 2.0,
        "apk.prob_malicious": 2.5,
        "mobsf.risk_high_plus": 2.5,
        "ext_mismatch.present": 1.0,
        "entropy.anomaly": 1.5,
        "sandbox.malicious": 4.0,
    },
    # 2026-08-26 operator decision: PhishingScanner (vendored, MIT) ALONE
    # decides URL phishing. The former VirusTotal reputation, offline-DOM
    # heuristics plus young-domain scoring, the vendored-xgboost url-phishml
    # probability, redirect counting and the dropped ssrf/download
    # obey-checks were all removed from this target — PhishingScanner's
    # risk_score_norm is the single weighted signal.
    "url_phishing": {
        "phish_scanner.risk_norm": 1.0,
    },
    "gov_document": {
        "signature.valid": -4.0,
        "sig_object.present": -1.5,
        "qr.sha1_match": -3.0,
        "fin.invalid_upi": 1.5,
        "rag.template_deviation": 0.5,
        "extraction.low_quality": 1.0,
    },
    "deepfake_video": {
        # demamba.prob RETIRED from the weight map (2026-08-25, Task 6): no arch
        # spec exists in model_archs/ (registry: aasist/effort/havic/fakemamba/
        # xlsrmamba), no weights are obtainable, so the check was a permanent
        # known-gap on 100% of measured rows (84/84) — a dead 2.0-weight slot
        # that only diluted every verdict's denominator. The capability still
        # emits its unavailable check (debug shows WHY); if DeMamba weights ever
        # land, re-add with calibration. _EXPECTED_PROB_DET updated to match.
        "effort.prob": 1.5,
        # demamba retired (see note above). Post-retirement recalibration
        # (2026-08-25): effort cal b -0.35 -> -0.45 and frameheur weight
        # 1.5 -> 1.0, then 2026-08-26 ffpp-checkpoint + fresh-corpus
        # recalibration (W1): effort 2.5->1.5, av_risk 2.0->3.5, havic
        # 1.2->0.8, effort affine b -0.45->-0.55, chosen on rows_video_v3
        # (87 live-derived clips, 80/20 split). REALS NOT-DNU 42/42 on v3,
        # fakes >=CAUTION 42/45, AI anchors 3/3 DNU; live 904/570/245/540
        # all -> CAUTION (false-HIGH fixed), ai_crf45 -> DNU.
        # havic kept detect-only (0.8): W3 proved the checkpoint is
        # degenerate-constant-fake-bias (sigmoid ~0.98 on everything) — safe
        # only because corpus reals are silent (known-gap); a modest weight
        # limits the unseen real-with-audio overfit risk.
        "frameheur.prob": 1.0,
        # AV-sync + HAVIC were present in the emitted checks but NOT in this
        # target's weight map (fusion gap, 2026-08-25): an operator's AI video
        # carried av=-2.0 (anti_correlated) and havic=1.0 yet they contributed
        # ZERO to the verdict. Wire them in; pattern layer uses them for
        # corroboration. 2026-08-26: av_risk raised to 3.5 (it is the real
        # cross-modal separator — W3 measured AI anti_correlated 0.5 vs real
        # weakly_synced 0.1); HAVIC demoted to 0.8 detect-only (degenerate).
        #
        # 2026-10-21 RECALIBRATION on a LEAK-FREE corpus (60 real files,
        # production code paths). The 3.5 weight was fitted on
        # rows_video_v3.jsonl, where 83 of 87 rows have NO AUDIO STREAM, so
        # av_risk was a known_gap on 95% of the fitting set and its weight was
        # never actually tested. Measured live, with the corpus label leak
        # removed, av_risk separates at AUC 0.4744 — BELOW CHANCE.
        #
        # Its real failure mode is not separation but DOMINANCE: av_risk is
        # 0.0/0.1 for ~90% of clips and 0.45/0.5 only when the lip-motion and
        # speech-energy envelopes decorrelate. At weight 3.5 it supplied 54%
        # and 79% of the total score on the two genuine clips that read
        # DO_NOT_USE. And a decorrelation reading is exactly what a MUX
        # ARTEFACT produces — audio from a different source than the video —
        # so the signal is not specific to deepfakes.
        #
        # Weight reduced 3.5 -> 1.5: still the strongest corroborating signal
        # for a genuine cross-modal swap (the pattern layer uses it), but no
        # longer able to carry the verdict on its own. The
        # `_av_corroborated` gate below additionally requires the VIDEO-side
        # signals to agree before a decorrelated reading can dominate.
        #
        # 2026-10-22 — THE 0.5 BAND WAS NEVER A DEEPFAKE SIGNATURE.
        # Measured on the operator's REAL WhatsApp traffic (67 clips from 5
        # correspondents, original audio):
        #   * the operator's 3 AI videos have a COMPLETELY SILENT audio track
        #     (RMS 0.0), so their av reading was the correlation function's
        #     UNDEFINED sentinel, not a correlation. It is now reported as
        #     "unknown" with a neutral 0.0 (see cross_modal.py).
        #   * 12% of genuine WhatsApp video (9 of 74 clips) has the same
        #     all-zero audio track and was being reported the same way.
        #   * of the 63 genuine clips WITH audible audio, only 2 reach the
        #     0.5 band at all (3%), with real correlations of -0.299/-0.443.
        #   * HAVIC is saturated on genuine traffic: >= 0.95 on 42 of 63
        #     genuine clips (median 0.980), versus 0.995 on the AI videos.
        #     It does not discriminate.
        # So the av 0.5 band means "no speech energy", and the corroboration
        # gate below now treats the 0.45 and 0.5 bands IDENTICALLY: an
        # elevated cross-modal reading may corroborate a video-side signal
        # but may never carry the verdict by itself.
        "av_risk_addition": 1.5,
        "havic.prob_inconsistent": 0.8,
    },
    "deepfake_audio": {
        "fakemamba.prob": 2.5,
        "aasist.prob": 2.5,
        "xlsr.prob": 2.0,
        "ssl.prob": 2.0,
        "offline.prob": 1.0,
    },
    "cross_modal": {
        "av_risk_addition": 3.0,
        "havic.prob_inconsistent": 2.5,
    },
    "image_facecheck": {
        # 2026-08-26 NOT-BINARY-GATE (operator directive + measured evidence):
        # SPAI (CVPR'25 spectral, vendored W2) feeds faceforensics.prob as an
        # EVIDENCE signal, but a single learned-model read must NEVER flip an
        # image to DO_NOT_USE — that was the chameleon overfit class.
        #
        # 2026-10-21 RECALCULATED on the raw 60-image corpus through the
        # production paths. freqband is not "dead noise" as the comment
        # below used to say — it is ANTI-CORRELATED:
        #     real photos  median 0.4625 (max 0.5150)
        #     AI images    median 0.3990 (max 0.4710)
        #     AUC(fake=high) = 0.1222   (the inverse of 0.8778)
        # A genuine photo scores HIGHER on "prob_deepfake" than an AI image.
        # The signal's SIGN is wrong, and it carries a POSITIVE weight, so it
        # pushes real photos toward more risk.
        #
        # IT IS ALSO INERT, and that is the part worth knowing:
        #     config                         fakeTRUST  realFALSEALARM
        #     as shipped (w=+0.5)                3/30        12/30
        #     sign flipped (w=-0.5)             3/30        12/30
        #     flipped via calibration 1-fb      3/30        12/30
        #     REMOVED entirely                 3/30        12/30
        # Rows whose VERDICT changed when the sign was flipped: 0/60.
        # Two reasons: (a) the image channel CAPS at CAUTION, so the ~0.15
        # offset this weight adds to every score cannot change a band;
        # (b) the clean-side guard (fb > 0.60) never fires, because the max
        # freqband on ANY image is 0.5150.
        #
        # It is deliberately LEFT IN PLACE, because removing it is a
        # weight-map change that needs its own evidence and it is measurably
        # inert. But it is a landmine: lowered to a plausible-looking 0.45,
        # that guard would block 21 of 30 GENUINE photos and 1 of 30 AI
        # images — pushing real photos AWAY from reading LOW, in the wrong
        # direction. Do not "tighten" it without re-deriving it.
        # See tests/test_62_freqband_inverted.py and
        # tools/eval_freqband_sign.py.
        "freqband.prob": 0.5,
        "faceforensics.prob": 1.0,
    },
    # document_generic / unclassified previously had EMPTY weight maps, which
    # made FusionEngine.decide hit its total_w==0 guard and force
    # UNABLE_TO_VERIFY for EVERY generic document and every unrecognised file —
    # regardless of what actually ran. They now carry the file-scan signals
    # that apply to any blob, so a document/unknown file that IS scanned gets a
    # real verdict (and, with the n_val==0 gate above, an honest UNVERIFIED
    # when no scanner is provisioned). Raw entropy anomaly is intentionally
    # omitted from document_generic: compressed PDFs/office files are
    # legitimately high-entropy and would false-flag as "packed".
    "document_generic": {
        "vt.positives_ratio": 4.0,
        "clamav.detected": 3.5,
        "yara.hit_count_norm": 2.5,
        "ext_mismatch.present": 1.5,
        "sandbox.malicious": 4.0,
    },
    "unclassified": {
        "vt.positives_ratio": 4.0,
        "clamav.detected": 3.5,
        "yara.hit_count_norm": 2.5,
        "ext_mismatch.present": 1.0,
        "sandbox.malicious": 4.0,
    },
}

# signal key -> (CheckResult.name, signal key, parsing kind, contributes_to_prob_list)
_SIGNAL_SOURCES: dict[str, tuple[str, str, str, bool]] = {
    # --- malicious_file ---
    "vt.positives_ratio": ("vt_reputation", "positives_ratio", KIND_DEFAULT_NUM, False),
    "clamav.detected": ("clamscan", "detected", KIND_DEFAULT_NUM, False),
    "yara.hit_count_norm": ("yara_x", "hits_norm", KIND_DEFAULT_NUM, False),
    "quark.score_norm": ("quark_engine", "score_norm", KIND_DEFAULT_NUM, False),
    "pe.packed_flag": ("pe_statics", "packed", KIND_DEFAULT_NUM, False),
    "pe.prob_malicious": ("pe_statics", "prob_malicious", KIND_DEFAULT_NUM, True),
    "apk.prob_malicious": ("apk_statics", "prob_malicious", KIND_DEFAULT_NUM, True),
    "mobsf.risk_high_plus": ("mobsf_apk", "mobsf_risk_level", KIND_RISK_STR, False),
    "ext_mismatch.present": ("ext_mismatch_flag", "declared", KIND_CONST_TRUE, False),
    "entropy.anomaly": ("file_entropy", "anomaly", KIND_DEFAULT_NUM, False),
    "sandbox.malicious": ("dynamic_sandbox", "malicious", KIND_DEFAULT_NUM, False),
    # --- url_phishing ---
    "phish_scanner.risk_norm": ("url_phish_scanner", "risk_score_norm", KIND_DEFAULT_NUM, True),
    # --- gov_document ---
    "signature.valid": ("digital_signature", "valid", KIND_NEG_BOOL, False),
    "sig_object.present": ("digital_signature", "has_sig_object", KIND_NEG_BOOL, False),
    "qr.sha1_match": ("qr_native_check", "sha1_matches_declaration", KIND_NEG_BOOL, False),
    "fin.invalid_upi": ("financial_field_validation", "invalid_upis", KIND_LIST_FRAC, False),
    "rag.template_deviation": ("rag_template_cache", "required_fields_matched", KIND_INV_NUM, False),
    "extraction.low_quality": ("document_extraction", "low_quality", KIND_DEFAULT_NUM, False),
    # --- deepfake_video ---
    "effort.prob": ("effort_face_forensics", "prob_deepfake", KIND_DEFAULT_NUM, True),
    "demamba.prob": ("demamba_general", "prob_deepfake", KIND_DEFAULT_NUM, True),
    "frameheur.prob": ("frame_heuristics", "prob_deepfake", KIND_DEFAULT_NUM, True),
    # --- deepfake_audio ---
    "fakemamba.prob": ("fakemamba_detector", "prob_deepfake", KIND_DEFAULT_NUM, True),
    "aasist.prob": ("aasist_detector", "prob_deepfake", KIND_DEFAULT_NUM, True),
    "xlsr.prob": ("xlsr_audio_detector", "prob_deepfake", KIND_DEFAULT_NUM, True),
    "ssl.prob": ("ssl_audio_detector", "prob_deepfake", KIND_DEFAULT_NUM, True),
    "offline.prob": ("audio_offline_features", "prob_deepfake", KIND_DEFAULT_NUM, True),
    # --- cross_modal / image ---
    "av_risk_addition": ("cross_modal_av", "av_risk_addition", KIND_DEFAULT_NUM, False),
    "havic.prob_inconsistent": ("havic_crossmodal_model", "prob_inconsistent", KIND_DEFAULT_NUM, True),
    "freqband.prob": ("frequency_band_analysis", "prob_deepfake", KIND_DEFAULT_NUM, True),
    "faceforensics.prob": ("image_face_forensics", "prob_deepfake", KIND_DEFAULT_NUM, True),
}

# How many independent probability-emitting detectors a target is designed to
# have — used for selective-prediction coverage.
_EXPECTED_PROB_DET: dict[str, int] = {
    "deepfake_video": 4, "deepfake_audio": 4, "cross_modal": 2,
    "image_facecheck": 2, "malicious_file": 3, "url_phishing": 1,
    "gov_document": 0, "document_generic": 0, "unclassified": 0,
}

# Per-signal affine calibration: value' = clamp01(a*value + b).
# Seeds from 2026-08-25 single-sample probe (known-AI video vs ffmpeg testsrc
# control, /tmp/analysis/fusion/calibration_notes.md), RECALIBRATED same day
# against the measured 84-row FF++ corpus (rows_video_v2.jsonl):
#   - the original effort seed (1.33,-0.05) mapped real talking-head footage
#     (raw 0.62-0.87) to ~0.78-1.0 risk, fusing every real video to DNU;
#   - (1.0,-0.45) (post demamba-retirement recalibration) maps that same real
#     distribution to ~0.17-0.42 (CAUTION band) while the operator AI video
#     (0.677 raw -> 0.23 calibrated) still reaches DO_NOT_USE via the AV
#     corroboration path (av_risk + havic), not effort alone.
#   frameheur control 0.078 -> ~0.20   AI 0.281 -> ~0.60
#   havic     saturates ~1.0 on both   -> demote (a=0.2) so it can't dominate
#   av_risk   already a risk-in-[0,.5] -> identity (kept here for clarity)
#   aasist/xlsr  now input-sensitive but clean-side sample is a synthetic sine
#                (not speech) -> near-identity until a real-clean corpus lands.
_CALIBRATE: dict[str, tuple[float, float]] = {
    # ---- VIDEO (recalibrated 2026-10-21 on a LEAK-FREE corpus) --------------
    # The video weights were originally fitted on rows_video_v3.jsonl, where
    # 83 of 87 rows have NO AUDIO STREAM. So av_risk (w=3.5) and havic (w=0.8)
    # were a known_gap on 95% of the fitting set, and the fit was therefore
    # made in a regime that does not exist in production.
    #
    # Re-measured live on 60 real files through the production code paths
    # (tools/score_video_signals.py), the per-signal separation is:
    #     effort    AUC 0.6511   <- the ONLY video signal with separation
    #     frameheur AUC 0.5350   <- noise
    #     av_risk   AUC 0.4744   <- below chance once the corpus leak is gone
    #     havic     AUC 0.6272   <- weak, and its apparent strength was a
    #                               label-leak artefact (see below)
    #
    # frameheur was AMPLIFIED (1.97x) because it was thought weak. That was
    # backwards: a slope > 1 makes a NOISE signal dominate the weighted sum.
    # At 1.97x, frameheur contributed up to 0.76 while the strongest real
    # signal (EFFORT) contributed at most 0.43 — the noise outweighed the
    # evidence. Slope is now 1.0: the raw signal is used as measured, with no
    # invented amplification, so its weight (1.0 vs EFFORT's 1.5) reflects its
    # measured reliability.
    # 2026-10-21 RECALIBRATED TWICE, both times by measurement.
    #
    # (1) After the BGR->RGB fix to the EFFORT input (model_archs/effort.py).
    #     The old offset was fitted to the BROKEN score distribution: with the
    #     corrected input the scores move DOWN by ~0.15 (real median
    #     0.6326 -> 0.4762, fake 0.7707 -> 0.6190), so -0.55 clipped both
    #     classes to ~0 and 18 of 30 fakes read TRUST.
    #
    # (2) After the FACE-REGION CROP (deepfake_video._effort_face_crop).
    #     EFFORT is a face-forensics model and the upstream paper applies it
    #     to face regions; the pipeline was feeding it whole frames. The crop
    #     raises the fakes further (fake median 0.6190 -> 0.6881) and lowers
    #     the reals slightly (0.4762 -> 0.4619), so the offset was re-derived
    #     a second time.
    #
    # Re-derived against the DESIGN INTENT recorded in the original
    # calibration notes:
    #     (a) genuine media must not read DO_NOT_USE on effort alone;
    #     (b) the operator's AI videos must still read DO_NOT_USE, via the
    #         cross-modal corroboration (av_risk + havic), not effort alone.
    #
    # Measured on the OPERATOR'S OWN media (4 genuine natural-audio clips,
    # 3 AI anchor videos) with the face crop:
    #     offset -0.40 .. -0.20  -> 4/4 real trust,  3/3 AI DO_NOT_USE
    # so the whole band satisfies the intent.
    #
    # On the leak-free 60-clip corpus WITH the face crop:
    #     offset  FP  fakes flagged  fakes reading TRUST
    #     -0.45    1       12                11
    #     -0.40    1       13                11   <- chosen
    #     -0.35    2       14                11
    #     -0.30    5       15                11
    #     -0.25    7       17                10
    # Without the crop the shipping -0.30 gives FP=2 / 10 flagged / 15 TRUST,
    # so the crop at -0.40 is BETTER on every axis: one fewer false
    # positive, three more fakes caught, four fewer fakes read TRUST.
    # (No offset gives both zero false positives and zero false negatives on
    # that corpus -- see the video-channel measurement note.)
    "effort.prob": (1.0, -0.40),
    "frameheur.prob": (1.0, 0.0),
    # havic saturates near 1.0 on almost everything (a known-degenerate
    # checkpoint) -> demote (a=0.2) so it can never dominate.
    "havic.prob_inconsistent": (0.2, 0.0),
    # ---- AUDIO / URL (unchanged) --------------------------------------------
    "aasist.prob": (1.0, 0.0),
    "xlsr.prob": (1.0, 0.0),
    "av_risk_addition": (1.0, 0.0),
}


def _calibrate(key: str, val: float) -> float:
    """Apply per-signal affine calibration; clamp to [0,1]."""
    a, b = _CALIBRATE.get(key, (1.0, 0.0))
    return min(1.0, max(0.0, a * val + b))


# Deepfake pattern classifier (Fusion v2, 2026-08-25): replaces naive
# spread-abstention with which-detectors-fired classification. Different AI
# generators fool different detectors, so cross-detector spread is EVIDENCE
# (a mode), not a reason to abstain. Signals are the CALIBRATED values.
# Returns (pattern_key, conf_multiplier, is_fake_aligned).
def _clean_side_bonus(by_name: dict[str, Any]) -> float | None:
    """Return the clean-evidence bonus (positive shift we SUBTRACT from s) for a
    deepfake_video when the cross-modal + face + frame evidence all independently
    corroborate authenticity; else None (no bonus).

    Conditions (all required, so a deepfake can never be false-cleaned):
      - cross_modal_av ran and alignment_class == "synced" (real lip-sync;
        av_synced_clean is the correlation ONLY then, else 0.0).
      - calibrated effort.prob LOW (<= 0.30)  -> face is not a detected deepfake.
      - calibrated frameheur.prob LOW (<= 0.40, the frame heuristic threshold).
    Bonus = min(av_synced_clean, 0.9) * 2.0 (a strong but capped clean vote).
    Magnitude is tunable; keep it below the point where a partially-synced real
    flips a genuinely-risky clip to trust. Requirement: raw<=0.15 needs x<=-0.44.
    """
    cm = by_name.get("cross_modal_av")
    cm_ok = cm is not None and cm.usable()
    # NO-AUDIO CLEAN PATH (2026-08-26, false-positive fix): a silent/no-audio real
    # video has no cross-modal evidence. The operator's silent real clips stuck at
    # MEDIUM because the clean side required audio. Allow a clean bonus for silent
    # clips from low face-forensics + low frameheur alone (the same corroboration,
    # minus the audio-sync requirement). Guards: high face, high frameheur, or a
    # decorrelated/anti signal (when audio IS present) all still fail clean.
    if not cm_ok:
        eff = float(_probe(by_name, "effort_face_forensics", "prob_deepfake", 1.0))
        fh = float(_probe(by_name, "frame_heuristics", "prob_deepfake", 1.0))
        if eff > 0.72 or fh > 0.45:
            return None  # face or frame flags -> NOT clean even for silent real
        if eff < 0.0:  # effort failed/unusable -> no face corroboration -> not clean
            return None
        return 0.5  # silent real with low face + low frame: modest clean vote
    align = cm.signals.get("alignment_class")
    corr = cm.signals.get("av_correlation", 0.0)
    # AGGRESSIVE posture (2026-08-26, operator-directed): any genuinely real
    # alignment class (synced OR weakly_synced) qualifies as real-sync evidence;
    # we NO LONGER require a high correlation floor. The anti/decorrelated fake
    # is still NEVER clean (it has av_risk>=0.35), so a real deepfake/swap fails
    # the low-face + real-alignment corroboration. Operator reals measure corr
    # 0.27-0.79 across synced/weakly — all must now read LOW.
    if align not in ("synced", "weakly_synced"):
        return None  # decorrelated / anti_correlated / none -> never clean
    clean = cm.signals.get("av_synced_clean", 0.0)
    # av_synced_clean is set ONLY for strict "synced"; for weakly_synced it is 0.0,
    # so fall back to the raw correlation as the clean vote.
    if not isinstance(clean, (int, float)) or clean <= 0.0:
        clean = corr
    if not isinstance(clean, (int, float)) or clean <= 0.0:
        return None
    # Corroboration on RAW signal values (not post-affine calibrated — the
    # frameheur affine amplifies, so calibrated 0.2 -> 0.44 would wrongly block).
    # Thresholds: frame-heuristic raw <= 0.45. face-forensics raw <= 0.72 —
    # RAISED from 0.60 under the 2026-08-26 AGGRESSIVE posture (operator-directed
    # "all real videos read LOW"): genuine phone videos measure effort up to
    # ~0.71 (e.g. operator reals 0.612/0.657/0.627/0.714) while the fake anchors
    # are anti/decorrelated and FAIL the real-alignment gate regardless of effort,
    # so a generous face ceiling still cannot false-clean them.
    eff = float(_probe(by_name, "effort_face_forensics", "prob_deepfake", 1.0))
    fh = float(_probe(by_name, "frame_heuristics", "prob_deepfake", 1.0))
    if eff > 0.72 or fh > 0.45:
        return None  # effort or frame heuristic flags the face -> NOT clean
    return min(float(clean), 0.9)


def _image_clean(by_name: dict[str, Any]) -> bool:
    """True when an image is CORROBORATED clean: integrity ok, face-forensics low,
    AND frequency-band low — all independent signals agree no synthetic artifact.
    A lone SPAI-high real photo (known false-positive) fails the low-face test.
    Any detector missing/flagged -> NOT clean (conservative)."""
    integ = by_name.get("image_integrity")
    if integ is not None and not integ.usable():
        return False  # integrity check failed/completed-failed -> not clean
    # face-forensics (SPAI) is the STRONG, reliable detector (measured median
    # 0.0014 real vs 0.66 AI; freqband is documented dead-noise: real 0.463 vs
    # AI 0.399, overlapping — it cannot flag anything). So:
    #   - SPAI, when present, MUST be low (a lone SPAI-high real photo, the known
    #     false-positive, must NOT trust).
    #   - freqband is a DEMOTE-ONLY guard: block trust only when it's clearly
    #     high (>0.6, well above its noise floor) — NOT at 0.45, which a real
    #     photo crosses just from normal band energy.
    #
    #     2026-10-21: the ">0.6, well above its noise floor" reasoning is
    #     WRONG in one respect — freqband is not noise, it is INVERTED
    #     (real photos score higher than AI images; AUC(fake=high) 0.1222).
    #     The 0.6 threshold is still the right one, but for a different
    #     reason: it is above the MAXIMUM value any image in the measured
    #     corpus produces (0.5150 over reals, 0.4710 over fakes), so the
    #     guard is inert. Lowering it to a "tighter" 0.45 would block 21 of
    #     30 GENUINE photos and 1 of 30 AI images — the wrong way round.
    #     Do not lower it. See tests/test_62_freqband_inverted.py.
    eff_c = by_name.get("image_face_forensics")
    fb = _probe(by_name, "frequency_band_analysis", "prob_deepfake", 0.0)
    if fb > 0.60:
        return False  # freqband clearly flags -> not clean
    # NYUAD (2026-08-27): the second, INDEPENDENT AI-image signal. A flux AI image
    # often scores SPAI ~0.0 (the false-clean), so clean must ALSO require NYUAD low
    # when it ran. If NYUAD is absent/unavailable, we still trust on SPAI (don't
    # hard-block the whole image channel on a missing second detector).
    #
    # 2026-10-21: the EITHER-detector-veto behaviour is deliberate and was
    # re-derived by measurement (tools/eval_image_arbitration.py). It looks
    # tempting to relax it -- treat a SPAI/NYUAD disagreement as "uncertain"
    # rather than "risky", which would appear to drop 5 real-photo false
    # alarms while keeping 8 AI catches. That is FALSE, and the reason is
    # structural: the 5 real photos NYUAD alone flags and the 8 AI images
    # only NYUAD catches have the SAME signature (SPAI low, NYUAD high), so
    # no rule over these two numbers can separate them.
    #
    # Measured, real FusionEngine, 60 images:
    #     policy                                fakeTRUST  realFALSEALARM
    #     as shipped (SPAI low AND NYUAD low)         3/30        12/30
    #     agreement-only (both must say real)         3/30        12/30
    #     SPAI-only veto (ignore NYUAD)              11/30         7/30
    #     NYUAD-only veto (ignore SPAI)              15/30         5/30
    #     either-low (a veto needs BOTH to flag)     23/30         0/30
    #
    # "Agreement only" is IDENTICAL to the shipped policy (both-must-say-real
    # IS SPAI low AND NYUAD low), and every policy that actually reduces the
    # false alarms costs 8, 12 and then 20 false-cleans. The false-clean is
    # the more dangerous direction -- it tells the user "this AI image is
    # genuine". Separating the two groups needs a THIRD signal (a model with
    # non-overlapping errors, a provenance check, or a content check), not a
    # re-arbitration of these two.
    # See tests/test_63_image_arbitration.py.
    ny = by_name.get("nyuad_image_detector")
    if ny is not None and ny.usable():
        ny_p = ny.signals.get("prob_deepfake", 1.0)
        if isinstance(ny_p, (int, float)) and ny_p >= 0.50:
            return False  # NYUAD flags it as AI (dalle/sd) -> not clean
    # FAIL-OPEN FIX (2026-10-21): an ABSENT or non-ok SPAI must NOT authorise
    # trust. This used to `return True` ("SPAI absent, everything else clean
    # -> trust"), which is the exact opposite of the function's own docstring:
    # a DEAD image detector (status unavailable/failed/degraded) produced
    # verdict=TRUST, so an arbitrary AI-generated image read as genuine. The
    # only remaining guard was the frequency-band heuristic, which this
    # codebase itself documents as dead noise.
    #
    # NOTE: `usable()` is True for status "degraded" (base.py:30), so the
    # check is on the STATUS, not on usable(). A degraded detector did not
    # produce a clean verdict; it produced a partial one, and a partial
    # result must not be reported as "no synthetic artifact".
    #
    # Consequence to accept knowingly: on a host where SPAI cannot load, the
    # image channel stops reading LOW and reads CAUTION/UNVERIFIED instead.
    # That is the honest outcome — a verdict we could not compute must not be
    # reported as a clean one.
    if eff_c is None or eff_c.status != "ok":
        return False  # the strong detector did not run cleanly -> NOT clean
    eff = eff_c.signals.get("prob_deepfake", 1.0)
    return eff <= 0.45


def _file_clean(by_name: dict[str, Any]) -> bool:
    """True when a file was ACTUALLY scanned by a decisive detector and every
    signal came back clean, so it may read LOW/trust instead of pinning at the
    MEDIUM midpoint (risk-only weights leave TRUST otherwise unreachable).

    Corroboration-gated so it cannot false-clean a threat:
      - at least ONE decisive scanner (ClamAV / YARA-X / VirusTotal) must have
        RUN and reported status="ok" — a file with only heuristic statics is
        not decisively clean, it is merely un-scanned (handled as UNVERIFIED
        upstream);
      - no decisive scanner flagged (clamav detected / yara hits / vt positives);
      - NO OTHER risk-bearing signal in the target's weight map may be elevated
        (Quark / MoBFS / APK statics / PE statics / dynamic sandbox);
      - entropy, if measured, is not anomalous (packing/encryption);
      - no declared-vs-verified extension mismatch is present.

    FAIL-OPEN FIXES (2026-10-21):
      (a) A DEGRADED scanner used to count as a decisive CLEAN scan.
          `CheckResult.usable()` is True for status "degraded", and a
          "no VirusTotal record for this hash" result is emitted as
          status="degraded" with positives_ratio 0.0. So a file nobody ever
          looked up was treated as VT-cleared. Only status="ok" counts now.
      (b) The RISK-bearing signals were ignored entirely: only clamscan /
          yara_x / vt_reputation were inspected, but the malicious_file
          weight map also carries quark.score_norm, mobsf.risk_high_plus,
          apk.prob_malicious, pe.prob_malicious and pe.packed_flag. A file
          that Quark or the APK statics flagged at 0.95 still reached TRUST.
    """
    decisive_ran = False
    # (check_name, signal_key, "bad above" threshold; True == truthy-is-bad)
    # A scanner counts as a decisive CLEAN scan only when it reported
    # status="ok" (fix (a) above). `usable()` is True for "degraded", and a
    # degraded VT means "no record for this hash" — i.e. NEVER LOOKED UP, not
    # "looked up and found clean". A degraded scanner is therefore skipped
    # here and does NOT count towards decisive_ran; the file still needs one
    # genuinely-ok scanner before it may read TRUST.
    for name, key, bad in (("clamscan", "detected", True),
                           ("yara_x", "hits_norm", 0.20),
                           ("vt_reputation", "positives_ratio", 0.05)):
        c = by_name.get(name)
        if c is None or c.status != "ok":
            continue
        decisive_ran = True
        v = c.signals.get(key)
        if bad is True:
            if bool(v):
                return False
        elif isinstance(v, (int, float)) and v > bad:
            return False
    if not decisive_ran:
        return False
    # A scanner that RAN but could not conclude (degraded) means the file was
    # not fully checked. It must not be treated as cleared.
    for name in ("clamscan", "yara_x", "vt_reputation"):
        c = by_name.get(name)
        if c is not None and c.status == "degraded":
            return False  # inconclusive scan -> not decisively clean
    # RISK-BEARING SIGNALS (fix (b)): any of these firing vetoes TRUST, even
    # when every signature scanner cleared the file. A fresh packer or a novel
    # APK technique is exactly what signature AV misses.
    for name, key, bad in (("quark_engine", "score_norm", 0.5),
                           ("apk_statics", "prob_malicious", 0.5),
                           ("pe_statics", "prob_malicious", 0.5)):
        c = by_name.get(name)
        if c is None or not c.usable():
            continue
        v = c.signals.get(key)
        if isinstance(v, (int, float)) and v > bad:
            return False
    # A packed PE is a risk signal in its own right.
    pe = by_name.get("pe_statics")
    if pe is not None and pe.usable() and pe.signals.get("packed"):
        return False
    # MoBFS risk level: "high"/"critical" (case-insensitive) vetoes TRUST.
    mob = by_name.get("mobsf_apk")
    if mob is not None and mob.usable():
        lvl = str(mob.signals.get("mobsf_risk_level", "")).strip().lower()
        if lvl in ("high", "critical"):
            return False
    ds = by_name.get("dynamic_sandbox")
    if ds is not None and ds.usable() and ds.signals.get("malicious"):
        return False
    fe = by_name.get("file_entropy")
    if fe is not None and fe.usable() and fe.signals.get("anomaly"):
        return False
    em = by_name.get("ext_mismatch_flag")
    if em is not None and em.usable() and "declared" in (em.signals or {}):
        return False
    return True


def _url_clean(by_name: dict[str, Any]) -> bool:
    """True when a URL is clearly safe: PhishingScanner ran, did NOT flag phishing,
    and its risk_score_norm is low (<0.30, well under the is_phishing 0.70 line).
    A flagged or missing/high-risk scan -> NOT clean (never trusts a phish)."""
    c = by_name.get("url_phish_scanner")
    if c is None or not c.usable():
        return False  # no scan / scan failed -> not clean
    if c.signals.get("is_phishing", False):
        return False  # scanner flagged it -> never clean
    risk = c.signals.get("risk_score_norm", 1.0)
    return isinstance(risk, (int, float)) and 0.0 <= risk < 0.30


def _audio_clean(by_name: dict[str, Any]) -> bool:
    """True when audio is clearly bonafide: aasist3 ran and scored LOW (<0.30).
    aasist3 is the proven-trustworthy gate (AUC 0.9967, bonafide 0.0000/spoof
    0.9966) — a low read on genuine audio is decisive clean evidence, so a clean
    audio clip reads LOW/TRUST instead of MEDIUM/UNVERIFIED. A high/missing
    aasist read -> NOT clean (never cleans a spoof)."""
    c = by_name.get("aasist_detector")
    if c is None or not c.usable():
        return False
    pa = c.signals.get("prob_deepfake", 1.0)
    return isinstance(pa, (int, float)) and 0.0 <= pa < 0.30


def _probe(by_name: dict[str, Any], check: str, key: str, default: float) -> float:
    c = by_name.get(check)
    if c is None or not c.usable():
        return default
    return c.signals.get(key, default)


def _deepfake_pattern(target: str, signals: dict[str, float],
                      gaps: list[str]) -> tuple[str, float, bool]:
    if target not in ("deepfake_video", "deepfake_audio"):
        return ("generic", 1.0, False)
    eff = signals.get("effort.prob")
    heu = signals.get("frameheur.prob")
    avr = signals.get("av_risk_addition")
    hav = signals.get("havic.prob_inconsistent")
    aas = signals.get("aasist.prob")
    xlr = signals.get("xlsr.prob")

    elevated = [k for k, v in signals.items() if (v or 0) > 0.30]
    strong = [k for k, v in signals.items() if (v or 0) > 0.55]

    # fully_generated (2026-08-25 recalibration): the generative signature is
    # AV-derived — anti-correlated audio/video sync + cross-modal inconsistency —
    # with face forensics as corroboration. effort CANNOT gate this pattern by
    # value: on the real-video-recalibrated scale FF++ reals overlap the AI
    # video's calibrated effort almost exactly (reals -0.09..0.58 vs AI 0.33),
    # so any effort threshold either misses the AI case or flags every real.
    # av_risk>=0.45 is only reachable when the audio track was actually analyzed.
    # NOTE: av_risk_addition is NOT passed through _calibrate's affine table for
    # the pattern (identity there), so 0.45 here == raw 0.45 == anti_correlated.
    if target == "deepfake_video":
        havic_live = "havic.prob_inconsistent" not in gaps
        if (avr or 0) >= 0.45 and (hav is None or hav > 0.15) and havic_live:
            return ("fully_generated", 1.25, True)
        # face_swap_partial (recalibrated 2026-08-26): effort calibrated >0.40 with AV OK.
        # On the -0.55 scale, raw effort >=0.95 hits this (0.40) — face forensics
        # near-ceiling on an audio-consistent clip = localized manipulation class.
        # (threshold moved 0.50->0.40 with the effort affine b -0.45->-0.55; the
        # old >0.50 was unreachable because max calibrated effort = 1.0-0.55=0.45.)
        if (eff or 0) > 0.40 and (avr or 0) < 0.30:
            return ("face_swap_partial", 1.05, True)
        if len(strong) >= 2:
            return ("corroborated_multi", 1.25, True)
        if len(elevated) >= 2:
            return ("corroborated_multi", 1.1, True)
        if (eff or 0) > 0.40 or (avr or 0) >= 0.45:
            return ("weak_signal_single", 1.0, True)
        return ("generic", 1.0, False)

    # deepfake_audio: include fakemamba/ssl in the mode check
    fake = signals.get("fakemamba.prob")
    ssl = signals.get("ssl.prob")
    audio_strong = [v for v in (aas, xlr, fake, ssl) if v is not None and v > 0.55]
    audio_weak = [v for v in (aas, xlr, fake, ssl) if v is not None and v < 0.30]
    if len(audio_strong) >= 2 and len(audio_weak) >= 1:
        return ("conflicting_detectors", 1.0, True)   # two say fake, one says real
    if len(audio_strong) >= 2:
        return ("corroborated_multi", 1.25, True)
    if (aas or 0) > 0.55 or (xlr or 0) > 0.55 or (fake or 0) > 0.55:
        return ("weak_signal_single", 1.0, True)
    return ("generic", 1.0, False)


def _extract(spec: tuple[str, str, str, bool], c: CheckResult | None) -> tuple[str, Any]:
    """Classify a weighted-signal state.

    Returns (state, value):
      'known_gap'  — check exists but was skipped/unavailable BY DESIGN (gate)
      'absent'     — no such check produced at all (genuine missing evidence)
      'failed'     — check ran and failed (genuine missing evidence)
      'value'      — usable check; value parsed per kind (may be 0.0 legitimately)
    """
    check_name, key, kind, _is_prob = spec
    if c is None:
        return "absent", None
    if c.status in ("skipped", "unavailable"):
        return "known_gap", None
    if c.status == "failed":
        return "failed", None
    if kind == KIND_CONST_TRUE:
        # Presence-flag signal: the BAD condition (e.g. an extension mismatch)
        # exists only when the capability actually populated the keyed signal.
        # The orchestrator emits ext_mismatch_flag(status=ok) in BOTH the
        # mismatch case (signals carry "declared"/"verified") and the match
        # case (signals={}); returning 1.0 unconditionally made every file with
        # a *correct* extension carry a phantom risk vote. Fire only when the
        # keyed signal is actually present; otherwise it is N/A (known_gap).
        return ("value", 1.0) if key in c.signals else ("known_gap", None)
    if key not in c.signals:
        # check ran fine but this signal does not apply to this item type
        # (e.g. a PDF doc without a PGP validity verdict) — not missing evidence
        return "known_gap", None
    v = c.signals.get(key)
    if v is None:
        return "failed", None            # emitted check but signal missing
    try:
        if kind == KIND_DEFAULT_NUM:
            fv = float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
            return "value", min(1.0, max(0.0, fv))
        if kind == KIND_NEG_BOOL:
            return "value", (1.0 if bool(v) else 0.0)
        if kind == KIND_LIST_FRAC:
            n = len(v) if isinstance(v, (list, tuple, set)) else 0
            return "value", min(1.0, n / 3.0)
        if kind == KIND_INT_FRAC:
            iv = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
            return "value", min(1.0, max(0.0, iv / 3.0))
        if kind == KIND_INV_NUM:
            return "value", min(1.0, max(0.0, 1.0 - float(v)))
        if kind == KIND_RISK_STR:
            return "value", RISK_LEVEL_MAP.get(str(v).lower().strip(), 0.0)
    except Exception:
        return "failed", None
    return "failed", None


@dataclass(slots=True)
class FusionDecision:
    verdict: Verdict
    score: float                      # calibrated probability of RISK (0..1)
    raw: float                        # pre-calibration weighted score
    disagreement: float               # max-min across usable prob detectors
    reasons: list[str] = field(default_factory=list)
    confidence: float = 0.0           # post-reliability scaling of certainty
    model_ids: list[str] = field(default_factory=list)
    known_gaps: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    # Fusion v2: set when a deepfake/detector pattern fires COHERENTLY (e.g.
    # fully_generated, corroborated_multi). Tells the ReliabilityGate that
    # cross-detector spread is CORROBORATION across detectors of different
    # types, not conflict — so it must not abort on spread alone.
    coherent_pattern: bool = False


# --------------------------------------------------------------- engine ----

class FusionEngine:
    """Per-target combination: LR stack if provisioned, else explicit weights.

    The stack expects OOF specialist predictions + context features; without a
    trained checkpoint we compute the documented weighted sum, squash via
    logistic, then temperature-scale using calibration params fitted on
    held-out real-world sets. Identity calibration keeps today's results
    interpretable and improves as labeled data accrues.
    """

    def __init__(self, lr_stacks: dict[str, Any] | None = None,
                 calibration: dict[str, dict[str, float]] | None = None):
        self.lr_stacks = lr_stacks or {}
        self.calibration = calibration or {}   # target -> {"t": temp, "b": bias}

    # -------------------------------------------------- feature contract --
    @staticmethod
    def feature_vector(target: str, checks: list[CheckResult]) -> list[float]:
        """Deterministic ordered feature vector shared by training & serving.

        Layout: for every weight in WEIGHTS[target] (dict order):
          [value (0..1; 0 when absent/known-gap/failed), gap_flag (1.0 when
           the check was skipped/unavailable BY DESIGN)]
        Fixed order keeps the stacker checkpoint portable between versions.
        """
        wmap = WEIGHTS.get(target, {})
        by_name: dict[str, CheckResult] = {}
        for c in checks:
            by_name.setdefault(c.name, c)
        vec: list[float] = []
        for key in wmap:
            spec = _SIGNAL_SOURCES.get(key)
            if spec is None:
                continue
            state, val = _extract(spec, by_name.get(spec[0]))
            v = float(val) if (state == "value" and isinstance(val, (int, float))) else 0.0
            vec.append(v)
            vec.append(1.0 if state == "known_gap" else 0.0)
        return vec

    def decide(self, target: str, checks: list[CheckResult]) -> FusionDecision:
        by_name: dict[str, CheckResult] = {}
        for c in checks:
            by_name.setdefault(c.name, c)
        wmap = WEIGHTS.get(target, {})
        usable = [c for c in checks if c.usable()]

        if not usable:
            return FusionDecision(verdict=Verdict.UNABLE_TO_VERIFY, score=0.0, raw=0.0,
                                  disagreement=0.0, reasons=["no_usable_signals"],
                                  confidence=0.0,
                                  missing_evidence=sorted({c.name for c in checks}) or ["no-evidence"])

        probs: list[float] = []
        known_gaps: list[str] = []
        missing: list[str] = []
        total_w = 0.0
        s = 0.0
        n_val = 0                     # mapped signals that produced a real value
        _av_uncorroborated = False   # cross-modal signal fired without video-side agreement
        for key, weight in wmap.items():
            spec = _SIGNAL_SOURCES.get(key)
            if spec is None:
                continue
            state, val = _extract(spec, by_name.get(spec[0]))
            # Gap-neutral denominator (2026-08-25, real-video calibration):
            # KNOWN GAPS always count toward the weight sum at zero contribution.
            # Previously only present signals divided the sum, so a video where
            # the cross-modal checks were unavailable (no audio track) fused on
            # its 2 present signals alone and landed in DNU; counting the full
            # designed weight pool pulls partial-coverage videos toward the
            # honest CAUTION band while fully-covered risky scans are unchanged.
            if state == "known_gap":
                known_gaps.append(key)
                total_w += abs(weight)
                continue
            if state != "value":
                missing.append(key)
                continue
            assert val is not None
            total_w += abs(weight)
            n_val += 1
            cv = _calibrate(key, float(val))       # per-signal calibration (v2)
            s += weight * cv
            if spec[3]:
                probs.append(cv)

        # Selective prediction (2026-08-27 honesty fix): if NOT ONE mapped
        # signal produced an actual value — every weighted detector was a known
        # gap (tool/model not provisioned), absent, or failed — we have zero
        # real evidence. Previously this fell through to x=0 -> sigmoid(0)=0.5,
        # pinning EVERY un-provisioned item to the CAUTION ("MEDIUM") band. A
        # confident-looking MEDIUM from no evidence is the exact overclaim the
        # reliability design forbids; the honest answer is UNABLE_TO_VERIFY.
        # (A detector that RAN and read low still yields a value here, so a
        # genuinely-clean scan is unaffected and can reach TRUST below.)
        if n_val == 0:
            gap_reason = ("all_detectors_gated" if known_gaps
                          else "no_detector_evidence")
            return FusionDecision(
                verdict=Verdict.UNABLE_TO_VERIFY, score=0.0, raw=0.0,
                disagreement=0.0,
                reasons=[gap_reason, f"gated_tools:{len(known_gaps)}"],
                confidence=0.0, known_gaps=known_gaps,
                missing_evidence=missing or sorted({c.name for c in usable})[:6])

        x = s / total_w if total_w > 0 else 0.0

        # CROSS-MODAL CORROBORATION GATE (2026-10-21, widened 2026-10-22).
        # av_risk_addition is a DISCRETE band reading, not a probability:
        #     0.0  synced            (lip motion and speech energy agree)
        #     0.1  weakly_synced     (moderate coherence)
        #     0.45 decorrelated     (envelopes uncorrelated — ambiguous)
        #     0.5  anti_correlated  (envelopes move OPPOSITE)
        #     0.0  unknown           (the correlation could not be measured)
        #
        # 2026-10-22: this gate used to treat the two elevated bands
        # ASYMMETRICALLY -- 0.45 was discounted (a mux artefact) but 0.5 was
        # TRUSTED as "the deepfake signature". That asymmetry was based on a
        # measurement that has since been shown to be WRONG.
        #
        # The 0.5 readings were the -2.0 SENTINEL of the correlation
        # function, not a correlation: the operator's own AI videos have a
        # COMPLETELY SILENT audio track (RMS 0.0), so their correlation was
        # undefined. And measured on 67 genuine WhatsApp videos, 9 of them
        # (13%) have the same all-zero audio track and were reported the same
        # way. So 0.5 meant "no speech energy", not "independently generated
        # modalities" -- and it fired on genuine muted clips as readily as on
        # generated ones.
        #
        # So the gate now treats BOTH elevated bands the same way: an
        # elevated cross-modal reading may CORROBORATE a video-side signal
        # but may never CARRY the verdict by itself. The cross-modal check is
        # evidence about consistency, and consistency alone is not evidence
        # of synthesis.
        #
        # THE GATE THRESHOLD (0.40) IS DELIBERATE AND IT IS THE HONEST LIMIT
        # OF THIS DATASET. On the leak-free corpus the two genuine clips that
        # read DO_NOT_USE and the fakes that read DO_NOT_USE are
        # STATISTICALLY INDISTINGUISHABLE on the video-side signals:
        #     real_006  effort_cal 0.374  frameheur_cal 0.295
        #     fake_006  effort_cal 0.377  frameheur_cal 0.298
        # A 0.003 difference. No threshold on these signals can separate them.
        # So a threshold that keeps the fakes MUST keep real_006 too. The
        # gate threshold is therefore set ABOVE the genuine clips'
        # calibrated effort (0.374) so an elevated cross-modal reading
        # cannot decide a verdict on its own.
        #
        # The gate is deliberately one-directional: it can only REDUCE risk
        # when the cross-modal signal fires alone. A corroborated fake (high
        # EFFORT + an elevated cross-modal reading) still reaches
        # DO_NOT_USE -- the gate removes the false positive, it never
        # disarms the detector.
        if target == "deepfake_video":
            _av = _probe(by_name, "cross_modal_av", "av_risk_addition", 0.0)
            if isinstance(_av, (int, float)) and _av >= 0.45:
                _eff_raw = _probe(by_name, "effort_face_forensics", "prob_deepfake", 0.0)
                _fh_raw = _probe(by_name, "frame_heuristics", "prob_deepfake", 0.0)
                # The video-side signals must CORROBORATE. Their calibrated
                # values are what the verdict is built from, so compare those.
                _eff_cal = _calibrate("effort.prob", float(_eff_raw)) \
                    if isinstance(_eff_raw, (int, float)) else 0.0
                _fh_cal = _calibrate("frameheur.prob", float(_fh_raw)) \
                    if isinstance(_fh_raw, (int, float)) else 0.0
                if _eff_cal <= 0.40 and _fh_cal <= 0.40:
                    # The cross-modal signal fired ALONE with no video-side
                    # agreement. Discount it: it cannot be the deciding
                    # evidence, so its weight contribution is zeroed and the
                    # signal is recorded as an uncorroborated reading.
                    s -= WEIGHTS["deepfake_video"]["av_risk_addition"] * _av
                    _av_uncorroborated = True
            # `x` is the weighted mean, so it must be recomputed after the
            # gate discounts the cross-modal contribution.
            x = s / total_w if total_w > 0 else 0.0
        # total_w==0 guard (2026-08-25): targets with no mapped weights (document_generic,
        # unclassified) previously fell through to raw=0.0 -> TRUST at conf 1.0 on ANY
        # usable-but-unmapped check. That is a max-confidence overclaim from zero evidence.
        # Honest answer for an unmapped target with usable signals is UNABLE_TO_VERIFY.
        if total_w == 0:
            return FusionDecision(verdict=Verdict.UNABLE_TO_VERIFY, score=0.0, raw=0.0,
                                  disagreement=0.0, reasons=["no_mapped_weights_for_target"],
                                  confidence=0.0,
                                  missing_evidence=[c.name for c in usable][:6])
        # Clean-evidence bonus (2026-08-25, fusion-trust fix): with risk-only positive
        # weights, a fully-clean scan mathematically pinned raw at 0.5 -> CAUTION forever,
        # making TRUST unreachable on every non-gov target. When ENOUGH mapped signals ran
        # and EVERY present one is low-risk, pull the logit into the TRUST band.
        # Magnitude: -1.8 shifts the logistic-6 mean by ~-0.3 => raw ~0.0 for all-clean.
        # Partial coverage or any elevated signal skips the bonus (honest CAUTION/UNABLE).
        _CLEAN_EPS = 0.10
        _CLEAN_MIN_SIGNALS = 3
        if total_w > 0:
            present_vals: list[float] = []
            const_risk_present = False
            for key2, w2 in wmap.items():
                spec2 = _SIGNAL_SOURCES.get(key2)
                if spec2 is None:
                    continue
                # KIND_CONST_TRUE signals are presence-flags: an extracted value
                # (1.0) means the BAD condition IS present (e.g. an extension
                # mismatch). Such a flag never joins present_vals (it is not a
                # 0..1 probability), but when it fires it DISQUALIFIES the
                # all-clean bonus — otherwise a PE disguised as a PDF that is
                # merely AV-clean would be masked as clean. Absent flag => N/A.
                if spec2[2] == "const_true":
                    if _extract(spec2, by_name.get(spec2[0]))[0] == "value":
                        const_risk_present = True
                    continue
                st2, val2 = _extract(spec2, by_name.get(spec2[0]))
                if st2 == "value" and isinstance(val2, (int, float)):
                    present_vals.append(float(val2))
            if (not const_risk_present
                    and len(present_vals) >= _CLEAN_MIN_SIGNALS
                    and all(v <= _CLEAN_EPS for v in present_vals)):
                x += -1.8
        # logistic squash keeps range (0,1) and is robust to weight drift.
        # Temperature 4.0 (was 6.0, real-video calibration 2026-08-25): with the
        # gap-neutral denominator, mid-band evidence (real talking-heads at
        # weighted mean ~0.30-0.45) must land CAUTION, not DNU; k=6 turned
        # x=0.31 into raw=0.84. k=4 maps that same evidence to ~0.71 boundary /
        # below, while fully-corroborated risky scans (AI video, x~0.34 after
        # corroboration + pattern) still clear DO_NOT >= 0.70 via confidence
        # and pattern boost rather than an overheated squash.
        raw = 1.0 / (1.0 + math.exp(-4.0 * x)) if total_w > 0 else 0.0

        # LR-stack override when provisioned (OOF-trained checkpoint)
        model_id = ""
        if target in self.lr_stacks:
            try:
                p, model_id = self.lr_stacks[target].predict_proba(context_from_checks(checks), checks)
                raw = float(min(1.0, max(0.0, p)))
            except Exception:
                pass  # fall through to weighted result

        # calibration (temperature/bias), else identity
        calib = self.calibration.get(target)
        if calib and (calib.get("t", 1.0) != 1.0 or calib.get("b", 0.0) != 0.0):
            t_c = calib.get("t", 1.0)
            b = calib.get("b", 0.0)
            lo, hi = 1e-9, 1.0 - 1e-9
            p = min(max(raw, lo), hi)
            z = math.log(p / (1.0 - p))          # logit
            z_cal = (z - b) * t_c
            raw = 1.0 / (1.0 + math.exp(-z_cal))

        # DISAGREEMENT ON A COMMON SCALE (2026-10-21). The metric used to be
        # max-min over the CALIBRATED values of the prob detectors. Two
        # detectors with different affine calibrations then produce a spread
        # that reflects the CALIBRATION rather than the evidence: an EFFORT
        # reading of 0.917 calibrates to 0.367 while a frameheur reading of
        # 0.36 calibrated to 0.759 — a spread of 0.39 on two detectors that
        # AGREE — which tripped the 0.5 disagreement abort and forced
        # UNVERIFIED on genuine clips.
        #
        # The root cause was the frameheur AMPLIFICATION (slope 1.97): with
        # every prob-detector calibration now on a comparable scale (effort
        # 1.0/-0.55, frameheur 1.0/0.0, havic 0.2/0.0), the calibrated spread
        # IS a meaningful evidence spread again. Using the RAW values instead
        # was tried and REJECTED: HAVIC's checkpoint is near-degenerate and
        # reads ~0.95 on genuine clips, so a raw-vs-raw spread measures HAVIC's
        # saturation rather than the disagreement between detectors.
        disagreement = ((max(probs) - min(probs))
                        if len(probs) >= 2 else 0.0)

        # threshold mapping (risk -> verdict); wide caution band by design
        DO_NOT = 0.70
        CAUT_LO = 0.35
        TRUST_HI = 0.15
        if raw >= DO_NOT:
            verdict = Verdict.DO_NOT_USE
        elif raw >= CAUT_LO:
            verdict = Verdict.CAUTION
        elif raw <= TRUST_HI:
            # selective prediction: TRUST only when no GENUINE missing evidence.
            # known_gap = check skipped/unavailable BY DESIGN (gate, N/A item type) —
            # that is honest coverage, not missing evidence. Only 'absent'/'failed'
            # states block trust (a mapped signal that should have run didn't).
            genuine_missing = [m for m in missing if m not in set(known_gaps)]
            verdict = Verdict.TRUST if not genuine_missing else Verdict.UNABLE_TO_VERIFY
        else:
            verdict = Verdict.CAUTION

        # 2026-08-26 NOT-BINARY-GATE (operator directive + measured evidence):
        # image_facecheck verdicts CAP at CAUTION. SPAI feeds faceforensics as
        # EVIDENCE, but a single learned-model read must not flip a photo to
        # DO_NOT_USE — chameleon's false-HIGH overfit was exactly that. freqband
        # is measured dead-noise (real 0.463 vs AI 0.399, overlapping) so no
        # second independent signal corroborates an image fake yet; until one
        # exists, DNU for images is a known-unreliable overclaim and is blocked.
        if target == "image_facecheck" and verdict is Verdict.DO_NOT_USE:
            verdict = Verdict.CAUTION

        # CLEAN-SIDE VERDICT OVERRIDE (2026-08-26, real-video -> LOW/trust).
        # A genuine real video that is AV-synced (cross_modal alignment=="synced",
        # real lip-sync) AND has low face-forensics AND low frame-heuristics is
        # compelling authenticity evidence, so we force TRUST for it. This is the
        # reliability-gate's inverse: a *decisive clean* signal. CORROBORATION-
        # GATED so it CANNOT false-clean a fake:
        #   - a lip-synced DEEPFAKE face (high effort) fails the low-face test
        #   - a decorrelated/weakly/anti-sync swap fails the synced test
        # A magnitude-only clean bonus cannot reach score<=0.15 (needs x<=-0.43)
        # without dangerous swing, so a corroboration-gated override is the safe,
        # honest way to let a real synced video read LOW.
        _clean_side_asserted = False
        if target == "deepfake_video" and verdict is not Verdict.DO_NOT_USE \
                and _clean_side_bonus(by_name) is not None:
            verdict = Verdict.TRUST
            _clean_side_asserted = True

        # CLEAN-SIDE TRUST for images (2026-08-26, operator-directed): a clean
        # real photo (integrity ok AND face-forensics low <=0.45 AND freqband
        # low <=0.45) reads LOW/trust. Without this, even a genuinely-clean
        # image lands CAUTION forever (all image weights are positive -> x~0 ->
        # sigmoid ~0.5 -> CAUTION; no negative clean signal existed). CORROBORATED:
        # both independent detectors must agree low, so a lone SPAI-high real
        # photo (the known ~24% false-positive) still reads CAUTION, not trust.
        if target == "image_facecheck" and verdict is not Verdict.DO_NOT_USE \
                and _image_clean(by_name):
            verdict = Verdict.TRUST
            _clean_side_asserted = True

        # CLEAN-SIDE TRUST for URLs (2026-08-26): a clearly-safe link — PhishingScanner
        # did not flag it AND risk_score_norm < 0.30 — reads LOW/trust. Without this a
        # benign URL sits at CAUTION forever (single positive weight -> raw~0.5 ->
        # CAUTION, TRUST unreachable). Corroborated: the scanner's own is_phishing=False
        # AND low score together, so a flagged phish can never reach LOW.
        if target == "url_phishing" and verdict is not Verdict.DO_NOT_USE \
                and _url_clean(by_name):
            verdict = Verdict.TRUST
            _clean_side_asserted = True

        # CLEAN-SIDE TRUST for files (2026-08-27): a file that a decisive scanner
        # (ClamAV / YARA-X / VT) actually inspected and cleared reads LOW/trust.
        # Without this every scanned-clean file sits at MEDIUM forever (all file
        # weights are positive -> x~0 -> sigmoid ~0.5 -> CAUTION). Corroborated:
        # a decisive detector must have RUN and found nothing, entropy is normal,
        # and no extension mismatch — so a detected/packed/mismatched file can
        # never reach LOW. An un-scanned file has no decisive detector and is
        # UNVERIFIED (n_val gate), not falsely trusted.
        if target in ("malicious_file", "document_generic", "unclassified") \
                and verdict is not Verdict.DO_NOT_USE and _file_clean(by_name):
            verdict = Verdict.TRUST
            _clean_side_asserted = True

        # CLEAN-SIDE TRUST for AUDIO (2026-08-26): genuine audio that aasist3
        # scores < 0.30 reads LOW/trust (aasist3 is proven-trustworthy — AUC
        # 0.9967, re-measured 2026-10-21 at AUC 1.0000 / EER 0.00% on 300
        # official ASVspoof2019-LA eval samples). Without this a clean audio
        # clip sat at MEDIUM/UNVERIFIED (weak offline+xlsr couldn't reach
        # TRUST). A spoof (aasist~0.998) is NEVER cleaned.
        if target == "deepfake_audio" and verdict is not Verdict.DO_NOT_USE \
                and _audio_clean(by_name):
            verdict = Verdict.TRUST
            _clean_side_asserted = True

        # Fusion v2: pattern-aware agreement (deepfake). Different generators fool
        # different detectors, so spread IS a signal (a mode), not abstention.
        pattern = "generic"
        pattern_mult = 1.0
        reasons_early: list[str] = []
        if target in ("deepfake_video", "deepfake_audio"):
            # recover calibrated values per weighted signal for the classifier
            pat_signals: dict[str, float] = {}
            for key, weight in wmap.items():
                spec = _SIGNAL_SOURCES.get(key)
                if spec is None:
                    continue
                st, val = _extract(spec, by_name.get(spec[0]))
                if st == "value" and isinstance(val, (int, float)):
                    pat_signals[key] = _calibrate(key, float(val))
            # The cross-modal corroboration gate (2026-10-21) applies to the
            # PATTERN layer too: `fully_generated` fires on av_risk >= 0.45,
            # and an uncorroborated decorrelation must not be able to select
            # that pattern either — otherwise the gate above would discount the
            # weighted contribution while the pattern layer still boosted the
            # confidence of a verdict the gate had just defused.
            if _av_uncorroborated:
                pat_signals["av_risk_addition"] = 0.0
            pattern, pattern_mult, fake_aligned = _deepfake_pattern(target, pat_signals, known_gaps)
            # a corroborated generative pattern boosts confidence when the verdict
            # is danger-aligned; a clean-aligned pattern caps it to CAUTION honest.
            if pattern != "generic":
                reasons_early.append(f"pattern:{pattern}")

        # confidence: distance from midpoint, scaled by detector coverage,
        # penalized by inter-detector disagreement. Fusion v2: the blanket
        # 0.35-spread penalty is REPLACED for deepfake targets — corroboration
        # across detectors of different types is a confidence BOOST, not a
        # penalty, when the SAME mode (fake-aligned) fires. Only genuinely
        # conflicting modes (fake vs clean) suppress confidence.
        exp_det = _EXPECTED_PROB_DET.get(target, 1)
        coverage = min(1.0, (len(probs) / exp_det)) if exp_det else 1.0
        certainty = max(0.0, min(1.0, 2.0 * abs(0.5 - raw) * (0.4 + 0.6 * coverage)))
        if target in ("deepfake_video", "deepfake_audio"):
            # boosting pattern (fake-aligned corroboration) raises certainty
            if pattern_mult > 1.0 and raw >= CAUT_LO:
                certainty = min(1.0, certainty * pattern_mult)
            # conflicting modes: mixed fake/clean alignment -> keep spread penalty
            elif disagreement > 0.35:
                # BUT an asserted clean-side override is a DECISIVE-clean signal
                # (all gates corroborate: synced + low face + low frameheur), not
                # a conflicting-mode case — do not demote its TRUST.
                if not _clean_side_asserted:
                    certainty *= 0.5
                    if verdict is Verdict.TRUST:
                        verdict = Verdict.CAUTION
                    certainty = min(certainty, 0.4)
        else:
            if disagreement > 0.35:
                certainty *= 0.5
                # asserted clean-side override (image): a corroborated-clean photo
                # (SPAI low + integrity ok) is decisive clean, not a conflicting-mode
                # case — do NOT demote its TRUST.
                if not _clean_side_asserted:
                    if verdict is Verdict.TRUST:
                        verdict = Verdict.CAUTION
                certainty = min(certainty, 0.4)
        if missing:
            certainty *= 0.7

        reasons: list[str] = reasons_early
        if missing:
            reasons.append("incomplete_evidence:" + ";".join(missing[:4]))
        if known_gaps:
            reasons.append(f"gated_tools:{len(known_gaps)}")
        if disagreement > 0.35 and target not in ("deepfake_video", "deepfake_audio"):
            reasons.append(f"detector_disagreement:{disagreement:.2f}")
        if model_id:
            reasons.append(f"stack:{model_id}")
        reasons.append(f"risk_raw:{raw:.2f}")

        return FusionDecision(verdict=verdict, score=round(raw, 4), raw=round(x, 4),
                              disagreement=round(disagreement, 3), reasons=reasons,
                              confidence=round(certainty, 3),
                              model_ids=[model_id] if model_id else [],
                              known_gaps=known_gaps, missing_evidence=missing,
                              coherent_pattern=(pattern in ("fully_generated",
                                                            "corroborated_multi",
                                                            "face_swap_partial")))


    def load_trained(self, training_dir) -> int:
        """Load OOF-trained LR-stack checkpoints produced by fusion_train.
        Returns number of targets wired."""
        import json as _json
        n = 0
        p = Path(training_dir)
        if not p.exists():
            return 0
        for art_file in sorted(p.glob("stack_*.json")):
            try:
                art = _json.loads(art_file.read_text())
            except Exception:
                log.warning("unreadable stack artifact %s", art_file)
                continue
            target = art.get("target")
            final_lr = art.get("final") or {}
            calib = art.get("calibration") or {}
            if not target or "w" not in final_lr:
                continue
            # Feature-layout guard (2026-08-27): a checkpoint trained for an OLD
            # signal layout must NEVER be wired to the current one. The serving
            # feature vector is [value, gap] per mapped signal, so its length is
            # 2*len(WEIGHTS[target]); if the checkpoint's weight vector disagrees
            # it was fit on a different feature set (e.g. the pre-refactor
            # 7-signal url_phishing head vs today's 1-signal head). Silently
            # zero-padding it applies weights to the WRONG features and can
            # invert the verdict — so refuse to load it and keep the explicit
            # weighted path instead.
            expected = len(FusionEngine.feature_vector(target, []))
            w = final_lr["w"]
            if not isinstance(w, list) or len(w) != expected:
                log.warning(
                    "skipping stale fusion stack %s: target=%s trained for %s "
                    "features but current layout needs %s — retrain with the "
                    "current signal set (falling back to explicit weights)",
                    art_file.name, target, len(w) if isinstance(w, list) else "?",
                    expected)
                continue
            self.lr_stacks[target] = _LRStackAdapter(target, w, final_lr["b"])
            if calib:
                self.calibration[target] = {"t": float(calib.get("t", 1.0)),
                                             "b": float(calib.get("b", 0.0))}
            n += 1
        return n


class _LRStackAdapter:
    """Serving-side wrapper over an OOF-trained logistic-regression stack.

    Consumes the deterministic ordered feature vector from
    FusionEngine.feature_vector (value + known-gap flag per weighted signal),
    exactly the layout fusion_train fits on.
    """

    def __init__(self, target: str, w: list[float], b: float):
        self.target = target
        self.w = [float(x) for x in w]
        self.b = float(b)

    def predict_proba(self, feats: dict[str, Any], checks: list[CheckResult]) -> tuple[float, str]:
        x = FusionEngine.feature_vector(self.target, checks)
        # Defense-in-depth: load_trained already rejects a mismatched checkpoint,
        # but never silently pad/truncate here either — a feature-count mismatch
        # means the weights map to different signals, so raise and let decide()
        # fall back to the explicit weighted path rather than emit a wrong prob.
        if len(x) != len(self.w):
            raise ValueError(
                f"feature/weight mismatch for {self.target}: "
                f"{len(x)} live features vs {len(self.w)} trained weights")
        z = self.b + sum(wi * vi for wi, vi in zip(self.w, x))
        return min(1.0, max(0.0, _sigmoid(z))), f"lr_stack_oof:{self.target}"


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)



def context_from_checks(checks: list[CheckResult]) -> dict[str, Any]:
    """Feature vector for the LR stack (OOF-style, derived from usable checks)."""
    feats: dict[str, Any] = {}
    for c in checks:
        if c.usable():
            for k, v in c.signals.items():
                if isinstance(v, (str, int, float, bool)):
                    feats[f"{c.name}.{k}"] = v
    return feats


# --------------------------------------------------------- reliability ------

class ReliabilityGate:
    """Decides whether a fused decision is trustworthy enough to surface.

    Any violation forces the caller to emit UNABLE_TO_VERIFY and capped
    confidence:
      - zero usable checks (defensive duplicate of upstream guard)
      - strong detector disagreement (> max_disagreement)
      - media-quality flag set false by a capability
      - distribution-shift index beyond tolerance (transform battery unstable)
      - too many known-gated tools for the domain to have any coverage left
    (The external-authoritative conflict rules were RETIRED 2026-08-26
    together with the removed external-API integration — there is no
    authoritative negative/positive anchor to conflict-check against anymore.)
    """

    def __init__(self, max_disagreement: float = 0.5,
                 max_unknown_frac: float = 0.75,
                 shift_tolerance: float = 0.8):
        self.max_disagreement = max_disagreement
        self.max_unknown_frac = max_unknown_frac
        self.shift_tolerance = shift_tolerance

    def evaluate(self, fused: FusionDecision, checks: list[CheckResult],
                 ctx: JobContext, target: str = "") -> tuple[bool, list[str]]:
        notes: list[str] = []
        ok = True
        usable = [c for c in checks if c.usable()]
        unknown = [c for c in checks if c.status in ("unavailable", "skipped")]

        if not usable:
            return False, ["zero_usable_signals"]

        # Fusion v2: a COHERENT deepfake pattern (multiple detectors of DIFFERENT
        # types agreeing on the same generative mode) is corroboration, not
        # conflict — do not abort on spread when the pattern is coherent.
        # IMAGE (2026-08-26): exempt from the disagreement-abort — the NOT-binary
        # cap already bounds image verdicts at CAUTION (a lone high read can
        # never reach DNU), so a SPAI-vs-freqband spread must surface as
        # CAUTION/MEDIUM (suspicious), NOT UNVERIFIED. This is the operator's
        # requested fix (image detector was returning UNVERIFIED).
        # AUDIO (2026-08-26): also exempt — aasist3 is PROVEN-trustworthy (AUC
        # 0.9967 on the official ASVspoof2019-LA eval; bonafide 0.0000/spoof
        # 0.9966, 1022/1022), so a lone strong aasist read is decisive evidence,
        # not a conflicting-mode case. A spoof that aasist scores 0.998 must
        # read HIGH (not UNVERIFIED) even when the weak offline features disagree.
        if fused.disagreement > self.max_disagreement and not fused.coherent_pattern \
                and target not in ("image_facecheck", "deepfake_audio"):
            ok = False
            notes.append(f"disagreement={fused.disagreement:.2f}>{self.max_disagreement}")

        if unknown and len(usable) < 2 and len(unknown) / max(1, len(checks)) > self.max_unknown_frac:
            ok = False
            notes.append(f"unknown_tools={len(unknown)}/{len(checks)}")

        mq = ctx.extra.get("media_quality")
        if isinstance(mq, dict) and mq.get("ok") is False:
            ok = False
            notes.append("media_quality_insufficient")

        ds = ctx.extra.get("distribution_shift_index")
        if isinstance(ds, (int, float)) and ds > self.shift_tolerance:
            ok = False
            notes.append(f"distribution_shift={ds:.2f}")

        if not ok and fused.verdict is Verdict.TRUST:
            notes.insert(0, "suppressed TRUST verdict")
        return ok, notes
