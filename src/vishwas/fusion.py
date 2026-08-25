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
    "url_phishing": {
        "vt.url_positives_ratio": 4.0,
        "phish.heuristic_score": 2.5,
        "domain.young": 1.0,
        "redirect.suspicious_hop": 1.5,
        "ssrf.blocked": 1.5,
        "download.ext_mismatch": 1.0,
    },
    "gov_document": {
        "digilocker.verified": -5.0,
        "apisetu.records_found": -1.0,
        "signature.valid": -4.0,
        "sig_object.present": -1.5,
        "qr.sha1_match": -3.0,
        "fin.invalid_upi": 1.5,
        "rag.template_deviation": 0.5,
        "extraction.low_quality": 1.0,
    },
    "deepfake_video": {
        "effort.prob": 2.5,
        "demamba.prob": 2.0,
        "frameheur.prob": 1.5,
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
        "freqband.prob": 1.5,
        "faceforensics.prob": 2.5,
    },
    "document_generic": {},
    "unclassified": {},
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
    "vt.url_positives_ratio": ("vt_url_reputation", "positives_ratio", KIND_DEFAULT_NUM, False),
    "phish.heuristic_score": ("phish_heuristics", "score_norm", KIND_DEFAULT_NUM, False),
    "domain.young": ("phish_heuristics", "young_domain", KIND_DEFAULT_NUM, False),
    "redirect.suspicious_hop": ("url_redirects", "suspicious_hops", KIND_DEFAULT_NUM, False),
    "ssrf.blocked": ("ssrf_guard", "blocked", KIND_DEFAULT_NUM, False),
    "download.ext_mismatch": ("url_download_revalidated", "ext_mismatch", KIND_DEFAULT_NUM, False),
    # --- gov_document ---
    "digilocker.verified": ("digilocker_verify", "dl_verified", KIND_NEG_BOOL, False),
    "apisetu.records_found": ("api_setu_lookup", "records_found", KIND_INT_FRAC, False),
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
    "deepfake_video": 3, "deepfake_audio": 4, "cross_modal": 2,
    "image_facecheck": 2, "malicious_file": 3, "url_phishing": 2,
    "gov_document": 0, "document_generic": 0, "unclassified": 0,
}


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
        return "value", 1.0
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
        for key, weight in wmap.items():
            spec = _SIGNAL_SOURCES.get(key)
            if spec is None:
                continue
            state, val = _extract(spec, by_name.get(spec[0]))
            if state == "known_gap":
                known_gaps.append(key)
                continue
            if state != "value":
                missing.append(key)
                continue
            assert val is not None
            total_w += abs(weight)
            s += weight * val
            if spec[3] and isinstance(val, float):
                probs.append(val)

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
            for key2, w2 in wmap.items():
                spec2 = _SIGNAL_SOURCES.get(key2)
                if spec2 is None:
                    continue
                # KIND_CONST_TRUE signals are presence-flags: value 1.0 means the BAD
                # condition exists (e.g. ext mismatch). Absent check => N/A, not risky.
                if spec2[2] == "const_true":
                    st0 = _extract(spec2, by_name.get(spec2[0]))[0]
                    continue  # never counts toward (or against) the clean gate
                st2, val2 = _extract(spec2, by_name.get(spec2[0]))
                if st2 == "value" and isinstance(val2, (int, float)):
                    present_vals.append(float(val2))
            if len(present_vals) >= _CLEAN_MIN_SIGNALS and all(v <= _CLEAN_EPS for v in present_vals):
                x += -1.8
        # logistic squash keeps range (0,1) and is robust to weight drift
        raw = 1.0 / (1.0 + math.exp(-6.0 * x)) if total_w > 0 else 0.0

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

        disagreement = (max(probs) - min(probs)) if len(probs) >= 2 else 0.0

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

        # confidence: distance from midpoint, scaled by detector coverage,
        # penalized by inter-detector disagreement
        exp_det = _EXPECTED_PROB_DET.get(target, 1)
        coverage = min(1.0, (len(probs) / exp_det)) if exp_det else 1.0
        certainty = max(0.0, min(1.0, 2.0 * abs(0.5 - raw) * (0.4 + 0.6 * coverage)))
        if disagreement > 0.35:
            certainty *= 0.5
            if verdict is Verdict.TRUST:
                verdict = Verdict.CAUTION
            certainty = min(certainty, 0.4)
        if missing:
            certainty *= 0.7

        reasons: list[str] = []
        if missing:
            reasons.append("incomplete_evidence:" + ";".join(missing[:4]))
        if known_gaps:
            reasons.append(f"gated_tools:{len(known_gaps)}")
        if disagreement > 0.35:
            reasons.append(f"detector_disagreement:{disagreement:.2f}")
        if model_id:
            reasons.append(f"stack:{model_id}")
        reasons.append(f"risk_raw:{raw:.2f}")

        return FusionDecision(verdict=verdict, score=round(raw, 4), raw=round(x, 4),
                              disagreement=round(disagreement, 3), reasons=reasons,
                              confidence=round(certainty, 3),
                              model_ids=[model_id] if model_id else [],
                              known_gaps=known_gaps, missing_evidence=missing)


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
            self.lr_stacks[target] = _LRStackAdapter(target, final_lr["w"], final_lr["b"])
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
        # pad/truncate for version drift between trained & live feature sets
        if len(x) < len(self.w):
            x = x + [0.0] * (len(self.w) - len(x))
        else:
            x = x[:len(self.w)]
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
      - explicit evidence conflicts (signature valid vs authoritative negative;
        authoritative positive vs QR/hash mismatch)
    """

    def __init__(self, max_disagreement: float = 0.5,
                 max_unknown_frac: float = 0.75,
                 shift_tolerance: float = 0.8):
        self.max_disagreement = max_disagreement
        self.max_unknown_frac = max_unknown_frac
        self.shift_tolerance = shift_tolerance

    def evaluate(self, fused: FusionDecision, checks: list[CheckResult],
                 ctx: JobContext) -> tuple[bool, list[str]]:
        notes: list[str] = []
        ok = True
        usable = [c for c in checks if c.usable()]
        unknown = [c for c in checks if c.status in ("unavailable", "skipped")]

        if not usable:
            return False, ["zero_usable_signals"]

        if fused.disagreement > self.max_disagreement:
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

        # conflict rules on the named checks
        sig_valid = _sig(usable, "digital_signature", "valid")
        sig_object = _sig(usable, "digital_signature", "has_sig_object")
        dl = by_name_state(usable, "digilocker_verify", "dl_verified")
        qr_ok = _sig(usable, "qr_native_check", "sha1_matches_declaration")
        upi_bad = _sig(usable, "financial_field_validation", "invalid_upis")

        if (sig_valid or (sig_object is True)) and (dl is False):
            ok = False
            notes.append("conflict: signature/object indicator vs DigiLookup negative")

        if dl is True and (qr_ok is False):
            ok = False
            notes.append("conflict: authoritative verify vs embedded-hash mismatch")
        if dl is False and upi_bad is True:
            ok = False
            notes.append("conflict: payment-instruction tamper indicators with failed authoritative check")
        if not ok and fused.verdict is Verdict.TRUST:
            notes.insert(0, "suppressed TRUST verdict")
        return ok, notes


def _sig(checks, name: str, key: str):
    """Signal value of the first *usable* check with given name/key, else None."""
    for c in checks:
        if c.name == name and c.usable():
            v = c.signals.get(key)
            return v
    return None


def by_name_state(checks, name: str, key: str):
    """Signal value of the first check with given name (any status), else None.

    Used for conflict detection where a DEGRADED/negative authoritative result
    still counts as evidence (unlike _sig which ignores non-usable checks)."""
    for c in checks:
        if c.name == name:
            return c.signals.get(key)
    return None
