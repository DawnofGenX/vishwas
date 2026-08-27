"""Regression tests for the fusion training harness fixes (2026-08-27).

  1. A trained checkpoint whose feature width does not match the CURRENT
     signal layout must be REFUSED at load (never silently zero-padded), and
     the serving adapter must raise rather than emit a wrong probability.
  2. The synthetic demo generator must produce a learnable (separable)
     dataset for the target's real signals.
"""
import json
import tempfile
from pathlib import Path

import pytest

from vishwas.fusion import FusionEngine, WEIGHTS, _LRStackAdapter
from vishwas.capabilities.base import CheckResult as C


def _expected_features(target: str) -> int:
    return len(FusionEngine.feature_vector(target, []))


def test_loader_rejects_mismatched_checkpoint():
    d = Path(tempfile.mkdtemp())
    # url_phishing today has 1 signal -> 2 features; a 14-feature head is stale.
    (d / "stack_url_phishing.json").write_text(json.dumps({
        "target": "url_phishing", "final": {"w": [0.1] * 14, "b": 0.0},
        "calibration": {"t": 0.5, "b": 0.0}}))
    fe = FusionEngine()
    assert fe.load_trained(d) == 0
    assert "url_phishing" not in fe.lr_stacks


def test_loader_accepts_matching_checkpoint():
    d = Path(tempfile.mkdtemp())
    width = _expected_features("url_phishing")
    (d / "stack_url_phishing.json").write_text(json.dumps({
        "target": "url_phishing", "final": {"w": [0.5] * width, "b": -0.2},
        "calibration": {"t": 1.0, "b": 0.0}}))
    fe = FusionEngine()
    assert fe.load_trained(d) == 1
    assert "url_phishing" in fe.lr_stacks


def test_adapter_raises_on_feature_mismatch_instead_of_padding():
    # A 14-weight adapter fed the current 2-feature vector must raise, so
    # decide() falls back to the explicit weighted path rather than scoring a
    # phishing URL against weights trained for a different signal set.
    adapter = _LRStackAdapter("url_phishing", [0.1] * 14, 0.0)
    checks = [C("url_phish_scanner", "mid", "ok",
                {"is_phishing": True, "risk_score_norm": 0.95}, "phish")]
    with pytest.raises(ValueError):
        adapter.predict_proba({}, checks)


def test_demo_dataset_is_learnable():
    from vishwas.fusion_train import generate_demo_dataset, oof_cross_val_preds
    d = Path(tempfile.mkdtemp())
    out = generate_demo_dataset(d, n_clean=120, n_fraud=120, target="malicious_file")
    rows = json.loads(Path(out).read_text())["rows"]
    X = [r["x"] for r in rows]
    y = [r["y"] for r in rows]
    metrics = oof_cross_val_preds(X, y, "malicious_file", dest_dir=d / "training")["metrics"]
    assert metrics["roc_auc"] > 0.8, metrics


def test_demo_respects_negative_weight_direction():
    # gov_document has NEGATIVE weights (e.g. signature.valid) that read HIGH for
    # clean docs. The demo must still be separable for such a target.
    from vishwas.fusion_train import generate_demo_dataset, oof_cross_val_preds
    d = Path(tempfile.mkdtemp())
    out = generate_demo_dataset(d, n_clean=120, n_fraud=120, target="gov_document")
    rows = json.loads(Path(out).read_text())["rows"]
    X = [r["x"] for r in rows]
    y = [r["y"] for r in rows]
    metrics = oof_cross_val_preds(X, y, "gov_document", dest_dir=d / "training")["metrics"]
    assert metrics["roc_auc"] > 0.8, metrics
