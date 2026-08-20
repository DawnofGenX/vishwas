"""Hermetic tests for the fusion labelled-dataset collector + trainer (D3).

All data is synthetic (numpy, fixed seeds) or operator JSONL under tmp_path —
zero network. Exercises the real on-disk checkpoint contract that
FusionEngine.load_trained() consumes ($VERISAFE_FUSION_DIR/training/
stack_<target>.json), plus the CLI entry points of fusion_train.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _py():
    return sys.executable


# ------------------------------------------------------------------ helpers --
@pytest.fixture(scope="module")
def np():
    return pytest.importorskip("numpy")


def _url_pairs():
    """Ordered (check_name, signal_key) pairs for the six url_phishing
    weighted signals — the exact source mapping FusionEngine.feature_vector
    consumes."""
    from verisafe.fusion import WEIGHTS, _SIGNAL_SOURCES
    out = []
    for sk in WEIGHTS["url_phishing"]:
        cname, sfield, _, _ = _SIGNAL_SOURCES[sk]
        out.append((cname, sfield))
    return out


def _checks_from_values(values_by_signal_field: dict[str, float]):
    """Build usable CheckResults whose signals feed feature_vector like
    serve-time checks do (one CheckResult per unique check name; multiple
    signal fields fold into its signals dict)."""
    from verisafe.capabilities.base import CheckResult
    grouped: dict[str, dict[str, object]] = {}
    order: list[tuple[str, str]] = []
    for cname, sfield in _url_pairs():
        if sfield not in values_by_signal_field:
            continue
        grouped.setdefault(cname, {})[sfield] = values_by_signal_field[sfield]
        order.append((cname, sfield))
    checks = [CheckResult(name=nm, cost="cheap", status="ok",
                          signals=dict(signals), notes="synthetic")
              for nm, signals in grouped.items()]
    return checks


_NFEAT = len(_url_pairs()) * 2          # value+gap flag per signal == 12


# ------------------------------------------------------------------- a -------
def test_a_synthesize_shapes_types_balance(np):
    from verisafe.dataset_collector import synthesize
    X, y = synthesize(1000, 12, seed=7)
    assert X.shape == (1000, 12)
    assert X.dtype == np.float32
    assert y.dtype == np.int8
    frac_pos = float(y.mean())
    assert abs(frac_pos - 0.5) <= 0.05, f"class balance off: {frac_pos}"
    X2, y2 = synthesize(1000, 12, seed=7)
    assert (X == X2).all() and (y == y2).all(), "synthesize must be seed-stable"


# ------------------------------------------------------------------- b -------
def test_b_split_stratified_disjoint_sizes(np):
    from verisafe.dataset_collector import synthesize, split
    X, y = synthesize(200, 8, seed=1)
    Xtr, ytr, Xte, yte = split(X, y, test_frac=0.25, seed=3)
    frac_te = len(yte) / 200
    assert 0.20 <= frac_te <= 0.30, f"test share {frac_te} outside expectation"
    for yy, part in ((ytr, "train"), (yte, "test")):
        n = len(yy)
        p = int(yy.sum())
        assert p >= 1 and n - p >= 1, f"{part} lost a class entirely"
        assert abs(p / n - 0.5) < 0.15, f"{part} class ratio unstratified"
    X2, y2, te2_x, te2_y = split(X, y, test_frac=0.25, seed=3)   # same seed => identical
    assert (X2 == Xtr).all() and (y2 == ytr).all(), "split must be deterministic"
    assert (te2_x == Xte).all() and (te2_y == yte).all(), "test partition must be deterministic"


# ------------------------------------------------------------------- c -------
def test_c_tiny_lr_beats_baseline_under_budget(np):
    t0 = time.monotonic()
    from verisafe.dataset_collector import synthesize, split
    from verisafe.fusion_train import _fit_logistic_np
    X, y = synthesize(1500, 15, seed=0)
    Xtr, ytr, Xte, yte = split(X, y, test_frac=0.2, seed=0)
    w, b = _fit_logistic_np(Xtr, ytr, seed=0)
    zt = Xte @ w + b
    prob = 1.0 / (1.0 + np.exp(-zt))
    acc = float(((prob >= 0.5).astype(int) == yte).mean())
    elapsed = time.monotonic() - t0
    assert acc > 0.85, f"held-out accuracy {acc:.3f} below bar"
    assert elapsed < 5.0, f"training took {elapsed:.2f}s (budget 5s)"


# ------------------------------------------------------------------- d -------
def test_d_jsonl_roundtrip(tmp_path, np):
    from verisafe import dataset_collector as dc
    X, y = dc.synthesize(500, 10, seed=5)
    p = tmp_path / "corpus.jsonl"
    dc.save_jsonl(X, y, p)
    Xr, yr = dc.load_jsonl(p)
    assert Xr.shape == X.shape
    assert (np.abs(Xr.astype(np.float64) - X) < 1e-6).all()
    assert (yr == y).all()
    # single-class corpus must be rejected with a typed error
    p1 = tmp_path / "one_class.jsonl"
    p1.write_text("\n".join(json.dumps({"features": [0.1], "label": 0})
                            for _ in range(9)))
    with pytest.raises(dc.DatasetError):
        dc.load_jsonl(p1)


# ------------------------------------------------------------------- e -------
def test_e_checkpoint_roundtrip_via_engine(tmp_path, np):
    from verisafe import dataset_collector as dc
    from verisafe.fusion import FusionEngine
    from verisafe.fusion_train import train_target_checkpoint

    X, y = dc.synthesize(1500, _NFEAT, seed=11)
    Xtr, ytr = dc.split(X, y, test_frac=0.2, seed=11)[:2]
    _, _, Xte, yte = dc.split(X, y, test_frac=0.2, seed=11)

    dest = tmp_path / "fusion" / "training"
    art = train_target_checkpoint(Xtr, ytr, "url_phishing", dest, seed=11)
    ckpt = dest / "stack_url_phishing.json"
    assert ckpt.is_file(), "checkpoint artifact missing after training"
    loaded = json.loads(ckpt.read_text())
    assert loaded["target"] == "url_phishing"
    assert len(loaded["final"]["w"]) == _NFEAT

    eng = FusionEngine()
    assert eng.load_trained(dest) == 1, "load_trained did not wire the new checkpoint"
    assert "url_phishing" in eng.lr_stacks
    assert eng.calibration.get("url_phishing") is not None

    # served predictions (engine path) must track held-out ground truth
    agree = 0
    total = 0
    for row, lab in zip(Xte.tolist(), yte.tolist()):
        vals = {_url_pairs()[i][1]: row[i * 2] for i in range(len(_url_pairs()))}
        checks = _checks_from_values(vals)
        d = eng.decide("url_phishing", checks)
        agree += int((1 if d.score >= 0.5 else 0) == lab)
        total += 1
    acc = agree / total
    assert acc > 0.85, f"engine-applied checkpoint accuracy {acc:.3f} below bar"


# ------------------------------------------------------------------- f -------
def _cli(args, tmp_path):
    env = dict(os.environ)
    env["VERISAFE_FUSION_DIR"] = str(tmp_path / "fdir")
    r = subprocess.run([_py(), "-m", "verisafe.fusion_train", *args],
                       capture_output=True, text=True, env=env, cwd=str(ROOT))
    return r, env


def test_f_empty_dataset_controlled_failure(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    r, _ = _cli(["--dataset", str(empty), "--target=t"], tmp_path)
    assert r.returncode != 0, "empty dataset must exit non-zero"
    payload = json.loads(r.stdout.strip().splitlines()[-1])
    assert "error" in payload, f"no structured error in stdout: {r.stdout!r}"
    assert "Traceback" not in r.stderr, f"traceback leaked: {r.stderr!r}"

    tiny = tmp_path / "tiny.jsonl"
    tiny.write_text("\n".join(json.dumps({"features": [1.0], "label": 1})
                              for _ in range(5)))
    r2, _ = _cli(["--dataset", str(tiny), "--target=t"], tmp_path)
    assert r2.returncode != 0, "single-class dataset must exit non-zero"


# ------------------------------------------------------------- bonus: cli ---
def test_cli_synthetic_end_to_end(tmp_path):
    r, env = _cli(["--synthetic", "800", "--features", "12",
                   "--target", "malicious_file"], tmp_path)
    assert r.returncode == 0, f"cli failed: {r.stderr or r.stdout}"
    last = json.loads(r.stdout.strip().splitlines()[-1])
    assert last["ok"] is True
    ck = Path(env["VERISAFE_FUSION_DIR"]) / "training" / "stack_malicious_file.json"
    assert ck.is_file()
    assert last["test_accuracy"] > 0.85
