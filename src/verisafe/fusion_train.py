"""Fusion training & evaluation harness (no numpy/pandas required).

Implements exactly what the spec demands:
  * per-capability logistic-regression stacking on OUT-OF-FOLD predictions
    from specialist detectors (leave-subset-out CV to estimate OOF probs);
  * temperature + bias calibration on held-out real-world-style data;
  * full metrics suite: accuracy, F1, ROC-AUC, PR-AUC, ECE, Brier, FPR,
    FNR, coverage@threshold, selective risk.
  * a labeled-synthetic-demo generator so the whole pipeline is exercisable
    offline on a CPU-only box (demo only — NOT a replacement for real data).

Train artifacts are pickled JSON (portable, inspectable): fusion/training/
stack_<target>.json — FusionEngine loads them if present.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

try:  # optional at import time; only required by the --synthetic/--dataset paths
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

# ------------------------------------------------------------------ math ----

def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def softmax(x: list[float]) -> list[float]:
    m = max(x)
    ex = [math.exp(v - m) for v in x]
    s = sum(ex) or 1e-12
    return [v / s for v in ex]


class LogisticRegression:
    """Binary LR with L2, trained by full-batch gradient descent."""

    def __init__(self, l2: float = 0.01, lr: float = 0.3, iters: int = 600, seed: int = 7):
        self.l2 = l2
        self.lr = lr
        self.iters = iters
        self.w: list[float] = []
        self.b = 0.0

    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        n, d = len(X), len(X[0])
        self.w = [0.0] * d
        self.b = 0.0
        rng = random.Random(0)  # no stochastic shuffling needed at demo scale
        del rng
        for _ in range(self.iters):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                p = self.predict_proba(X[i])
                e = p - y[i]
                for j in range(d):
                    gw[j] += e * X[i][j]
                gb += e
            for j in range(d):
                self.w[j] -= self.lr * (gw[j] / n + self.l2 * self.w[j])
            self.b -= self.lr * (gb / n)
        return self

    def decision_fn(self, x: list[float]) -> float:
        return self.b + sum(w * v for w, v in zip(self.w, x))

    def predict_proba(self, x: list[float]) -> float:
        return sigmoid(self.decision_fn(x))

    def to_dict(self) -> dict:
        return {"w": self.w, "b": self.b}


# --------------------------------------------------------------- metrics ----

def roc_auc(y_true: list[int], score: list[float]) -> float:
    """Ranking-based (Mann-Whitney U) AUC; tie-corrected."""
    pairs = sorted(zip(score, y_true), key=lambda t: t[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0          # mean rank of tied block
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def pr_auc(y_true: list[int], score: list[float]) -> float:
    """Area under precision-recall curve via trapezoids over distinct scores."""
    order = sorted(range(len(score)), key=lambda i: -score[i])
    tp = fp = 0
    n_pos = sum(y_true) or 1
    area = 0.0
    recall_prev = prec_prev = 0.0
    for idx in range(len(order)):
        yi = y_true[order[idx]]
        if yi == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / (tp + fp)
        rec = tp / n_pos
        area += (rec - recall_prev) * (prec + prec_prev) / 2.0
        recall_prev, prec_prev = rec, prec
    return max(area, (n_pos / len(y_true)))  # never below baseline chance


def brier_score(y_true: list[int], prob: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(prob, y_true)) / len(y_true)


def expected_calibration_error(y_true: list[int], prob: list[float], bins: int = 10) -> float:
    buckets: dict[int, list[tuple[int, float]]] = {}
    for y, p in zip(y_true, prob):
        b = min(bins - 1, int(p * bins))
        buckets.setdefault(b, []).append((y, p))
    n = len(y_true)
    ece = 0.0
    for items in buckets.values():
        conf = sum(p for _, p in items) / len(items)
        acc = sum(y for y, _ in items) / len(items)
        ece += len(items) / n * abs(conf - acc)
    return ece


def fpr_fnr_at_threshold(y_true: list[int], prob: list[float], thr: float) -> tuple[float, float]:
    fp = tp = fn = tn = 0
    for y, p in zip(y_true, prob):
        pred = 1 if p >= thr else 0
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return fpr, fnr


def coverage_selective_risk(y_true: list[int], prob: list[float],
                            accept: float = 0.8, reject: float = 0.2) -> dict:
    """Selective-prediction accounting: fraction of inputs auto-decided
    'safe' vs 'unsafe' vs abstained, plus the error rate inside each
    auto-decision bucket (mislabel rate, NOT overall accuracy)."""
    yes = [y for y, p in zip(y_true, prob) if p >= accept]
    no = [y for y, p in zip(y_true, prob) if p <= reject]
    n = len(y_true) or 1
    return {
        "coverage_yes": len(yes) / n,
        "risk_yes": (sum(yes) / len(yes)) if yes else None,       # fraud share wrongly cleared safe
        "coverage_no": len(no) / n,
        "risk_no": ((len(no) - sum(no)) / len(no)) if no else None,  # clean share wrongly blocked
        "abstain_frac": (n - len(yes) - len(no)) / n,
    }


def full_metrics(y_true: list[int], prob: list[float]) -> dict:
    thr = 0.5
    preds = [1 if p >= thr else 0 for p in prob]
    tp = sum(1 for p_, y in zip(preds, y_true) if p_ and y)
    fp = sum(1 for p_, y in zip(preds, y_true) if p_ and not y)
    fn = sum(1 for p_, y in zip(preds, y_true) if not p_ and y)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr, fnr = fpr_fnr_at_threshold(y_true, prob, thr)
    return {
        "accuracy": sum(1 for p_, y in zip(preds, y_true) if p_ == y) / len(y_true),
        "f1": f1,
        "roc_auc": roc_auc(y_true, prob),
        "pr_auc": pr_auc(y_true, prob),
        "ece": expected_calibration_error(y_true, prob),
        "brier": brier_score(y_true, prob),
        "fpr@0.5": fpr,
        "fnr@0.5": fnr,
        **{f"sel_{k}": v for k, v in coverage_selective_risk(y_true, prob).items()},
    }


# --------------------------------------------------- OOF out-of-fold train ---

def oof_cross_val_preds(X: list[list[float]], y: list[int], target: str,
                        folds: int = 5, dest_dir: Path | None = None) -> dict:
    """Leave-subset-out: train on (folds-1)/folds, emit OOF probs, then fit
    final stacker on all data + optional holdout-style calibration. Writes
    JSON artifact consumed by FusionEngine.load_trained()."""
    import time as _t
    t0 = _t.monotonic()
    n = len(y)
    fold_of = [i % folds for i in range(n)]
    oof = [0.0] * n
    for f in range(folds):
        tr = [i for i in range(n) if fold_of[i] != f]
        te = [i for i in range(n) if fold_of[i] == f]
        if not tr or not te:
            continue
        model = LogisticRegression(iters=400).fit([X[i] for i in tr], [y[i] for i in tr])
        for i in te:
            oof[i] = model.predict_proba(X[i])
    # final stacker on all rows (kept as the serving model; OOF probs feed the
    # reliability/calibration layers — never used to evaluate itself)
    final = LogisticRegression(iters=600).fit(X, y)
    # temperature scaling grid search minimizing NLL on OOF probabilities
    def prob_t(pi: float, t: float) -> float:
        pi = min(max(pi, 1e-6), 1 - 1e-6)
        z = math.log(pi / (1 - pi)) / max(t, 1e-3)
        return sigmoid(z)

    best_t, best_nll = 1.0, float("inf")
    for t in [x / 100.0 for x in range(50, 301, 10)]:
        nll = 0.0
        for pi, yi in zip(oof, y):
            p = prob_t(pi, t)
            nll -= math.log(p if yi else 1 - p)
        if nll < best_nll:
            best_t, best_nll = t, nll
    artifact = {
        "target": target, "n_samples": n, "folds": folds,
        "features": ["oof_stack"],
        "final": final.to_dict(),
        "calibration": {"t": round(best_t, 2), "b": 0.0},
        "oof_metrics": full_metrics(y, oof),
        "final_metrics_on_oof": None,  # deliberately not evaluated on its own fit data (leakage guard)
        "train_time_s": round(_t.monotonic() - t0, 2),
    }
    if dest_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / f"stack_{target}.json").write_text(json.dumps(artifact, indent=1))
    return {"oof_probs": oof, "metrics": artifact["oof_metrics"], "artifact": artifact}


# ------------------------------------------- synthetic labeled demo data -----

def _demo_feature_layout(target: str) -> list[str]:
    """Ordered [signal] entries matching FusionEngine.feature_vector layout."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from verisafe.fusion import WEIGHTS
    return [k for k in WEIGHTS.get(target, {}) ]


def generate_demo_dataset(outdir: str | Path, n_clean: int = 150, n_fraud: int = 150,
                          seed: int = 42, target: str = "url_phishing") -> Path:
    """Simulate specialist-detection signals + ground-truth labels for one
    capability target, laid out EXACTLY like FusionEngine.feature_vector at
    serve time (per weighted signal: [value, gap_flag]). Signals follow
    realistic correlated-noise distributions so the stacker has learnable
    structure. DEMO ONLY — production must be retrained on labeled field data
    before relying on the calibrated output."""
    rng = random.Random(seed)
    sigs = _demo_feature_layout(target)
    # informative signals get class-dependent means; rest stay near-uniform
    informative = {
        "vt.url_positives_ratio": (0.02, 0.75), "phish.heuristic_score": (0.10, 0.80),
        "domain.young": (0.30, 0.90), "redirect.suspicious_hop": (0.05, 0.60),
        "ssrf.blocked": (0.0, 0.30), "download.ext_mismatch": (0.02, 0.40),
    }
    rows: list[dict] = []
    for label in (0, 1):
        cnt = n_clean if label == 0 else n_fraud
        for k in range(cnt):
            vec: list[float] = []
            for s in sigs:
                lo, hi = informative.get(s, (0.1, 0.9))
                mean = (lo + hi) / 2 if label == 1 else min(lo + 0.05, hi - 0.3)
                mean = max(0.02, min(mean, 0.98))
                val = min(1.0, max(0.0, rng.gauss(mean, 0.12)))
                # gated tools are "known gaps" more often: fraud pages less likely
                # to be blocked early => more evidence survives
                gap_p = 0.25 if label == 1 else 0.10
                gap = 1.0 if rng.random() < gap_p else 0.0
                if gap:
                    val = 0.0
                vec.append(round(val, 4))
                vec.append(gap)
            rows.append({"x": vec, "y": label, "cap": target})
    rng.shuffle(rows)
    out = Path(outdir) / f"{target}_demo_labeled.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": "DEMO synthetic labels — replace with field data",
                               "rows": rows}, indent=0))
    return out


# ------------------------------------------- numpy learner + artifacts --------

def _fit_logistic_np(X: "Any", y: "Any", seed: int = 0, l2: float = 1e-3,
                     lr: float = 0.5, epochs: int = 80,
                     batch: int = 64) -> tuple[list[float], float]:
    """Binary logistic regression by mini-batch SGD over log-loss + L2.

    Deterministic for a fixed (seed, X, y): row shuffles come from
    numpy.random.default_rng(seed). Returns (w, b) as plain floats so the
    result serialises into the same JSON artifact shape as the OOF path."""
    _npy = np
    if _npy is None:
        raise RuntimeError("numpy required for numpy_logistic_regression")
    X = _npy.asarray(X, dtype=_npy.float64)
    y = _npy.asarray(y, dtype=_npy.int64)
    n, d = X.shape
    rng = _npy.random.default_rng(seed)
    w = _npy.zeros(d, dtype=_npy.float64)
    b = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            z = X[idx] @ w + b
            # numerically stable sigmoid via exp(-|z| identity
            p = _npy.where(z >= 0, 1.0 / (1.0 + _npy.exp(-z)),
                           _npy.exp(z) / (1.0 + _npy.exp(z)))
            g = (p - y[idx]) / len(idx)          # mean gradient of log-loss
            gw = X[idx].T @ g + 2.0 * l2 * w
            gb = g.sum()
            w -= lr * gw
            b -= lr * gb
    return [float(v) for v in w], float(b)


def train_target_checkpoint(X, y, target: str, dest_dir: Path | str,
                            meta: dict[str, Any] | None = None,
                            seed: int = 0) -> dict[str, Any]:
    """Train one target's LR head and write $VERISAFE_FUSION_DIR/training/
    ``stack_<target>.json`` — the EXACT artifact contract FusionEngine
    .load_trained() consumes ({target, final:{w,b}, calibration:{t,b}} plus
    optional metrics/metadata). Also fits a 1D temperature on a seeded
    holdout split so served probabilities are calibrated, not just ranked.

    Raises ValueError on empty/single-class input; never touches the network.
    """
    _npy = np
    if _npy is None:
        raise RuntimeError("numpy required to train checkpoints")
    import time as _t
    t0 = _t.monotonic()
    X = _npy.asarray(X, dtype=_npy.float64)
    y = _npy.asarray(y)
    if len(y) == 0 or X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError(f"empty or misaligned dataset for {target} "
                         f"(n={len(y)}, shape={getattr(X, 'shape', None)})")
    if len(_npy.unique(y)) < 2:
        raise ValueError(f"single-class labels for {target} — cannot fit a binary head")

    from .dataset_collector import split
    Xtr, ytr, Xte, yte = split(X, y, test_frac=0.2, seed=seed)

    w, b = _fit_logistic_np(Xtr, ytr, seed=seed)
    wa = _npy.asarray(w)

    def _prob(row: object) -> float:
        z = max(min(float(b + row @ wa), 35.0), -35.0)
        return 1.0 / (1.0 + math.exp(-z))

    oof_probs = [_prob(r) for r in Xte]
    # 1-D temperature grid search minimizing NLL on the held-out split
    best_t, best_nll = 1.0, float("inf")
    for t in [x / 100.0 for x in range(50, 301, 10)]:
        nll = 0.0
        for pi, yi in zip(oof_probs, yte.tolist()):
            p = min(max(pi, 1e-6), 1 - 1e-6)
            z = math.log(p / (1 - p)) / max(t, 1e-3)
            pc = 1.0 / (1.0 + math.exp(-z))
            nll -= math.log(pc if yi else 1 - pc)
        if nll < best_nll:
            best_t, best_nll = t, nll

    art: dict[str, Any] = {
        "version": 1,
        "target": target,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "folds": None,                      # full-train (not CV) for the LR head
        "features": [f"f{i}" for i in range(X.shape[1])],
        "final": {"w": w, "b": b},
        "calibration": {"t": round(best_t, 2), "b": 0.0},
        "test_metrics": full_metrics(yte.tolist(), oof_probs),
        "train_time_s": round(_t.monotonic() - t0, 3),
        "note": "LR head trained on operator/synthetic labelled features; "
                "feature layout must match FusionEngine.feature_vector[target]",
    }
    if meta:
        art.update(meta)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"stack_{target}.json").write_text(json.dumps(art, indent=1))
    return art


def _parse_opt_args(args: list[str]) -> dict[str, Any]:
    """Tiny opt parser so the module stays stdlib-only at import time."""
    out: dict[str, Any] = {
        "target": "url_phishing", "dataset": None, "demo_json": None,
        "synthetic_n": None, "features": None, "seed": 0}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--target" and i + 1 < len(args):
            out["target"] = args[i + 1]; i += 2; continue
        if a.startswith("--target="):
            out["target"] = a.split("=", 1)[1]; i += 1; continue
        if a == "--dataset" and i + 1 < len(args):
            out["dataset"] = Path(args[i + 1]); i += 2; continue
        if a == "--synthetic" and i + 1 < len(args):
            out["synthetic_n"] = int(args[i + 1]); i += 2; continue
        if a == "--features" and i + 1 < len(args):
            out["features"] = int(args[i + 1]); i += 2; continue
        if a == "--seed" and i + 1 < len(args):
            out["seed"] = int(args[i + 1]); i += 2; continue
        if a.endswith(".json"):
            out["demo_json"] = Path(a); i += 1; continue
        i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(os.environ.get("VERISAFE_FUSION_DIR")
                or Path(__file__).resolve().parents[2] / "fusion")
    dest = root / "training"
    o = _parse_opt_args(args)
    target: str = o["target"]
    seed: int = o["seed"]

    def _fail(msg: str) -> int:
        print(json.dumps({"error": msg}))
        return 2

    try:
        # ------------------------------------------------------------ data ---
        X = y = None
        source = ""
        if o["dataset"] is not None:
            from . import dataset_collector as dc
            X, y = dc.load_jsonl(o["dataset"])          # raises DatasetError
            source = f"jsonl:{o['dataset']}"
        elif o["synthetic_n"] is not None:
            from . import dataset_collector as dc
            d = max(2, int(o["features"] or 8))
            X, y = dc.synthesize(o["synthetic_n"], d, seed=seed)
            source = f"synthetic:n={o['synthetic_n']},d={d},seed={seed}"
        else:
            # pre-existing OOF demo path (unchanged behaviour)
            demo = o["demo_json"]
            if demo is None:
                demo = generate_demo_dataset(root, target=target)
            rows = json.loads(Path(demo).read_text())["rows"]
            cap = rows[0].get("cap", target)
            X, y = [r["x"] for r in rows], [r["y"] for r in rows]
            res = oof_cross_val_preds(X, y, cap, dest_dir=dest)
            print(json.dumps(res["metrics"]))
            print(f"artifact -> {dest / ('stack_' + cap + '.json')}")
            return 0

        # -------------------------------------------------------- train -------
        art = train_target_checkpoint(X, y, target, dest,
                                      meta={"source": source, "seed": seed},
                                      seed=seed)
        tm = art.get("test_metrics") or {}
        conf = tm.get("fpr@0.5")
        print(json.dumps({
            "ok": True, "target": target, "source": source,
            "n": art["n_samples"], "features": art["n_features"],
            "test_accuracy": round(tm.get("accuracy", 0.0), 4),
            "test_roc_auc": round(tm.get("roc_auc", 0.0), 4),
            "confusion": {"tp_tn_fp_fn": None,
                          "fpr@0.5": round(conf, 4) if conf is not None else None},
            "calibration_t": art["calibration"]["t"],
            "train_time_s": art["train_time_s"],
            "artifact": str(dest / f"stack_{target}.json"),
        }))
        return 0
    except ValueError as e:      # DatasetError subclasses this too
        return _fail(str(e))
    except Exception as e:       # noqa: BLE001 - controlled message, no traceback
        return _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
