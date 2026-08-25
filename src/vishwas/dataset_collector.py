"""Labelled-sample collection & curation helpers for the fusion trainer (D3).

Zero network. Two input sources, both deterministic:
  * ``synthesize`` — seeded synthetic feature matrices (balanced classes,
    separable-with-noise feature blocks) so a linear learner gets high
    accuracy and the whole pipeline is exercisable offline;
  * ``load_jsonl``  — operator-labelled corpus, one JSON object per line:
      {"features": [0.1, ...], "label": 0|1}
  ({"x": ..., "y": ...} — the legacy fusion_train demo layout — is also
  accepted so demo and field data interoperate.)

Output records are plain-python dicts (``to_dataset_dict``) so they survive
json.dumps round-trips without numpy-specific types. Training/checkpoint
writing lives in ``fusion_train`` (it owns the $VISHWAS_FUSION_DIR layout);
this module never touches disk except through explicit save helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment gate
    np = None  # type: ignore[assignment]


class DatasetError(ValueError):
    """Raised for any curation-time defect (empty, ragged, bad labels)."""


def _require_np():
    if np is None:
        raise DatasetError("numpy is required for dataset synthesis (pip install numpy)")
    return np


# ---------------------------------------------------------------- synthetic --

def synthesize(n: int, n_features: int, seed: int = 0) -> tuple[Any, Any]:
    """Balanced synthetic labels over noisy-feature blocks.

    Layout: the first ``max(2, n_features // 3)`` features are
    *informative* — class 1 sits at mean +delta, class 0 at -delta, with
    additive Gaussian noise (still separable by a linear model). Remaining
    features are uniform noise shared by both classes. Rows are generated
    per class then shuffled with the fixed seed, so output is byte-stable
    for a given (n, n_features, seed).

    Returns (X float32 [n x n_features], y int8 [n] of 0/1).
    """
    np = _require_np()
    if n < 4:
        raise DatasetError(f"synthesize needs n>=4, got {n}")
    if n_features < 2:
        raise DatasetError(f"synthesize needs n_features>=2, got {n_features}")
    rng = np.random.default_rng(seed)

    delta = 1.4
    noise = 0.9
    n_info = max(2, n_features // 3)

    def block(label: int, cnt: int):
        Xb = np.empty((cnt, n_features), dtype=np.float32)
        for j in range(n_features):
            if j < n_info:
                mu = delta if label == 1 else -delta
                Xb[:, j] = mu + noise * rng.standard_normal(cnt)
            else:
                Xb[:, j] = rng.uniform(0.0, 1.0, size=cnt)
        return Xb

    n0 = n // 2
    n1 = n - n0                       # class 1 absorbs any odd remainder
    X = np.vstack([block(0, n0), block(1, n1)])
    y = np.concatenate([np.zeros(n0, dtype=np.int8),
                        np.ones(n1, dtype=np.int8)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


# ------------------------------------------------------------------- jsonl ----

def load_jsonl(path: str | Path) -> tuple[Any, Any]:
    """Read an operator-labelled JSONL corpus.

    Accepts {"features":[...],"label":0|1} per line (and the legacy
    {"x":[...],"y":0|1} demo layout). Returns (X float32, y int8);
    raises DatasetError on empty/ragged/invalid input.
    """
    np = _require_np()
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"dataset file not found: {p}")
    feats: list[list[float]] = []
    labels: list[int] = []
    bad_lines: list[int] = []
    for ln, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            fv = obj.get("features", obj.get("x"))
            lb = obj.get("label", obj.get("y"))
            if not isinstance(fv, (list, tuple)) or len(fv) < 1 or lb not in (0, 1):
                bad_lines.append(ln)
                continue
            feats.append([float(v) for v in fv])
            labels.append(int(lb))
        except (ValueError, TypeError) as e:
            bad_lines.append(ln)
            del e
    if not feats:
        raise DatasetError(f"no valid labelled rows in {p}"
                           + (f" (bad lines: {bad_lines[:10]})" if bad_lines else ""))
    d = len(feats[0])
    if any(len(r) != d for r in feats):
        raise DatasetError(f"ragged feature widths in {p} (expected {d} columns)")
    if min(labels) == max(labels):
        raise DatasetError(f"single-class dataset in {p} — cannot train a binary head")
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(labels, dtype=np.int8))


def save_jsonl(X: Any, y: Any, path: str | Path) -> Path:
    """Write a corpus as operator-format JSONL ({features, label} per line)."""
    np = _require_np()
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or len(y) != X.shape[0]:
        raise DatasetError("save_jsonl expects X[n,d] aligned with y[n]")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row, lab in zip(X.tolist(), y.tolist()):
            f.write(json.dumps({"features": row, "label": int(lab)}) + "\n")
    return p


# -------------------------------------------------------------------- split ---

def split(X: Any, y: Any, test_frac: float = 0.2, seed: int = 0
          ) -> tuple[Any, Any, Any, Any]:
    """Stratified train/test split (fixed-seed permutation, disjoint by class).

    Returns (X_train, y_train, X_test, y_test). Preserves each class's
    proportion in both partitions within rounding.
    """
    np = _require_np()
    if not 0.0 < test_frac < 1.0:
        raise DatasetError(f"test_frac must be in (0,1), got {test_frac}")
    X = np.asarray(X)
    y = np.asarray(y)
    if len(y) < 4:
        raise DatasetError(f"cannot split a {len(y)}-row dataset (need >=4)")
    rng = np.random.default_rng(seed)
    tr_idx: list = []
    te_idx: list = []
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if idx.size < 2:
            raise DatasetError(f"class {cls} has only {idx.size} row(s) — cannot stratify")
        idx = rng.permutation(idx)
        n_te = max(1, int(round(len(idx) * test_frac)))
        n_te = min(n_te, len(idx) - 1)         # keep >=1 train row per class
        te_idx.append(idx[:n_te])
        tr_idx.append(idx[n_te:])
    tr = np.concatenate(tr_idx)
    te = np.concatenate(te_idx)
    return X[tr], y[tr], X[te], y[te]


# ------------------------------------------------------------- dataset dict ---

def to_dataset_dict(X: Any, y: Any, meta: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Plain-python record of a labelled corpus (json-safe, no numpy types).

    {"version":1, "n_rows", "n_features", "labels":[int], "data":[[float]],
     "feature_names":[f0..fd], **meta} — dump with json.dumps for archival
    or for feeding back through load_jsonl-style tooling.
    """
    np = _require_np()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or len(y) != X.shape[0]:
        raise DatasetError("to_dataset_dict expects X[n,d] aligned with y[n]")
    out: dict[str, Any] = {
        "version": 1,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": [f"f{i}" for i in range(X.shape[1])],
        "data": [[float(v) for v in row] for row in X.tolist()],
        "labels": [int(v) for v in y.tolist()],
    }
    if meta:
        out.update(meta)
    return out
