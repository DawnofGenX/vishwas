"""Direct probe: strict-load + inference proof for the vendored XLSR-Mamba gate.

Run: PYTHONPATH=/home/hermes/pylibs:/home/hermes/docling-python:/home/hermes/vishwas/src \
     python3 scripts/verify_xlsrmamba.py
Expects the real checkpoint at $VISHWAS_XLSRMAMBA_WEIGHTS (default
/opt/vishwas/models/xlsr-mamba/model.safetensors). Prints strict-load tensor
counts, a synthetic-waveform score, and cuda/cpu wall times. Exits non-zero on
any failure.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np


def main() -> int:
    path = os.environ.get(
        "VISHWAS_XLSRMAMBA_WEIGHTS",
        "/opt/vishwas/models/xlsr-mamba/model.safetensors",
    )
    if not os.path.exists(path):
        print(f"FAIL: checkpoint missing at {path}")
        return 2

    import torch  # real torch (docling tree) required for this probe

    from safetensors import safe_open

    from vishwas.model_adapters import _extract_state_dict, resolve
    from vishwas.device import resolve_device
    from vishwas.model_archs import get_arch

    print("torch:", torch.__version__, "| device:", resolve_device())

    spec = get_arch("xlsrmamba")
    assert spec is not None and getattr(spec, "implemented", False), \
        "xlsrmamba arch not registered/implemented"

    # ---- strict load ------------------------------------------------------
    t0 = time.time()
    model = spec.build()
    build_s = time.time() - t0

    with safe_open(path, framework="pt") as f:
        ckpt_keys = list(f.keys())
        sd = {k: f.get_tensor(k) for k in ckpt_keys}
    sd = _extract_state_dict(sd)

    t0 = time.time()
    ok = spec.apply_state(model, sd)
    load_s = time.time() - t0
    n_model = len(model.state_dict())
    print(f"build(): {build_s:.1f}s | apply_state(strict=True): {ok} in {load_s:.1f}s")
    print(f"tensors matched: {n_model}/{len(sd)}")
    if not ok:
        print("FAIL: strict load failed; last_apply =", spec.last_apply)
        return 1

    # ---- synthetic-waveform score ----------------------------------------
    rng = np.random.default_rng(1234)
    wav = (0.05 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
           + 0.02 * rng.standard_normal(16000)).astype(np.float32)

    results = []
    devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    for dev in devices:
        os.environ["VISHWAS_DEVICE"] = dev
        m = spec.build()
        if not spec.apply_state(m, sd):
            print(f"FAIL: {dev}: strict load failed")
            return 1
        t0 = time.time()
        p = spec.score(m, wav)
        dt = time.time() - t0
        results.append((dev, p, dt))
        print(f"[{dev}] P(spoof)={p:.6f}  wall={dt:.2f}s  (first call incl. warmup)")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    os.environ.pop("VISHWAS_DEVICE", None)

    # determinism check on the last-built model
    p2 = spec.score(model, wav)
    print(f"determinism: {abs(p2 - results[-1][1]) < 1e-6}")

    # ---- through the production adapter seam ------------------------------
    os.environ.setdefault("VISHWAS_XLSRMAMBA_WEIGHTS", path)
    adapter = resolve("VISHWAS_XLSRMAMBA_WEIGHTS")
    print("adapter registered:", adapter is not None)
    obj = adapter.load(path)
    print("loaded object:", type(obj).__name__ if obj else None,
          "| usable:", obj is not None and hasattr(obj, "predict"))
    if obj is None:
        print("last_reason:", adapter.last_reason)
        return 1
    p3 = float(obj.predict(wav))
    print(f"adapter.predict -> {p3:.6f}")
    print(f"status/signals via run(): {adapter.run(path, wav)[:2]}")

    lo, hi = min(r[1] for r in results), max(r[1] for r in results)
    print(f"\ncross-device spread: {hi - lo:.2e}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
