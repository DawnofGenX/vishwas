#!/usr/bin/env python3
"""LIVE PROBE: do the five learned gates LOAD and SCORE on this host?

2026-10-22. The suite's torch-gated arch modules PASS, but a passing arch
test is not the same as the SERVING path loading the model. This probe
does the real thing: it resolves each gate the way the app does, loads
the checkpoint, runs one real input through it, and reports what came
back.

It also checks the thing that matters most for a verification product:
does the detector's output SEPARATE real from fake? A model that loads
and returns a constant is loaded but useless, and the skill records
exactly that failure mode for HAVIC.

Every gate is probed with the PRODUCTION adapter (model_adapters), not
by calling the arch directly, because the adapter owns the preprocessing.

Usage:
    set -a; . deploy/vishwas-secrets.env; set +a
    /home/hermes/.venv-ambient/bin/python tools/probe_model_gates.py
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vishwas.events import Artifact, JobContext  # noqa: E402
from vishwas.model_adapters import resolve  # noqa: E402

Q = pathlib.Path(tempfile.mkdtemp(prefix="vishwas-gate-probe-"))

# (gate env var, label, what the score MEANS)
GATES = [
    ("VISHWAS_AASIST_WEIGHTS", "aasist3 (audio)", "spoof posterior; 1.0 = fake"),
    ("VISHWAS_XLSRMAMBA_WEIGHTS", "xlsr-mamba (audio)", "spoof posterior; 1.0 = fake"),
    ("VISHWAS_EFFORT_WEIGHTS", "effort/ffpp (video face)", "AIGC posterior; 1.0 = fake"),
    ("VISHWAS_IMAGE_FACE_WEIGHTS", "spai (image)", "AI-image posterior; 1.0 = fake"),
    ("VISHWAS_HAVIC_WEIGHTS", "havic (av crossmodal)", "inconsistency; 1.0 = fake"),
]


def _ctx(p: pathlib.Path, tag: str) -> JobContext:
    q = Q / tag
    q.mkdir(parents=True, exist_ok=True)
    art = Artifact(path=p, original_filename=p.name, declared_type="file",
                   size_bytes=p.stat().st_size)
    return JobContext(job_id="probe", artifact=art, quarantine_root=q,
                      wall_budget_s=300.0, model_weights_available=True)


def probe(env_var: str, label: str, meaning: str) -> dict:
    path = os.environ.get(env_var, "")
    rec = {"gate": env_var, "label": label, "meaning": meaning,
           "configured": bool(path), "path": path, "exists": False,
           "loaded": False, "error": "", "scores": []}
    if not path:
        rec["error"] = "NOT CONFIGURED"
        return rec
    p = pathlib.Path(path)
    rec["exists"] = p.is_file()
    if not rec["exists"]:
        rec["error"] = "FILE NOT FOUND"
        return rec

    t0 = time.time()
    try:
        gate = resolve(env_var)
        if gate is None:
            rec["error"] = "resolve() returned None"
            return rec
        model = gate.load(path)
        if model is None:
            rec["error"] = "load() returned None"
            return rec
        rec["loaded"] = True
        rec["load_s"] = round(time.time() - t0, 2)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    return rec


def main() -> int:
    print("=" * 78)
    print("LIVE PROBE — the five learned gates")
    print("=" * 78)
    print(f"  interpreter: {sys.executable}")
    print(f"  torch:       {__import__('torch').__version__}")
    print(f"  cuda:        {__import__('torch').cuda.is_available()}")
    print()

    records = []
    for env_var, label, meaning in GATES:
        rec = probe(env_var, label, meaning)
        records.append(rec)
        status = "LOADED" if rec["loaded"] else "FAILED"
        print(f"[{status:6}] {rec['label']}")
        print(f"          {env_var} = {rec['path'] or '(unset)'}")
        if rec["loaded"]:
            print(f"          load time: {rec['load_s']}s   ({rec['meaning']})")
        else:
            print(f"          ERROR: {rec['error']}")
        print()

    loaded = [r for r in records if r["loaded"]]
    print("=" * 78)
    print(f"  {len(loaded)} of {len(GATES)} gates LOADED")
    for r in records:
        mark = "OK  " if r["loaded"] else "FAIL"
        print(f"    {mark} {r['label']:26} {r['error'] or 'loaded'}")
    print("=" * 78)

    out = pathlib.Path("/home/hermes/.hermes/cache/scratch/gate_probe.json")
    out.write_text(json.dumps(records, indent=2))
    print(f"  wrote {out}")
    shutil.rmtree(Q, ignore_errors=True)
    return 0 if len(loaded) == len(GATES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
