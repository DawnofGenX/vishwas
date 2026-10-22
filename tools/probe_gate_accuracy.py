#!/usr/bin/env python3
"""LIVE ACCURACY PROBE — do the loaded gates actually SEPARATE real from fake?

2026-10-22. The five gates all LOAD (see probe_model_gates.py). But
loading is not usefulness: the skill records HAVIC as "degenerate /
constant fake-bias — reads ~0.98 on everything, even zeros". A model
that loads and returns a constant is loaded but useless.

This probe scores every gate through the PRODUCTION adapter on the REAL
corpora and reports the separation:
  * AUDIO  — the official ASVspoof2019-LA eval parquet (bonafide vs spoof)
  * IMAGE — the raw 30-AI / 30-real image corpus
  * VIDEO — the raw FaceForensics++ subset (fake vs real)

Everything is scored via the adapter (which owns the preprocessing) and
via the app's own capability functions where they exist -- never by
calling the arch directly.

Usage:
    set -a; . deploy/vishwas-secrets.env; set +a
    /home/hermes/.venv-ambient/bin/python tools/probe_gate_accuracy.py
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import shutil
import statistics as st
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

Q = pathlib.Path(tempfile.mkdtemp(prefix="vishwas-acc-probe-"))

PARQUET = pathlib.Path("/home/hermes/fusion_audio/asv19_la/asv19_val.parquet")
IMG_DIR = pathlib.Path("/home/hermes/fusion_img")
VID_DIR = pathlib.Path("/home/hermes/fusion_av/ffc23/subset")


def auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC where label 1 = fake (higher score)."""
    if not scores or len(set(labels)) < 2:
        return None
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    # rank-sum with tie handling
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    s_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def sep(scores: list[float], labels: list[int]) -> str:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return "n/a"
    return (f"fake med {st.median(pos):.4f} (min {min(pos):.4f}) | "
            f"real med {st.median(neg):.4f} (max {max(neg):.4f})")


# ---------------------------------------------------------------- AUDIO
def probe_audio(n: int = 60) -> list[dict]:
    if not PARQUET.exists():
        return [{"gate": "aasist3/xlsr-mamba", "error": "eval parquet missing"}]
    import pyarrow.parquet as pq
    import soundfile as sf
    from vishwas.capabilities.deepfake_audio import _infer_prob
    from vishwas.model_adapters import resolve

    tbl = pq.read_table(PARQUET, columns=["audio", "key"])
    audio = tbl.column("audio").to_pylist()
    keys = tbl.column("key").to_pylist()
    bonafide = [i for i, k in enumerate(keys) if k == 0][:n]
    spoof = [i for i, k in enumerate(keys) if k == 1][:n]
    picks = [(i, 0) for i in bonafide] + [(i, 1) for i in spoof]

    # ⚠️ THE PRODUCTION AUDIO LANE WORKS ON WAV FILES ON DISK (2026-10-22).
    # The first version of this probe called `model.predict(data, sr)` and
    # got n=0 for BOTH audio gates -- a silent failure that looked like
    # "the model does not work". The truth is that the adapter's contract
    # is `preprocess(raw) -> model input`, where raw is a WAV PATH or a
    # 1-D waveform, and the audio capability scores CROP WINDOWS written
    # to disk (`_multi_crop` / `_xlsr_detector`).
    #
    # So: write each clip to a wav, then score it the way the capability
    # does. An n=0 result must always be read as "the probe called it
    # wrong", never as "the model is broken".
    out = []
    wavdir = Q / "wav"
    wavdir.mkdir(parents=True, exist_ok=True)

    for env_var, label in (("VISHWAS_AASIST_WEIGHTS", "aasist3"),
                            ("VISHWAS_XLSRMAMBA_WEIGHTS", "xlsr-mamba")):
        path = os.environ.get(env_var, "")
        if not path or not pathlib.Path(path).is_file():
            out.append({"gate": label, "error": "not configured"})
            continue
        try:
            gate = resolve(env_var)
            model = gate.load(path)
        except Exception as e:  # noqa: BLE001
            out.append({"gate": label, "error": f"{type(e).__name__}: {e}"})
            continue
        if model is None:
            out.append({"gate": label, "error": "load() -> None"})
            continue

        scores, labels, errs = [], [], 0
        for k, (idx, lab) in enumerate(picks):
            try:
                data, sr = sf.read(io.BytesIO(audio[idx]["bytes"]), dtype="float32")
            except Exception:
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            w = wavdir / f"{label}_{k}.wav"
            try:
                sf.write(str(w), data, sr)
            except Exception:
                continue
            # THE PRODUCTION CALL, not a reimplementation of it: the
            # capability's own `_infer_prob` (preprocess -> call -> extract
            # the probability), because reimplementing it is exactly how a
            # probe ends up reporting "the model is broken" when the PROBE
            # is wrong.
            try:
                s = _infer_prob(gate, model, str(w))
            except Exception:
                errs += 1
                continue
            if s is None:
                errs += 1
                continue
            scores.append(s)
            labels.append(lab)
        rec = {"gate": label, "n": len(scores), "errors": errs,
               "auc": auc(scores, labels), "sep": sep(scores, labels)}
        out.append(rec)
    return out


# ---------------------------------------------------------------- IMAGE
def probe_image(n: int = 30) -> list[dict]:
    import cv2
    from vishwas.model_adapters import resolve

    out = []
    env_var, label = "VISHWAS_IMAGE_FACE_WEIGHTS", "spai"
    path = os.environ.get(env_var, "")
    if not path or not pathlib.Path(path).is_file():
        return [{"gate": label, "error": "not configured"}]
    try:
        gate = resolve(env_var)
        model = gate.load(path)
    except Exception as e:  # noqa: BLE001
        return [{"gate": label, "error": f"{type(e).__name__}: {e}"}]
    if model is None:
        return [{"gate": label, "error": "load() -> None"}]

    scores, labels = [], []
    for sub, lab in (("ai", 1), ("real", 0)):
        d = IMG_DIR / sub
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir())[:n]:
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                s = model.predict(img)
            except Exception:
                continue
            if s is None:
                continue
            scores.append(float(s))
            labels.append(lab)
    return [{"gate": label, "n": len(scores), "auc": auc(scores, labels),
             "sep": sep(scores, labels)}]


# ---------------------------------------------------------------- VIDEO
def probe_video(n: int = 40) -> list[dict]:
    from vishwas.capabilities.deepfake_video import DeepfakeVideoCapability
    from vishwas.events import Artifact, JobContext
    from vishwas.media_utils import extract_frames

    out = []
    env_var, label = "VISHWAS_EFFORT_WEIGHTS", "effort/ffpp"
    path = os.environ.get(env_var, "")
    if not path or not pathlib.Path(path).is_file():
        return [{"gate": label, "error": "not configured"}]

    cap = DeepfakeVideoCapability()
    clips = []
    for sub, lab in (("fake", 1), ("real", 0)):
        d = VID_DIR / sub
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.mp4"))[:n]:
            clips.append((f, lab))
    if not clips:
        return [{"gate": label, "error": f"no clips under {VID_DIR}"}]

    scores, labels = [], []
    for i, (f, lab) in enumerate(clips):
        q = Q / f"v{i}"
        q.mkdir(parents=True, exist_ok=True)
        art = Artifact(path=f, original_filename=f.name, declared_type="video",
                       size_bytes=f.stat().st_size)
        ctx = JobContext(job_id="vp", artifact=art, quarantine_root=q,
                         wall_budget_s=300.0, model_weights_available=True)
        try:
            frames = extract_frames(f, q / "frames", n=8)
            if not frames:
                continue
            checks = cap._effort(ctx, frames)
        except Exception:
            continue
        for c in checks:
            if c.name == "effort_face_forensics" and c.status == "ok":
                v = c.signals.get("prob_deepfake")
                if v is not None:
                    scores.append(float(v))
                    labels.append(lab)
                break
    return [{"gate": label, "n": len(scores), "auc": auc(scores, labels),
             "sep": sep(scores, labels)}]


def main() -> int:
    print("=" * 78)
    print("LIVE ACCURACY — the loaded gates on REAL corpora")
    print("=" * 78)
    results = {"audio": probe_audio(), "image": probe_image(),
               "video": probe_video()}
    for channel, recs in results.items():
        print(f"\n--- {channel.upper()} ---")
        for r in recs:
            if r.get("error"):
                print(f"  {r['gate']:14} ERROR: {r['error']}")
                continue
            a = r.get("auc")
            a_txt = f"{a:.4f}" if a is not None else "n/a"
            print(f"  {r['gate']:14} n={r['n']:<4} AUC={a_txt}   {r['sep']}")
    out = pathlib.Path("/home/hermes/.hermes/cache/scratch/gate_accuracy.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    shutil.rmtree(Q, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
