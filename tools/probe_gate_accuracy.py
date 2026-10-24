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


def eer(scores: list[float], labels: list[int]) -> float | None:
    """Equal Error Rate: the threshold minimising (FAR + FRR) / 2.

    ⚠️ THE DIRECTION MATTERS (2026-10-22). A "fake" decision is
    `score >= th`, so the FALSE-ACCEPT rate is (real scored >= th) and the
    FALSE-REJECT rate is (spoof scored < th). The first version had them
    SWAPPED, which reported **50% EER for a perfectly separating
    detector** -- the numbers were the giveaway, not the code. A detector
    that separates its classes perfectly cannot have a 50% error rate.

    ⚠️ AND A NaN SCORE IS NOT A SCORE. A NaN would compare False against
    every threshold and silently land on whichever side, so NaN is
    dropped before the computation.
    """
    pairs = [(s, l) for s, l in zip(scores, labels)
             if s == s and s is not None]
    if not pairs:
        return None
    pos = [s for s, l in pairs if l == 1]
    neg = [s for s, l in pairs if l == 0]
    if not pos or not neg:
        return None
    # ⚠️ THE THRESHOLD SEARCH MUST COVER BOTH POLARITIES.
    # An INVERTED detector (fake LOW, real HIGH) is perfectly separated,
    # but the decision rule "score >= th => fake" cannot express "fake is
    # the LOW score" -- so a one-sided search reports 50% for a detector
    # that is in fact perfect.
    #
    # The EER is therefore computed over BOTH polarities: for each
    # threshold, take the better of "fake is high" and "fake is low".
    # That is the standard threshold-free EER, and it is what makes the
    # number mean "how well can this score separate the classes" rather
    # than "is the score oriented the right way".
    #
    # (The AUDIENCE cares about orientation: a detector that scores
    # bonafide HIGH would read as "everything is fake". That is a
    # SEPARATE defect, and it shows up in the AUC -- an AUC near 0.0 --
    # not in the EER.)
    vals = sorted(set(s for s, _ in pairs))
    lo = min(0.0, vals[0] - 1.0)
    hi = max(1.0, vals[-1] + 1.0)
    thresholds = [lo] + vals + [hi]
    best = 1.0
    for th in thresholds:
        # polarity A: score >= th means fake
        far_a = sum(1 for s in neg if s >= th) / len(neg)
        frr_a = sum(1 for s in pos if s < th) / len(pos)
        # polarity B: score <= th means fake (the INVERTED reading)
        far_b = sum(1 for s in neg if s <= th) / len(neg)
        frr_b = sum(1 for s in pos if s > th) / len(pos)
        best = min(best, (far_a + frr_a) / 2.0, (far_b + frr_b) / 2.0)
    return best


def sep(scores: list[float], labels: list[int]) -> str:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return "n/a"
    out = (f"fake med {st.median(pos):.4f} (min {min(pos):.4f}) | "
           f"real med {st.median(neg):.4f} (max {max(neg):.4f})")
    e = eer(scores, labels)
    if e is not None:
        out += f" | EER {e*100:.2f}%"
    mis = (sum(1 for s in pos if s < 0.5)
           + sum(1 for s in neg if s >= 0.5))
    out += f" | misclassified @0.5: {mis} of {len(scores)}"
    return out


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

        scores, labels, errs, skipped = [], [], 0, 0
        dropped = []
        for k, (idx, lab) in enumerate(picks):
            # ⚠️ COUNT EVERY DROP (2026-10-22). These two branches used to be
            # a bare `except Exception: continue` -- the exact shape that
            # made the image channel report n=59 on a 60-image corpus
            # (PR #34) and made the audio channel report "the model is
            # broken" when the PROBE was broken (n=0, PR #27). A number
            # does not advertise its own missingness, so the drop is
            # counted AND attributed.
            try:
                data, sr = sf.read(io.BytesIO(audio[idx]["bytes"]),
                                   dtype="float32")
            except Exception as e:  # noqa: BLE001
                skipped += 1
                dropped.append(f"row {idx}: decode {type(e).__name__}")
                continue
            if data.ndim > 1:
                data = data.mean(axis=1)
            w = wavdir / f"{label}_{k}.wav"
            try:
                sf.write(str(w), data, sr)
            except Exception as e:  # noqa: BLE001
                skipped += 1
                dropped.append(f"row {idx}: write {type(e).__name__}")
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
        if skipped:
            print(f"    ⚠️ audio/{label}: {skipped} of {len(picks)} did not "
                  f"score (corpus items DROPPED, not model failures)")
            for why in dropped[:5]:
                print(f"        {why}")
            if len(dropped) > 5:
                print(f"        ... and {len(dropped) - 5} more")
        rec = {"gate": label, "n": len(scores), "errors": errs,
               "skipped": skipped, "auc": auc(scores, labels),
               "sep": sep(scores, labels)}
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
    total_skipped = 0
    for sub, lab in (("ai", 1), ("real", 0)):
        d = IMG_DIR / sub
        if not d.is_dir():
            continue
        # ⚠️ FILTER BEFORE YOU SLICE (2026-10-22). This used to be
        #     for f in sorted(d.iterdir())[:n]:
        #         if f.suffix.lower() not in (...): continue
        # which slices FIRST and filters SECOND. `fusion_img/ai/` contains
        # 31 entries: 30 images plus a `sources.csv` manifest. `sources.csv`
        # sorts BEFORE `stable_diffusion_3_*`, so the `[:30]` slice
        # dropped a REAL AI IMAGE (stable_diffusion_3_07.webp) and kept the
        # CSV -- which was then skipped by the extension check, and the
        # shortfall was INVISIBLE: the probe reported n=59 for a 60-image
        # corpus.
        #
        # ⇒ A SILENT SKIP HIDES A BROKEN CORPUS AND LOOKS IDENTICAL TO A
        # MODEL THAT FAILED. The count must be checked against the corpus
        # size, and the corpus size is printed below.
        files = [f for f in sorted(d.iterdir())
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        if not files:
            continue
        corpus_n = len(files)
        files = files[:n]
        skipped = 0
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
            try:
                s = model.predict(img)
            except Exception:
                skipped += 1
                continue
            if s is None:
                skipped += 1
                continue
            scores.append(float(s))
            labels.append(lab)
        if skipped:
            print(f"    ⚠️ image/{sub}: {skipped} of {len(files)} did not score")
        if len(files) < corpus_n:
            # "sampled", not "scored": `files` is the SAMPLE the probe took.
            # Conflating the two hid a drop for a day (PR #34).
            print(f"    (image/{sub}: sampled {len(files)} of {corpus_n} "
                  f"available, scored {len(files) - skipped})")
        total_skipped += skipped
    return [{"gate": label, "n": len(scores), "skipped": total_skipped,
             "auc": auc(scores, labels), "sep": sep(scores, labels)}]


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
    skipped = 0
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
                skipped += 1
                continue
            checks = cap._effort(ctx, frames)
        except Exception:
            skipped += 1
            continue
        for c in checks:
            if c.name == "effort_face_forensics" and c.status == "ok":
                v = c.signals.get("prob_deepfake")
                if v is not None:
                    scores.append(float(v))
                    labels.append(lab)
                    break
        else:
            skipped += 1
    if skipped:
        print(f"    ⚠️ video: {skipped} of {len(clips)} clips did not score")
    return [{"gate": label, "n": len(scores), "skipped": skipped,
             "auc": auc(scores, labels), "sep": sep(scores, labels)}]


def _expected_counts() -> dict:
    """How many items each corpus SHOULD yield.

    ⚠️ THE PROBE SAMPLES, SO THE EXPECTED COUNT IS THE SAMPLE SIZE, NOT
    THE CORPUS SIZE (2026-10-22). The image corpus has 30 AI + 30 real
    and the probe takes 30 per class, so 60. The audio corpus is
    24,844 rows and the probe takes 60 per class, so 120. The video
    corpus is 20 clips in `fake/`+`real/` and the probe takes 40 per
    class, so 20.

    This exists so a SILENT SKIP is visible. Every channel does
    `except Exception: continue`, so a model that fails on everything
    and a corpus that is missing an item look IDENTICAL in the output.
    """
    counts = {}
    if IMG_DIR.is_dir():
        total = 0
        for sub in ("ai", "real"):
            d = IMG_DIR / sub
            if d.is_dir():
                total += min(30, len([f for f in d.iterdir()
                                      if f.suffix.lower() in
                                      (".jpg", ".jpeg", ".png", ".webp")]))
        counts["image"] = total
    if PARQUET.exists():
        counts["audio"] = 2 * 60
    if VID_DIR.is_dir():
        total = 0
        for sub in ("fake", "real"):
            d = VID_DIR / sub
            if d.is_dir():
                total += min(40, len(list(d.glob("*.mp4"))))
        counts["video"] = total
    return counts


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
    # ⚠️ A COUNT THAT DOES NOT MATCH THE CORPUS IS A RED FLAG (2026-10-22).
    # Every probe channel SILENTLY SKIPS what it cannot score, so a broken
    # model and a broken corpus look IDENTICAL in the output. The corpus
    # sizes are printed so a shortfall is visible rather than inferred.
    print()
    print("=== CORPUS vs MEASURED (a shortfall means something was dropped) ===")
    expected = _expected_counts()
    for channel, recs in results.items():
        for r in recs:
            if r.get("error"):
                continue
            exp = expected.get(channel)
            if exp is None:
                continue
            n = r.get("n", 0)
            sk = r.get("skipped", 0)
            if sk:
                print(f"  ⚠️ {channel:6} {r['gate']:14} {sk} corpus item(s) "
                      f"were DROPPED before scoring (see the per-channel "
                      f"warning above)")
            if n != exp:
                print(f"  ⚠️ {channel:6} {r['gate']:14} measured n={n} but the "
                      f"corpus has {exp} — SOMETHING WAS DROPPED")
            else:
                print(f"  ok  {channel:6} {r['gate']:14} n={n} (all of the corpus)")
    shutil.rmtree(Q, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
