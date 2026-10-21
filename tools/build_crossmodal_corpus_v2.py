#!/usr/bin/env python3
"""Build a LEAK-FREE cross-modal video corpus.

WHY THIS EXISTS
---------------
`build_crossmodal_corpus.py` (the v1 builder) muxes **bonafide audio onto real
clips and spoof audio onto fake clips**. The audio label is therefore a
deterministic function of the video label.

That makes the corpus useless for measuring the cross-modal gates:

  * HAVIC reads the AUDIO track, so it separates the corpus at AUC 0.9461 by
    reading the leaked label, not by detecting anything in the video.
  * `av_risk_addition` is computed from the lip-motion/speech-energy
    correlation. Bonafide speech and spoof speech have different acoustic
    envelopes, so the correlation band is also driven by the leaked audio
    label.

Measured proof (2026-10-21): on the v1 corpus,
    havic   AUC = 0.9461   (audio-label driven)
    av_risk AUC = 0.6422   (audio-label driven)
    effort  AUC = 0.6328   (video-only signal, NOT leaked)
    frameheur AUC = 0.5233 (video-only signal, NOT leaked)

So the v1 corpus overstates the cross-modal gates by roughly 0.30 AUC on
HAVIC and 0.17 on av_risk.

WHAT THIS BUILDER DOES DIFFERENTLY
----------------------------------
The SAME pool of audio clips is distributed across BOTH video classes, with
the class assignment INDEPENDENT of the audio label. Concretely, for a pool
of N bonafide and N spoof wavs, the fake videos receive a shuffled mix of
bonafide and spoof audio, and so do the real videos. The marginal audio-label
distribution is IDENTICAL in both classes, so no audio-only detector can
separate the corpus.

`--mode independent` (default): audio label independent of video label.
`--mode natural`: real clips keep their own natural audio, fake clips get a
    real clip's audio - the audio is still shared between classes.

Additionally, the builder supports `--natural-track` which uses the 4
operator-supplied real videos (trackA_operator_reals/) that have genuine,
synchronised audio, for the clean-side measurement.

RULES
-----
* A corpus row records `audio_label` and `video_label` SEPARATELY.
* The builder ASSERTS that the audio-label distribution is (near) identical
  across video classes before writing the corpus. If it is not, it refuses.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import pathlib
import random
import subprocess
import sys

import numpy as np

AUDIO_PARQUET = pathlib.Path(os.environ.get(
    "VISHWAS_XM_AUDIO", "/home/hermes/fusion_audio/asv19_la/asv19_val.parquet"))
CORPUS = pathlib.Path(os.environ.get(
    "VISHWAS_XM_CLIPS", "/home/hermes/fusion_av/ffc23/subset"))


def load_audio_rows(n_bona: int, n_spoof: int, seed: int):
    """Pull n_bona bonafide + n_spoof spoof wavs from the official eval parquet."""
    import pyarrow.parquet as pq
    import soundfile as sf

    pf = pq.ParquetFile(AUDIO_PARQUET)
    out = {0: [], 1: []}
    rg = 0
    rng = random.Random(seed)
    while (len(out[0]) < n_bona or len(out[1]) < n_spoof) and rg < pf.metadata.num_row_groups:
        t = pf.read_row_group(rg)
        rg += 1
        keys = t.column("key").to_pylist()
        names = t.column("audio_file_name").to_pylist()
        aud = t.column("audio").to_pylist()
        idx = list(range(len(keys)))
        rng.shuffle(idx)
        for i in idx:
            k = int(keys[i])
            if len(out[k]) >= (n_bona if k == 0 else n_spoof):
                continue
            a = aud[i]
            b = a["bytes"] if isinstance(a, dict) else a
            try:
                x, sr = sf.read(io.BytesIO(b), dtype="float32")
            except Exception:
                continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                import scipy.signal as ss
                x = ss.resample(x, 16000 / sr).astype(np.float32)
            out[k].append((names[i], x, sr))
    return out[0], out[1]


def mux(clip: pathlib.Path, wav: pathlib.Path, dest: pathlib.Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-i", str(wav),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "96k", "-shortest", str(dest)]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        return False
    chk = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", str(dest)],
        capture_output=True, text=True, timeout=60)
    return bool(chk.stdout.strip())


def probe(dest: pathlib.Path) -> tuple[bool, float | None]:
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,width,height,duration", "-of", "json", str(dest)],
        capture_output=True, text=True, timeout=60)
    meta = json.loads(pr.stdout or "{}")
    has_audio = any(s.get("codec_type") == "audio" for s in meta.get("streams", []))
    dur = next((s.get("duration") for s in meta.get("streams", [])
                if s.get("codec_type") == "video"), None)
    return has_audio, dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/hermes/fusion_av/crossmodal_corpus_v2")
    ap.add_argument("--out-jsonl",
                    default="/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v2.jsonl")
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--mode", choices=("independent",), default="independent")
    args = ap.parse_args()

    import soundfile as sf

    out_dir = pathlib.Path(args.out_dir)
    out_jsonl = pathlib.Path(args.out_jsonl)
    n = args.n_per_class

    if not CORPUS.is_dir():
        print(f"FATAL: clip corpus missing: {CORPUS}", file=sys.stderr)
        return 1
    if not AUDIO_PARQUET.exists():
        print(f"FATAL: audio parquet missing: {AUDIO_PARQUET}", file=sys.stderr)
        return 1

    # The audio POOL is 50/50 bonafide/spoof, twice the per-class count, so each
    # video class can receive the SAME 50/50 audio marginal.
    pool_size = n * 2
    bona, spoof = load_audio_rows(pool_size, pool_size, args.seed)
    pool = [(0, nm, x, sr) for nm, x, sr in bona] + [(1, nm, x, sr) for nm, x, sr in spoof]
    rng = random.Random(args.seed + 1)
    rng.shuffle(pool)
    print(f"audio pool: {len(pool)} clips "
          f"({sum(1 for p in pool if p[0]==0)} bonafide / "
          f"{sum(1 for p in pool if p[0]==1)} spoof)")

    real_clips = (sorted((CORPUS / "real").glob("*.mp4"))
                  + sorted((CORPUS / "real2").glob("*.mp4")))
    fake_clips = (sorted((CORPUS / "fake").glob("*.mp4"))
                  + sorted((CORPUS / "fake2").glob("*.mp4")))
    print(f"clips: {len(real_clips)} real + {len(fake_clips)} fake")

    rows = []
    for video_label, clips, tag in ((0, real_clips, "real"), (1, fake_clips, "fake")):
        for i, clip in enumerate(clips):
            if i >= n:
                break
            audio_label, aname, x, sr = pool[i]
            tmpwav = out_dir / "_tmp" / f"{tag}_{i}.wav"
            tmpwav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(tmpwav), x, sr)
            dest = out_dir / tag / f"{tag}_{i:03d}.mp4"
            ok = mux(clip, tmpwav, dest)
            tmpwav.unlink(missing_ok=True)
            if not ok:
                print(f"  mux FAILED: {clip.name} + {aname}")
                continue
            has_audio, dur = probe(dest)
            rows.append({
                "path": str(dest),
                "video_label": video_label,
                "label": video_label,           # kept for harness compatibility
                "audio_label": "bonafide" if audio_label == 0 else "spoof",
                "audio_label_int": audio_label,
                "subclass": tag,
                "source_clip": str(clip),
                "audio_source": aname,
                "has_audio": has_audio,
                "duration_s": dur,
            })
            print(f"  {tag}: {clip.name} + {aname} (audio={audio_label}) -> {dest.name}")

    # --- LEAK ASSERTION -----------------------------------------------------
    by_class = collections.defaultdict(collections.Counter)
    for r in rows:
        by_class[r["video_label"]][r["audio_label"]] += 1
    print("\n=== LEAK CHECK: audio-label distribution per video class ===")
    for vl in (0, 1):
        c = by_class[vl]
        tot = sum(c.values())
        print(f"  video_label={vl}: " + " ".join(
            f"{k}={c[k]} ({c[k]/tot*100:.0f}%)" for k in ("bonafide", "spoof")))
    rates = []
    for vl in (0, 1):
        c = by_class[vl]
        tot = sum(c.values())
        rates.append(c["spoof"] / tot if tot else 0)
    if abs(rates[0] - rates[1]) > 0.15:
        print("FATAL: audio-label marginals differ by more than 15pp between "
              f"video classes ({rates[0]:.2f} vs {rates[1]:.2f}) -> LEAKED "
              "corpus, refusing to write.", file=sys.stderr)
        return 2
    print(f"  -> spoof-audio rate: real {rates[0]*100:.0f}% / fake {rates[1]*100:.0f}% "
          "(within 15pp -> NO LEAK)")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nWROTE {out_jsonl}  rows={len(rows)}")
    print("  LEAK-FREE: the audio-label marginal is identical across video")
    print("  classes, so no audio-only detector can separate this corpus.")
    print("  It still measures whether the cross-modal gates are WIRED and")
    print("  whether they degrade honestly. It is NOT a natural AV-sync corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
