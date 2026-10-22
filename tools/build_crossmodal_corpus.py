#!/usr/bin/env python3
"""Build a CROSS-MODAL video corpus: the ffc23 clips WITH a labelled audio track.

WHY (2026-10-21)
-----------------
The published 87-clip calibration corpus (`rows_video_v3.jsonl`) has NO AUDIO
STREAM on 83 of its 87 rows. The two cross-modal gates therefore never ran on
95% of the set the weights were tuned on:

    av_risk  (weight 3.5)  -> known_gap on 83/87
    havic    (weight 0.8)  -> known_gap on 83/87

so the "42/42 reals not flagged DO_NOT_USE" figure is really a
single-signal (effort + frameheur) result, and the live EFFORT accuracy on
the raw corpus is AUC 0.6654.

The labelled audio exists: the official ASVspoof2019-LA eval parquet
(24,844 rows, `key` 0=bonafide 1=spoof). Muxing it onto the clips gives the
cross-modal gates something real to measure:
    real clip + bonafide speech -> label 0
    fake clip + spoof speech    -> label 1

This is a SYNTHETIC corpus: the audio is not synchronised with the video
(they come from different sources), so the AV-sync gate will not have a
"natural" signal to find. It measures whether the gate is even WIRED and
whether it degrades honestly — it does NOT measure natural AV consistency.
That limitation is stated in the output and must be stated wherever the
result is cited.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

CORPUS = pathlib.Path(os.environ.get("VISHWAS_XM_CLIPS",
                                   "/home/hermes/fusion_av/ffc23/subset"))
AUDIO_PARQUET = pathlib.Path(os.environ.get("VISHWAS_XM_AUDIO",
                                      "/home/hermes/fusion_audio/asv19_la/asv19_val.parquet"))
OUT_DIR = pathlib.Path(os.environ.get("VISHWAS_XM_OUT_DIR",
                                  "/home/hermes/fusion_av/crossmodal_corpus"))
OUT_JSONL = pathlib.Path(os.environ.get("VISHWAS_XM_OUT_JSONL",
                                   "/home/hermes/fusion_av/feat_vectors/rows_crossmodal_v1.jsonl"))

N_PER_CLASS = 30
FPS = 25
SIZE = (640, 480)


def load_audio_rows(n_per_class):
    """Pull n bonafide + n spoof wavs from the official eval parquet."""
    import pyarrow.parquet as pq
    import soundfile as sf

    pf = pq.ParquetFile(AUDIO_PARQUET)
    out = {0: [], 1: []}
    rg = 0
    dropped = 0
    while (len(out[0]) < n_per_class or len(out[1]) < n_per_class) and rg < pf.metadata.num_row_groups:
        t = pf.read_row_group(rg)
        rg += 1
        keys = t.column("key").to_pylist()
        names = t.column("audio_file_name").to_pylist()
        aud = t.column("audio").to_pylist()
        for k, nm, a in zip(keys, names, aud):
            k = int(k)
            if len(out[k]) >= n_per_class:
                continue
            b = a["bytes"] if isinstance(a, dict) else a
            try:
                x, sr = sf.read(io.BytesIO(b), dtype="float32")
            except Exception as e:  # noqa: BLE001
                # ⚠️ COUNTED, NOT SILENTLY LOST (2026-10-22). This used to be
                # `except Exception: continue`: a parquet whose audio column
                # was unreadable would build a SHORT corpus with no warning,
                # and every AUC measured on it afterwards would be over a
                # denominator nobody chose.
                dropped += 1
                if dropped <= 3:
                    print(f"  ⚠️ row decode failed ({type(e).__name__}: {e})")
                continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                import scipy.signal as ss
                x = ss.resample(x, 16000 / sr).astype(np.float32)
            out[k].append((nm, x, sr))
    if dropped:
        print(f"  ⚠️ load_audio_rows: {dropped} parquet row(s) failed to "
              f"decode and were DROPPED from the corpus")
    for k in (0, 1):
        if len(out[k]) < n_per_class:
            print(f"  ⚠️ CORPUS UNDER-FILLED: class {k} has {len(out[k])} of "
                  f"{n_per_class} requested — every AUC measured on this "
                  f"corpus will be over a smaller denominator")
    return out[0], out[1]


def mux(clip: pathlib.Path, wav: pathlib.Path, dest: pathlib.Path):
    """Mux the wav into the clip, keeping the video stream as-is."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(clip), "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "96k",
        "-shortest", str(dest),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=300)
    if r.returncode != 0:
        return False
    # verify the output really has an audio stream
    chk = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(dest)],
        capture_output=True, text=True, timeout=60)
    return bool(chk.stdout.strip())


def main():
    if not CORPUS.is_dir():
        print(f"FATAL: clip corpus missing: {CORPUS}", file=sys.stderr)
        return 1
    if not AUDIO_PARQUET.exists():
        print(f"FATAL: audio parquet missing: {AUDIO_PARQUET}", file=sys.stderr)
        return 1

    bona, spoof = load_audio_rows(N_PER_CLASS)
    print(f"audio: {len(bona)} bonafide + {len(spoof)} spoof from the official eval")

    real_clips = sorted((CORPUS / "real").glob("*.mp4")) + sorted((CORPUS / "real2").glob("*.mp4"))
    fake_clips = sorted((CORPUS / "fake").glob("*.mp4")) + sorted((CORPUS / "fake2").glob("*.mp4"))
    print(f"clips: {len(real_clips)} real + {len(fake_clips)} fake")

    rows = []
    for label, clips, audios, tag in (
        (0, real_clips, bona, "real"),
        (1, fake_clips, spoof, "fake"),
    ):
        for i, (clip, (aname, x, sr)) in enumerate(zip(clips, audios)):
            if i >= N_PER_CLASS:
                break
            # write the audio to a temp wav
            import soundfile as sf
            tmpwav = OUT_DIR / "_tmp" / f"{tag}_{i}.wav"
            tmpwav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(tmpwav), x, sr)
            dest = OUT_DIR / tag / f"{tag}_{i:03d}.mp4"
            ok = mux(clip, tmpwav, dest)
            tmpwav.unlink(missing_ok=True)
            if not ok:
                print(f"  mux FAILED: {clip.name} + {aname}")
                continue
            # probe the result
            pr = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_type,width,height,duration", "-of", "json", str(dest)],
                capture_output=True, text=True, timeout=60)
            meta = json.loads(pr.stdout or "{}")
            has_audio = any(s.get("codec_type") == "audio" for s in meta.get("streams", []))
            dur = next((s.get("duration") for s in meta.get("streams", [])
                        if s.get("codec_type") == "video"), None)
            rows.append({
                "path": str(dest),
                "label": label,
                "subclass": tag,
                "source_clip": str(clip),
                "audio_source": aname,
                "audio_label": "bonafide" if label == 0 else "spoof",
                "has_audio": has_audio,
                "duration_s": dur,
            })
            print(f"  {tag}: {clip.name} + {aname} -> {dest.name} "
                  f"(audio={has_audio}, dur={dur})")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    n_real = sum(1 for r in rows if r["label"] == 0)
    n_fake = sum(1 for r in rows if r["label"] == 1)
    n_audio = sum(1 for r in rows if r["has_audio"])
    print(f"\nWROTE {OUT_JSONL}")
    print(f"  rows={len(rows)}  real={n_real}  fake={n_fake}  with_audio={n_audio}")
    print("  NOTE: this is a SYNTHETIC cross-modal corpus — the audio is not")
    print("  synchronised with the video. It measures whether the cross-modal")
    print("  gates are WIRED and degrade honestly; it does NOT measure natural")
    print("  AV consistency. Do not cite it as an AV-sync accuracy number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
