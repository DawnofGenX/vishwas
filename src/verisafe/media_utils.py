"""Shared media utilities: ffprobe probing, frame extraction, robustness transform matrix.

All subprocess use bounded timeouts and quarantine-bound output paths so a hung
codec can never leak work out of the job dir. Thread caps honor thermal-safe
defaults on this i5-8250U box (VERISAFE_FFMPEG_THREADS, default 2).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FFMPEG = os.environ.get("VERISAFE_FFMPEG_BIN", "ffmpeg")
FFPROBE = os.environ.get("VERISAFE_FFPROBE_BIN", "ffprobe")
THREADS = int(os.environ.get("VERISAFE_FFMPEG_THREADS", "2"))   # thermal-safe cap


def _threads_args() -> list[str]:
    return ["-threads", str(THREADS)]


@dataclass(slots=True)
class ProbeInfo:
    ok: bool
    duration_s: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    vcodec: str
    acodec: str
    pix_fmt: str
    bit_rate: int          # overall container bitrate if known
    raw: dict              # raw ffprobe json (subset)

    @property
    def usable(self) -> bool:
        return self.ok and self.duration_s > 0.4


def probe(path: Path) -> ProbeInfo:
    if not shutil.which(FFPROBE):
        return ProbeInfo(False, 0.0, 0, 0, 0.0, False, False, "", "", "", 0, {})
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            return ProbeInfo(False, 0.0, 0, 0, 0.0, False, False, "", "", "", 0, {})
        j = json.loads(p.stdout)
    except Exception:
        return ProbeInfo(False, 0.0, 0, 0, 0.0, False, False, "", "", "", 0, {})
    fmt = j.get("format", {})
    vs, aus = [], []
    for s in j.get("streams", []):
        if s.get("codec_type") == "video":
            vs.append(s)
        elif s.get("codec_type") == "audio":
            aus.append(s)
    v = vs[0] if vs else {}
    a = aus[0] if aus else {}
    dur = float(fmt.get("duration") or 0.0)
    fps = 0.0
    if v:
        m = re.search(r"(\d+)/(\d+)", str(v.get("avg_frame_rate", "")))
        if m and int(m.group(2)) > 0:
            fps = int(m.group(1)) / int(m.group(2))
    br = int(fmt.get("bit_rate") or 0)
    return ProbeInfo(
        ok=bool(vs) or bool(aus),
        duration_s=dur,
        width=int(v.get("width") or 0), height=int(v.get("height") or 0),
        fps=fps,
        has_video=bool(vs), has_audio=bool(aus),
        vcodec=str(v.get("codec_name") or ""), acodec=str(a.get("codec_name") or ""),
        pix_fmt=str(v.get("pix_fmt") or ""), bit_rate=br, raw=j,
    )


def extract_frames(src: Path, outdir: Path, n: int = 8, every_s: float | None = None,
                   max_side: int = 256, jpeg_q: int = 4) -> list[Path]:
    """Deterministic sampling: every-other-ish frame up to n. All files stay inside outdir."""
    if not shutil.which(FFMPEG):
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    dur = probe(src).duration_s
    step = every_s or (dur / n if n and dur > 0 else 1.0)
    # Uniform timebase sampling — deterministic across runs for the same input.
    cmd = [FFMPEG, "-nostdin", "-i", str(src), *(_threads_args()),
           "-vf", f"fps={1.0 / max(step, 0.05)},scale={max_side}:{max_side}:force_original_aspect_ratio=decrease",
           "-vsync", "vfr", "-q:v", str(jpeg_q), "-frames:v", str(n),
           str(outdir / "%03d.jpg")]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        pass
    return sorted(outdir.glob("*.jpg"))[:n]


def extract_audio_wav(src: Path, outpath: Path, sample_rate: int = 16000, mono: bool = True) -> Path | None:
    """Pull the audio track (or full-mixdown) to 16 kHz PCM WAV for detectors."""
    if not shutil.which(FFMPEG):
        return None
    args = [FFMPEG, "-nostdin", "-y", "-i", str(src), *(_threads_args())]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(sample_rate), "-f", "wav", str(outpath)]
    try:
        subprocess.run(args, capture_output=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    return outpath if outpath.exists() else None


def apply_transform_matrix(wav_path: Path, workdir: Path) -> dict[str, Path]:
    """Real-world degradation battery (audio side). Each variant gets its own file.

    Mirrors spec: Opus/AAC/MP3 transcoding, resampling, noise, bandwidth limits.
    """
    variants: dict[str, Path] = {}
    out = workdir / "transforms"
    out.mkdir(parents=True, exist_ok=True)

    def run(name: str, extra: list[str]) -> None:
        dst = out / f"{name}.wav"
        cmd = [FFMPEG, "-nostdin", "-y", "-i", str(wav_path), *(_threads_args()), *extra, "-ar", "16000", "-f", "wav", str(dst)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            pass
        if dst.exists():
            variants[name] = dst

    run("opus_resample", ["-codec:a", "libopus", "-b:a", "24k"])       # WhatsApp voice-note style
    run("aac_44k", ["-codec:a", "aac", "-b:a", "64k"])
    run("mp3_lossy", ["-codec:a", "libmp3lame", "-b:a", "96k"])
    run("bandwidth_phone", ["-af", "highpass=f=300,lowpass=f=3400"])  # telephony band
    run("noise_snr15", ["-af", "anoisesrc=d=10:c=pink:r=15[a];[0:a][a]amix=weights=1 0.18[out]", "-map", "[out]"])
    run("resample_8k_up", ["-af", "aresample=8000,aresample=16000"])
    variants["original"] = wav_path
    return {k: v for k, v in variants.items() if v.exists()}


def apply_video_transform_matrix(video_src: Path, workdir: Path) -> dict[str, Path]:
    """Compression/degradation battery for video (H.264/H.265, resize, crop,
    fps change, screen-record simulation, WhatsApp-style transcode)."""
    out = workdir / "vid_transforms"
    out.mkdir(parents=True, exist_ok=True)
    src_info = probe(video_src)
    w = max(320, min(src_info.width or 480, 720))
    h = max(240, min(src_info.height or 360, 480))

    specs = {
        "h264_lowbitrate": [FFMPEG, "-nostdin", "-y", "-i", str(video_src), *(_threads_args()),
                            "-c:v", "libx264", "-preset", "veryfast", "-b:v", "250k", "-an"],
        "hevc_640p": [FFMPEG, "-nostdin", "-y", "-i", str(video_src), *(_threads_args()),
                      "-c:v", "libx265", "-preset", "veryfast", "-x265-params", "tune=zerolatency:keyint=60",
                      "-vf", f"scale={w}:{h}", "-b:v", "500k", "-an"],
        "resize_crop_center": [FFMPEG, "-nostdin", "-y", "-i", str(video_src), *(_threads_args()),
                               "-vf", f"crop=iw*0.7:ih*0.7,(iw-iw*0.7)/2:(ih-ih*0.7)/2,scale={w}:{h}",
                               "-c:v", "libx264", "-preset", "veryfast", "-an"],
        "fps_change_12": [FFMPEG, "-nostdin", "-y", "-i", str(video_src), *(_threads_args()),
                          "-vf", f"fps=12,scale={w}:{h}", "-c:v", "libx264", "-preset", "veryfast", "-r", "12", "-an"],
        "screen_rec_sim": [FFMPEG, "-nostdin", "-y", "-i", str(video_src), *(_threads_args()),
                           "-vf", f"scale={int(w*0.8)}:{int(h*0.8)},pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,fps=30",
                           "-c:v", "libx264", "-preset", "medium", "-an"],
        "whatsapp_style": [FFMPEG, "-nostdin", "-y", "-i", str(video_src), *(_threads_args()),
                           "-vf", f"scale={min(w,640)}:-2", "-c:v", "libx264", "-profile:v", "baseline",
                           "-level", "3.1", "-b:v", "150k", "-c:a", "aac", "-b:a", "48k"],
    }
    produced: dict[str, Path] = {}
    for name, cmd in specs.items():
        dst = out / f"{name}.mp4"
        try:
            p = subprocess.run(cmd + [str(dst)], capture_output=True, timeout=300)
            if dst.exists() and dst.stat().st_size > 2048:
                produced[name] = dst
        except subprocess.TimeoutExpired:
            continue
    produced["original"] = video_src
    return produced


def sha256_of(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
