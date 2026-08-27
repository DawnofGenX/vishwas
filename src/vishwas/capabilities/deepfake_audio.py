"""Deepfake-audio detection capability.

Detector lineup (spec): FakeMamba (SOTA mamba), AASIST (classic CNN-BiLSTM),
plus a complementary SSL-based detector (wav2vec2-HuBERT probe). All heavy
weights behind gates; offline fallback extracts 13-dim MFCCs + spectral
flatness so a bare CPU box still emits *some* signal (degraded status).
Multi-crop inference: N offset crops are scored and medians aggregated to
reduce single-artifact sensitivity. Robustness: apply_transform_matrix()
re-runs Opus/AAC/MP3/resample/bandwidth/noise variants — the spread across
variants feeds reliability and adversarial-evasion testing.
"""
from __future__ import annotations

import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..device import resolve_device
from ..events import Artifact, JobContext
from ..llm_guard import LLMClient, build_interpretation_prompt, interpret_with_fallback
from ..media_utils import extract_audio_wav, apply_transform_matrix, probe
from ..model_adapters import resolve as _resolve_adapter, _call_model as _call_model_compat, _auto_extract, is_usable_model as _is_usable_model
from .base import CheckResult


# 2026-08-26 audio-trustworthy fix: the webhook's docling-python tree cannot load
# Wav2Vec2Model (transformers-5.15 + accelerate circular-import / lib.GEN_EMAIL),
# so in-process aasist3 _load_weights() returns None -> missing_dependency -> the
# audio channel read MEDIUM/UNVERIFIED. The ONE env that loads it is the isolated
# .venv-ambient tree (proven: bonafide 0.0000 / spoof 0.9966 on the official eval).
# deepfake_audio shells out to this helper for the aasist3 (torch) crop when the
# in-process arch is unavailable. Score-of-record polarity matches Track C.
_AASIST3_HELPER = str(Path(__file__).resolve().parents[2] / "scripts" / "audio_score_aasist3.py")
_VENV_AMBIENT_PY = "/home/hermes/.venv-ambient/bin/python"
_AASIST3_SUBPROC_TIMEOUT_S = 150  # first load ~37s + inference; bounded


def _subprocess_aasist3_score(crop: Path, device: str, timeout_s: float = _AASIST3_SUBPROC_TIMEOUT_S) -> float | None:
    """Score one crop via the .venv-ambient helper. Returns spoof posterior [0,1]
    or None on any failure (never fabricates)."""
    try:
        r = subprocess.run(
            [_VENV_AMBIENT_PY, _AASIST3_HELPER, str(crop), "--device", device or "cpu"],
            capture_output=True, text=True, timeout=timeout_s)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        x = line.strip()
        if x.startswith("#") or not x:
            continue
        try:
            return float(x)
        except ValueError:
            continue
    return None  # no parseable posterior line


def _load_weights(env: str):
    """Registry-gated loader; returns None unless env path exists AND torch is
    importable. Routes through model_adapters so AASIST/SSL/FakeMamba families
    get correct preprocessing / head extraction."""
    p = os.environ.get(env)
    if not p or not os.path.exists(p):
        return None
    adapter = _resolve_adapter(env)
    if adapter is not None:
        obj = adapter.load(p)
        return obj if _is_usable_model(obj) else None
    # unregistered env var -> legacy behaviour (plain torch.load + .predict)
    try:
        import torch  # type: ignore
        obj = torch.load(p, map_location=resolve_device(), weights_only=False)
        return obj if _is_usable_model(obj) else None
    except Exception:
        return None


def _infer_prob(adapter, model, raw_input) -> float | None:
    """One inference via the registry adapter (or legacy .predict). [0,1] or None;
    never raises."""
    if adapter is not None:
        processed = adapter.preprocess(raw_input)
        out = _call_model_compat(model, processed)
        prob = adapter.extract_prob(out) if adapter.extract_prob else None
        if prob is None:
            prob = _auto_extract(out)
        return prob
    # Legacy uniform contract
    if hasattr(model, "predict"):
        out = model.predict(raw_input)
        val = out[0] if isinstance(out, (list, tuple)) else out
        p = float(val)
        return min(1.0, max(0.0, p))
    return None


class DeepfakeAudioCapability:
    requires: tuple[str, ...] = ()
    stage_cost = "heavy"   # 2.1: eligible for non-blocking budget + follow-up

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        pi = probe(art.path)
        if not pi.has_audio and pi.duration_s <= 0.4:
            out.append(CheckResult("audio_probe", "cheap", "failed", {},
                                   "no usable audio stream found for forensics"))
            return out
        wav = extract_audio_wav(art.path, ctx.quarantine_root / "aud.wav")
        if wav is None or not wav.exists():
            out.append(CheckResult("audio_probe", "cheap", "failed", {},
                                   "cannot decode audio to PCM; insufficient evidence"))
            return out
        out.append(CheckResult("audio_probe", "cheap", "ok",
                               {"duration_s": round(pi.duration_s, 2),
                                "sample_rate_out": 16000, "pcm_present": True},
                               f"audio normalized to 16 kHz mono PCM ({pi.duration_s:.1f}s)"))

        # Offline baseline: mel-ish features (always available)
        heur = self._offline_features(wav)
        out.extend(heur)

        # Gated specialists, multi-crop
        n_crops = 3
        out.extend(self._multi_crop("fakemamba", "VISHWAS_FAKEMAMBA_WEIGHTS", n_crops, ctx))
        out.extend(self._multi_crop("aasist", "VISHWAS_AASIST_WEIGHTS", n_crops, ctx))
        out.extend(self._xlsr_detector(ctx, wav))
        out.extend(self._ssl_detector(ctx, wav))

        # Degradation battery (adversarial-robustness input)
        strong_positive = any(c.usable() and (c.signals.get("prob_deepfake") or 0) > 0.5
                              for c in out if c.name.startswith(("fakemamba", "aasist")))
        if strong_positive and ctx.remaining_s() > 60:
            out.extend(self._degradation_consistency(ctx, wav))
        elif strong_positive:
            out.append(CheckResult("audio_degradation_consistency", "heavy", "skipped",
                                   {}, "budget exhausted before audio transcode battery could run"))
        return out

    # ---------------------------------------------------------- helpers ----
    def _offline_features(self, wav_path: Path) -> list[CheckResult]:
        feats = _extract_mfcc_stats(wav_path)
        if feats.get("mel_band_energy_ratio") is None or not feats.get("usable_audio", True):
            return [CheckResult("audio_offline_features", "mid", "degraded",
                                {**feats, "prob_deepfake": None},
                                "no usable acoustics (near-silence / unparseable audio); "
                                "abstaining rather than scoring noise")]
        prob = min(1.0, 0.3 * min(1.0, feats["mel_band_energy_ratio"] * 3)
                   + 0.2 * (1.0 if (feats.get("spectral_flatness_mean") or 1.0) < 0.05 else 0.0))
        notes = "offline MFCC/spectral baseline (fallback when heavy models absent)"
        return [CheckResult("audio_offline_features", "mid",
                            "ok" if feats else "degraded",
                            {"prob_deepfake": round(prob, 3),
                             **feats,
                             "source": "offline_feature_baseline"},
                            notes)]

    def _multi_crop(self, model_id: str, env: str, n_crops: int, ctx: JobContext) -> list[CheckResult]:
        name = model_id.lower()
        m = _load_weights(env)
        # 2026-08-26 audio fix (subprocess fallback): when the in-process arch
        # cannot load on the webhook's docling-python/transformers-5.15 stack
        # (Wav2Vec2Model import fails -> None), the AASIST lane falls back to the
        # proven .venv-ambient helper via subprocess so real audio reads LOW and
        # fake audio reads HIGH instead of everything landing MEDIUM/UNVERIFIED.
        # Only falls back when weights ARE provisioned (env path exists) but the
        # in-process load failed — a genuine provisioning gap still reports the
        # "weights not provisioned" unavailable. AASIST-only (fakemamba/ssl stay
        # in-process: their arch returns None in .venv-ambient too — secondary).
        _weights_set = os.environ.get(env) and os.path.exists(os.environ.get(env, ""))
        if m is None and _weights_set and name == "aasist" and env == "VISHWAS_AASIST_WEIGHTS":
            return self._multi_crop_subprocess_aasist(ctx, name, n_crops)
        if m is None:
            return [CheckResult(name + "_detector", "heavy", "unavailable",
                                 {"missing_dependency": "model-weights"},
                                 f"{name.upper()} weights not provisioned ({env}); skipped")]
        adapter = _resolve_adapter(env)
        crops = _crop_windows(ctx.quarantine_root / "crops", n_crops)
        probs: list[float] = []
        for cp in crops:
            try:
                p = _infer_prob(adapter, m, cp)
                if p is not None:
                    probs.append(min(1.0, max(0.0, p)))
            except Exception:
                continue
        if not probs:
            return [CheckResult(name + "_detector", "heavy", "failed", {},
                                "no decodable crop windows produced a score")]
        return [CheckResult(name + "_detector", "heavy", "ok",
                            {"prob_deepfake": round(statistics.median(probs), 3),
                             "n_crops_scored": len(probs),
                             "max_prob": round(max(probs), 3)},
                            f"{name} pass, median over {len(probs)} crop window(s)")]

    def _multi_crop_subprocess_aasist(self, ctx: JobContext, name: str, n_crops: int) -> list[CheckResult]:
        """AASIST3 via the .venv-ambient subprocess helper (the env that can load
        Wav2Vec2Model). Scores the crop windows, returns the median spoof posterior.
        Honest unavailable (never a fabricated score) on helper failure."""
        crops = _crop_windows(ctx.quarantine_root / "crops", n_crops)
        if not crops:
            return [CheckResult(name + "_detector", "heavy", "unavailable", {},
                                "no crop windows to score (audio decode unavailable); aasist3 skipped")]
        device = os.environ.get("VISHWAS_DEVICE", "cpu")
        probs = [_subprocess_aasist3_score(cp, device) for cp in crops]
        probs = [p for p in probs if p is not None and 0.0 <= p <= 1.0]
        if not probs:
            return [CheckResult(name + "_detector", "heavy", "unavailable",
                                {"missing_dependency": "subprocess-env"},
                                "aasist3 subprocess scoring failed (see .venv-ambient/helper); audio evidence unavailable")]
        return [CheckResult(name + "_detector", "heavy", "ok",
                            {"prob_deepfake": round(statistics.median(probs), 3),
                             "n_crops_scored": len(probs),
                             "max_prob": round(max(probs), 3),
                             "source": "aasist3.subprocess(.venv-ambient)"},
                            f"aasist3 subprocess, median spoof {round(statistics.median(probs),3)} over {len(probs)} crop(s)")]

    def _xlsr_detector(self, ctx: JobContext, wav_path: Path) -> list[CheckResult]:
        """XLSR-Mamba-LA second opinion (MIT, arXiv 2411.10027) — independent
        Mamba-family architecture over a wav2vec2-XLSR frontend. Its checkpoint
        trains bonafide=1/spoof=0 (INVERTED vs fakemamba); the spec's score()
        already returns the spoof posterior, so no caller-side flip is needed."""
        env = "VISHWAS_XLSRMAMBA_WEIGHTS"
        m = _load_weights(env)
        if m is None:
            return [CheckResult("xlsr_audio_detector", "heavy", "unavailable",
                                {"missing_dependency": "model-weights"},
                                "XLSR-Mamba weights not provisioned; single-family coverage only")]
        adapter = _resolve_adapter(env)
        try:
            p = _infer_prob(adapter, m, str(wav_path))
            if p is None:
                return [CheckResult("xlsr_audio_detector", "heavy", "degraded",
                                    {}, "XLSR-Mamba detector produced no usable score")]
            return [CheckResult("xlsr_audio_detector", "heavy", "ok",
                                {"prob_deepfake": round(min(1.0, max(0.0, p)), 3)},
                                "XLSR-Mamba second opinion (independent arch family)")]
        except Exception as e:  # noqa: BLE001
            return [CheckResult("xlsr_audio_detector", "heavy", "failed",
                                {"error_class": e.__class__.__name__}, "XLSR detector inference error")]

    def _ssl_detector(self, ctx: JobContext, wav_path: Path) -> list[CheckResult]:
        """Complementary SSL probe (wav2vec2-HuBERT class) for diversity against
        a single model family being fooled by one artifact type."""
        env = "VISHWAS_SSL_AUDIO_WEIGHTS"
        m = _load_weights(env)
        if m is None:
            return [CheckResult("ssl_audio_detector", "heavy", "unavailable",
                                {"missing_dependency": "model-weights"},
                                "SSL-complement weights not provisioned; single-family coverage only")]
        adapter = _resolve_adapter(env)
        try:
            p = _infer_prob(adapter, m, str(wav_path))
            if p is None:
                return [CheckResult("ssl_audio_detector", "heavy", "degraded",
                                    {}, "SSL detector produced no usable score")]
            return [CheckResult("ssl_audio_detector", "heavy", "ok",
                                {"prob_deepfake": round(min(1.0, max(0.0, p)), 3)},
                                "SSL-based complementary pass (diversity guard vs FakeMamba/AASIST family)")]
        except Exception as e:  # noqa: BLE001
            return [CheckResult("ssl_audio_detector", "heavy", "failed",
                                {"error_class": e.__class__.__name__}, "SSL detector inference error")]

    def _degradation_consistency(self, ctx: JobContext, wav_path: Path) -> list[CheckResult]:
        matrix = apply_transform_matrix(wav_path, ctx.quarantine_root)
        scored: dict[str, float | None] = {}
        for name, p in matrix.items():
            h = self._offline_features(p)
            scored[name] = (h[0].prob if h else None)
        vals = [v for v in scored.values() if v is not None]
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else None
        consistent = (spread is None) or (spread <= 0.4)
        notes = ("score stable across Opus/AAC/MP3/resample/bandwidth/noise battery"
                 if consistent else
                 f"score shifts under realistic degradation (spread={spread:.2f}); likely codec-sensitive region — lower confidence")
        return [CheckResult("audio_degradation_consistency", "heavy",
                            "ok", 
                            {"variants": sorted(scored.keys()),
                             "prob_by_variant": {k: (round(v, 3) if v is not None else None) for k, v in scored.items()},
                             "consistency_spread": round(spread, 3) if spread is not None else None,
                             "consistent": bool(consistent)},
                            notes)]


# ------------------------------------------------------------ internals --

def _extract_mfcc_stats(wav_path: Path, n_mfcc: int = 13, frame_ms: int = 25) -> dict:
    """Pure-stdlib MFCC-ish proxy stats: log-mel energy ratio + spectral flatness.

    Full DCT pipeline requires numpy>=1.x (present here); we keep it light so
    this stays useful without the heavy audio ML stack installed.
    """
    pcm = _load_wav_pcm(wav_path)
    if pcm is None or len(pcm) < 8000:
        return {"mel_band_energy_ratio": None, "spectral_flatness_mean": None,
                "rms_dbfs": None, "usable_audio": False}
    # near-silence gate: a signal this quiet has no acoustics to judge on —
    # scoring it produces fabricated evidence (classic TTS/test-tones trap).
    rms = float(np.sqrt(np.mean(np.asarray(pcm, dtype=np.float64) ** 2)))
    dbfs = 20 * np.log10(rms + 1e-9)
    usable = rms >= 1e-3  # ≈ -60 dBFS floor
    if not usable:
        return {"mel_band_energy_ratio": None, "spectral_flatness_mean": None,
                "rms_dbfs": round(float(dbfs), 1), "usable_audio": False}
    sr = 16000
    hop = int(sr * frame_ms / 1000)
    win = 512
    frames = []
    for i in range(0, len(pcm) - win, hop):
        seg = pcm[i:i + win]
        frames.append(np.abs(np.fft.rfft(seg * np.hanning(win))) ** 2)
    if not frames:
        return {"mel_band_energy_ratio": None, "spectral_flatness_mean": None,
                "rms_dbfs": round(float(dbfs), 1), "usable_audio": True}
    # split rFFT bins into 3 mel-ish bands (low/mid/high)
    nbins = len(frames[0])
    b = nbins // 3
    low = np.mean([f[:b].sum() for f in frames])
    mid = np.mean([f[b:2 * b].sum() for f in frames])
    high = np.mean([f[2 * b:].sum() for f in frames])
    total = (low + mid + high) or 1.0
    ratio = float(high / (low + mid + 1e-9))          # real speech skews low/mid
    # spectral flatness per frame: geo-mean / arith-mean of the NORMALIZED
    # power spectrum. Bounded to (0,1]; 1.0 ~ white noise, ~0.0 ~ single-tone.
    # Computing on raw power (pre-normalization) overflows for strong signals.
    flats = []
    for f in frames:
        s = f / (f.sum() + 1e-12)
        gm = float(np.exp(np.mean(np.log(s + 1e-12))))
        am = float(s.mean())
        flats.append(gm / (am + 1e-12))
    sfm = min(1.0, float(np.mean(flats))) if flats else None
    return {"mel_band_energy_ratio": round(ratio, 4),
            "spectral_flatness_mean": (round(sfm, 4) if sfm is not None else None),
            "rms_dbfs": round(float(dbfs), 1), "usable_audio": True}


def _load_wav_pcm(p: Path) -> np.ndarray | None:
    """Read 16-bit mono WAV -> float array in [-1,1]; ffmpeg-produced WAVs qualify."""
    import struct
    raw = p.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    # find 'data' chunk
    i = 12
    while i + 8 <= len(raw):
        cid = raw[i:i + 4]
        sz = struct.unpack("<I", raw[i + 4:i + 8])[0]
        if cid == b"data":
            body = raw[i + 8:i + 8 + sz]
            samples = struct.unpack(f"<{len(body) // 2}h", body[:len(body) // 2 * 2])
            arr = np.asarray(samples, dtype=np.float32) / 32768.0
            return arr
        i += 8 + sz
    return None


def _crop_windows(outdir: Path, n: int = 3, win_s: float = 3.0) -> list[Path]:
    """Evenly spaced fixed-length crops of the normalized wav (multi-crop infer)."""
    src = outdir.parent / "aud.wav"
    import shutil as _sh, os
    if not _sh.which(os.environ.get("VISHWAS_FFMPEG_BIN", "ffmpeg")) or not src.exists():
        return []
    pi = probe(src)
    total = pi.duration_s
    step = total / max(1, n)
    made: list[Path] = []
    outdir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        dst = outdir / f"win{i}.wav"
        cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{i * step:.1f}", "-t", str(win_s),
               "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=90)
            if dst.exists() and dst.stat().st_size > 512:
                made.append(dst)
        except Exception:
            continue
    return made
