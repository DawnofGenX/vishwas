"""Cross-modal forensic consistency layer (spec: HAVIC-class check).

For media containing BOTH video and audio: the two streams are analyzed
independently by their specialist detectors; this module then tests whether
the modalities are *mutually consistent* — a deepfake that swaps the voice
but keeps the original lips (or vice-versa) shows up here as a modality-level
mismatch that no single-modality detector would catch.

Implementation strategy (CPU-only, deterministic):
  1. Lip-to-speech plausibility proxy: mouth-motion amplitude envelope from
     sampled frames vs speech-energy envelope of the audio track. Their
     cross-correlation over time is a cheap AV-sync / co-manipulation probe:
       - strongly positive correlation      -> consistent (both genuine OR both faked in lockstep)
       - near-zero/negative correlation     -> mismatch: one modality was swapped/edited
  2. Temporal alignment probe: dominant cadence peaks should line up across
     the two envelopes within +-100 ms after resampling to a common grid.
  3. HAVIC gated heavy model for learned cross-modal scoring when weights exist.
The output feeds fusion with weight class 'heavy'; unavailable status is the
default on boxes without the weights provisioned.
"""
from __future__ import annotations

import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..events import Artifact, JobContext
from ..media_utils import probe, extract_frames, extract_audio_wav
from .base import CheckResult


def _mouth_envelope(frames: list[Path], tgrid: int = 30) -> list[float]:
    """Per-frame mouth-region motion proxy: luma variance of lower-third band
    across consecutive samples. Deterministic; no face-detection dependency."""
    vals: list[float] = []
    prev_band: np.ndarray | None = None
    for fp in frames:
        img = _gray(fp)
        if img is None:
            continue
        h = len(img)
        band = img[int(h * 0.6):int(h * 0.95), :]
        if prev_band is not None and band.shape == prev_band.shape:
            d = float(np.abs(band.astype(np.int16) - prev_band.astype(np.int16)).astype(np.float32).mean())
        else:
            d = float(band.mean()) / 255.0
        prev_band = band
        vals.append(d)
    return _resample_to_grid(vals, tgrid)


def _speech_energy_envelope(wav: Path, tgrid: int = 30) -> list[float]:
    pcm = _load_pcm(wav)
    if pcm is None or len(pcm) < 16000:
        return []
    sr = 16000
    hop = sr // tgrid                      # one window per tgrid tick
    win = sr // 4                          # 250 ms windows
    env: list[float] = []
    for i in range(0, len(pcm) - win, hop):
        seg = pcm[i:i + win]
        rms = float(np.sqrt(np.mean(np.square(seg))))
        env.append(rms)
    return _resample_to_grid(env, tgrid)


def _resample_to_grid(x: list[float], n: int) -> list[float]:
    if not x:
        return []
    out: list[float] = []
    for i in range(n):
        idx = int(i * (len(x) - 1) / max(1, n - 1))
        out.append(x[idx])
    return out


def _norm(x: list[float]) -> list[float]:
    m, s = statistics.mean(x), statistics.pstdev(x) or 1e-9
    return [(v - m) / s for v in x]


def _lagged_crosscorr(a: list[float], b: list[float], max_lag_ms: int = 200, tgrid: int = 30) -> tuple[float, int]:
    """Best lag (in ms) and its Pearson r between two normalized envelopes."""
    if len(a) < 4 or len(b) < 4:
        return 0.0, 0
    la, lb = _norm(a), _norm(b)
    best_r, best_lag = -2.0, 0
    max_lags = max(1, max_lag_ms // (1000 // tgrid))
    for lag in range(-max_lags, max_lags + 1):
        pa = la[max(0, lag):min(len(la), len(lb) + lag)]
        pb = lb[max(0, -lag):min(len(lb), len(la) - lag)]
        n = min(len(pa), len(pb))
        if n < 3:
            continue
        try:
            r = float(np.corrcoef(pa[:n], pb[:n])[0, 1])
        except Exception:
            continue
        if r > best_r:
            best_r, best_lag = r, lag
    return best_r, int(abs(best_lag) * (1000 // tgrid))


class CrossModalCapability:
    requires: tuple[str, ...] = ()

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        pi = probe(art.path)
        if not (pi.has_video and pi.has_audio):
            out.append(CheckResult("cross_modal_av", "heavy", "skipped", {},
                                   "no paired audio+video streams; cross-modal forensics N/A"))
            return out
        frames_dir = ctx.quarantine_root / "xm_frames"
        frames = extract_frames(art.path, frames_dir, n=8)
        wav = extract_audio_wav(art.path, ctx.quarantine_root / "xm.wav")
        if not frames or wav is None:
            out.append(CheckResult("cross_modal_av", "heavy", "degraded", {},
                                   "could not derive paired envelopes; limited AV evidence"))
            return out
        mouth = _mouth_envelope(frames)
        speech = _speech_energy_envelope(wav)
        r, lag_ms = _lagged_crosscorr(mouth, speech)
        # Interpretation bands (empirical, documented):
        #   r >= 0.35 & lag <= 100  : synced
        #   |r| small (<0.15)       : decorrelated -> possible modality swap
        #   strong negative         : anti-correlated -> likely independent generation
        verdict_class = ("synced" if r >= 0.35 and lag_ms <= 100
                         else "decorrelated" if abs(r) < 0.15
                         else "anti_correlated" if r < -0.25
                         else "weakly_synced")
        risk_add = {"synced": 0.0, "weakly_synced": 0.1, "decorrelated": 0.45,
                    "anti_correlated": 0.5}.get(verdict_class, 0.1)
        notes_map = {
            "synced": "lip motion and speech energy move together in time (coherent)",
            "weakly_synced": "moderate AV coherence; some regions diverge",
            "decorrelated": "LIP AND AUDIO MOVEMENTS DO NOT LINE UP — classic sign of one track being replaced (voice-swap or lip-sync patch)",
            "anti_correlated": "envelopes move oppositely; suggests independently generated modalities",
        }
        base = [CheckResult("cross_modal_av", "heavy", "ok",
                            {"av_correlation": round(float(r), 3),
                             "best_lag_ms": lag_ms,
                             "alignment_class": verdict_class,
                             "av_risk_addition": risk_add},
                            notes_map[verdict_class])]
        # Gated HAVIC learned model (refines the heuristic band)
        hav = _load_havic()
        if hav is not None:
            try:
                p = float(hav.predict(art.path)[0])
                base.append(CheckResult("havic_crossmodal_model", "heavy", "ok",
                                        {"prob_inconsistent": round(min(1.0, max(0.0, p)), 3)},
                                        "HAVIC learned cross-modal consistency pass"))
            except Exception as e:  # noqa: BLE001
                base.append(CheckResult("havic_crossmodal_model", "heavy", "failed",
                                        {"error_class": e.__class__.__name__}, "HAVIC inference error"))
        else:
            base.append(CheckResult("havic_crossmodal_model", "heavy", "unavailable",
                                    {"missing_dependency": "model-weights"},
                                    "HAVIC weights not provisioned (VERISAFE_HAVIC_WEIGHTS); heuristic AV probe stands"))
        return base


def _load_havic():
    p = os.environ.get("VERISAFE_HAVIC_WEIGHTS")
    if not p or not os.path.exists(p):
        return None
    try:
        import torch  # type: ignore
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _gray(fp: Path, target: int = 224) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        return cv2.resize(im, (target, target)).astype(np.uint8)
    except Exception:
        return None


def _load_pcm(p: Path) -> np.ndarray | None:
    import struct
    raw = p.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    i = 12
    while i + 8 <= len(raw):
        cid = raw[i:i + 4]
        sz = struct.unpack("<I", raw[i + 4:i + 8])[0]
        if cid == b"data":
            body = raw[i + 8:i + 8 + sz]
            samples = struct.unpack(f"<{len(body) // 2}h", body[:len(body) // 2 * 2])
            return np.asarray(samples, dtype=np.float32) / 32768.0
        i += 8 + sz
    return None
