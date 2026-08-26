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
  3. HAVIC gated heavy model for learned cross-modal scoring when weights
     exist (task 1.4): routed through the model_adapters arch-aware seam
     (resolve -> adapter.load -> is_usable_model) exactly like deepfake_audio;
     preprocessing is a numpy-only port of the reference kaldi-fbank path
     (torchaudio is not installed in this tree) plus a 16-frame [0,1] RGB
     tensor. Any failure degrades to a 'failed' CheckResult, never an escape.
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
from ..model_adapters import resolve as _resolve_adapter, is_usable_model as _is_usable_model
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
    stage_cost = "heavy"   # 2.1: eligible for non-blocking budget + follow-up

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
        # Gated HAVIC learned model (refines the heuristic band). _havic_check
        # never raises: unavailable / ok / failed are all CheckResult-shaped.
        base.append(_havic_check(_load_havic(), art.path, wav, ctx.quarantine_root))
        return base


def _load_havic():
    """Registry-gated HAVIC loader (task 1.4 wiring).

    Routes through the model_adapters arch-aware seam — resolve() ->
    adapter.load() (builds the HAVIC_FT skeleton and applies the checkpoint
    with key-coverage verification) -> is_usable_model(). Returns a READY
    ArchModelWrapper whose ``.predict((audio_fbank, visual_frames))`` is the
    [0, 1] inconsistency posterior, or None when the env var is unset, the
    weights are unreadable, or the arch is unavailable. Never raises and
    never returns a bare state dict (the old raw torch.load behaviour).
    """
    p = os.environ.get("VISHWAS_HAVIC_WEIGHTS")
    if not p or not os.path.exists(p):
        return None
    adapter = _resolve_adapter("VISHWAS_HAVIC_WEIGHTS")
    if adapter is None:
        return None
    obj = adapter.load(p)
    return obj if _is_usable_model(obj) else None


def _havic_check(hav, video_path: Path, wav: Path | None, workdir: Path) -> CheckResult:
    """Score one artifact through a loaded HAVIC wrapper (never raises).

    hav None            -> 'unavailable' (weights not provisioned)
    wrapper .predict ok -> 'ok' + prob_inconsistent in [0, 1]
    ANY exception       -> 'failed' + {error_class}   (never escapes)
    """
    if hav is None:
        return CheckResult("havic_crossmodal_model", "heavy", "unavailable",
                           {"missing_dependency": "model-weights"},
                           "HAVIC weights not provisioned (VISHWAS_HAVIC_WEIGHTS); heuristic AV probe stands")
    try:
        x = _havic_preprocess(video_path, wav, workdir)
        out = hav.predict(x)
        val = out[0] if isinstance(out, (list, tuple)) else out
        p = float(val)
        return CheckResult("havic_crossmodal_model", "heavy", "ok",
                           {"prob_inconsistent": round(min(1.0, max(0.0, p)), 3)},
                           "HAVIC learned cross-modal consistency pass")
    except Exception as e:  # noqa: BLE001
        return CheckResult("havic_crossmodal_model", "heavy", "failed",
                           {"error_class": e.__class__.__name__}, "HAVIC inference error")


# --------------------------------------------------- HAVIC preprocessing ----
#: Reference dataset stats for fbank normalisation — the exact constants from
#: the upstream finetune config (scripts/finetune.sh) used verbatim by the
#: reference inference script: fbank = (fbank - (-6.9960)) / 3.1205.
_HAVIC_FBANK_MEAN = -6.9960
_HAVIC_FBANK_STD = 3.1205

#: Upstream HAVIC segments each clip into ~3.2 s windows before sampling 16
#: frames and computing the matched audio fbank (video_engine.py split_video /
#: split_videos_from_csv + src/dataloader._wav2fbank). We replicate a single
#: window from t=0 (upstream start_time=0).
_HAVIC_WINDOW_S = 3.2
#: FaceX-Zoo crop expansion factor (video_engine.py FaceX_Zoo_FaceCropper).
_HAVIC_FACE_SCALE = 1.3
#: YuNet onnx candidates for the first-face crop. This box ships no Haar
#: cascades, so we drive cv2's built-in dnn engine (already imported) with the
#: small ~230 KB YuNet model. Any candidate file is accepted; the env var lets
#: a deployment point at its own copy.
_HAVIC_FACE_MODEL_CANDIDATES = (
    os.environ.get("VISHWAS_HAVIC_FACE_MODEL", ""),
    "/opt/verisafe/models/havic/face_detection_yunet_2023mar.onnx",
    str(Path.home() / "avtools" / "face_detection_yunet_2023mar.onnx"),
)
#: module-level cache: None = untried, False = unavailable, else detector obj.
_HAVIC_FACE_DETECTOR_CACHE = None


def _load_face_detector():
    """Return a YuNet detector (or None) for the HAVIC first-face crop.

    Loads lazily and caches. Degrades to None (whole-frame visual path) when
    no model file resolves, so the cross-modal capability never hard-fails on
    boxes without a face model.
    """
    global _HAVIC_FACE_DETECTOR_CACHE
    if _HAVIC_FACE_DETECTOR_CACHE is not None:
        return _HAVIC_FACE_DETECTOR_CACHE or None
    import cv2  # type: ignore
    for cand in _HAVIC_FACE_MODEL_CANDIDATES:
        if cand and os.path.exists(cand):
            try:
                det = cv2.FaceDetectorYN_create(cand, "", (320, 320), 0.6, 0.3, 5000)
                _HAVIC_FACE_DETECTOR_CACHE = det
                return det
            except Exception:
                continue
    _HAVIC_FACE_DETECTOR_CACHE = False
    return None


def _kaldi_fbank(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    target_frames: int = 1024,
) -> np.ndarray:
    """Kaldi-style log-fbank — numpy port of the exact torchaudio call the
    HAVIC reference pipeline makes::

        torchaudio.compliance.kaldi.fbank(
            wav, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)

    torchaudio is absent in this tree, so the frame pipeline is reproduced
    with torchaudio's kaldi defaults: 25 ms frames / 10 ms shift, snip_edges,
    per-frame DC removal, 0.97 preemphasis (replicate-padded, so sample 0
    scales by 1-0.97), symmetric Hann window (kaldi 'hanning' ==
    torch.hann_window(periodic=False) == np.hanning), 512-point zero-padded
    FFT, power spectrum, kaldi mel filterbank (1127*ln(1+f/700) scale, low
    20 Hz, high -> Nyquist; htk_compat only moves the energy column, which
    use_energy=False omits anyway) over the first n_fft/2 bins, natural log
    clamped at float32 eps. The time axis is then linearly interpolated to
    *target_frames* rows (align_corners=False, matching the reference
    F.interpolate call). Returns float32 (target_frames, n_mels).
    """
    eps = float(np.finfo(np.float32).eps)
    wf = np.asarray(waveform, dtype=np.float32).ravel()
    win = int(round(sr * 0.025))           # 400 @16 kHz
    hop = max(1, int(round(sr * 0.010)))   # 160 @16 kHz
    if wf.size < win:
        wf = np.pad(wf, (0, win - wf.size))
    n_frames = 1 + (wf.size - win) // hop
    frames = np.lib.stride_tricks.sliding_window_view(wf, win)[::hop][:n_frames].copy()
    frames -= frames.mean(axis=1, keepdims=True)          # remove_dc_offset
    prev = np.pad(frames, ((0, 0), (1, 0)), mode="edge")  # replicate pad
    frames -= 0.97 * prev[:, :-1]                         # preemphasis
    frames *= (np.hanning(win) ** 2).astype(np.float32)  # kaldi 'hanning' == torch.hann_window(periodic=False), BUT torchaudio's _get_window ALSO raises it to power 2.0 (window.pow_(2.0)) so the net window is hanning^2 — mirror that exactly.
    n_fft = 512                                           # round_to_power_of_two
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2  # use_power
    banks = _kaldi_mel_banks(n_mels, n_fft, sr)           # (n_mels, n_fft // 2)
    fbank = spec[:, :n_fft // 2] @ banks.T
    fbank = np.log(np.maximum(fbank, eps))                # use_log_fbank
    return _linear_interp_rows(fbank, target_frames).astype(np.float32)


def _kaldi_mel_banks(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Kaldi/torchaudio mel filterbank (vtln_warp=1.0), (n_mels, n_fft//2)."""
    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    low, high = 20.0, 0.5 * sr        # kaldi defaults; high_freq<=0 -> nyquist
    bin_w = sr / n_fft
    mel_low, mel_high = float(mel(low)), float(mel(high))
    delta = (mel_high - mel_low) / (n_mels + 1)
    k = np.arange(n_mels, dtype=np.float64)
    left_mel = mel_low + k * delta
    center_mel = mel_low + (k + 1.0) * delta
    right_mel = mel_low + (k + 2.0) * delta
    mel_bins = mel(bin_w * np.arange(n_fft // 2, dtype=np.float64))
    up = (mel_bins[None, :] - left_mel[:, None]) / (center_mel - left_mel)[:, None]
    down = (right_mel[:, None] - mel_bins[None, :]) / (right_mel - center_mel)[:, None]
    # left_mel < center_mel < right_mel -> triangle = min of slopes, clamp >= 0
    return np.maximum(0.0, np.minimum(up, down)).astype(np.float32)


def _linear_interp_rows(a: np.ndarray, n_out: int) -> np.ndarray:
    """align_corners=False linear interpolation of axis 0 to *n_out* rows
    (equivalent of F.interpolate(size=n_out, mode='linear', align_corners=False))."""
    n_in = a.shape[0]
    if n_in == n_out:
        return a
    pos = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    pos = np.clip(pos, 0.0, max(n_in - 1, 0))
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (pos - lo)[:, None]
    return a[lo] * (1.0 - w) + a[hi] * w


def _frame_chw(fp: Path, size: int = 224) -> np.ndarray | None:
    """Image path -> (3, size, size) float32 RGB in [0,1] (bilinear resize)."""
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if im is None:
            return None
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
        return np.transpose(im.astype(np.float32) / 255.0, (2, 0, 1))
    except Exception:
        return None


def _havic_visual(frames: list[Path], n: int = 16, size: int = 224) -> np.ndarray | None:
    """WHOLE-FRAME fallback visual path (image files -> (3, n, s, s) RGB [0,1]).

    Used only when the preferred face-crop video window is unavailable (or a
    clip can't be opened). Mirrors T.Resize+ToTensor on RGB; no ImageNet
    mean/std. If fewer than *n* frames the last decodable one repeats.
    """
    tensors: list[np.ndarray] = []
    last: np.ndarray | None = None
    for i in range(n):
        t = _frame_chw(frames[min(i, len(frames) - 1)], size) if frames else None
        if t is None:
            if last is None:
                return None
            t = last
        else:
            last = t
        tensors.append(t)
    return np.transpose(np.stack(tensors), (1, 0, 2, 3)).astype(np.float32)


def _frame_face_chw(frame_bgr: np.ndarray, detector, size: int = 224) -> np.ndarray:
    """BGR frame -> (3, size, size) float32 RGB in [0, 1].

    Crops the first detectable face (YuNet) expanded by the reference
    scale=1.3 around its center, matching the upstream FaceX-Zoo cropper
    (video_engine.py:279-291). Falls back to the whole frame when no face is
    found (e.g. synthetic controls). Bilinear resize + /255 — no ImageNet
    mean/std, matching the upstream dataloader preprocess.
    """
    import cv2  # type: ignore
    h, w = frame_bgr.shape[:2]
    box = None
    if detector is not None:
        try:
            detector.setInputSize((w, h))
            ok, faces = detector.detect(frame_bgr)
            if ok and faces is not None and len(faces):
                x, y, bw, bh = (float(v) for v in faces[0][:4])
                cx, cy = x + bw / 2.0, y + bh / 2.0
                nw, nh = bw * _HAVIC_FACE_SCALE, bh * _HAVIC_FACE_SCALE
                x0 = max(int(cx - nw / 2), 0)
                x1 = min(int(cx + nw / 2), w)
                y0 = max(int(cy - nh / 2), 0)
                y1 = min(int(cy + nh / 2), h)
                if x1 > x0 and y1 > y0:
                    box = (x0, y0, x1, y1)
        except Exception:
            box = None
    crop = frame_bgr if box is None else frame_bgr[box[1]:box[3], box[0]:box[2]]
    im = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.transpose(im.astype(np.float32) / 255.0, (2, 0, 1))


def _havic_visual_window(video_path: Path, window_s: float, n: int = 16,
                         size: int = 224) -> np.ndarray | None:
    """Full-res frame path -> (3, n, size, size) float32 RGB in [0, 1].

    Decodes *n* uniformly-timed frames (``np.linspace(0, nwin-1, n)`` — the
    exact upstream sampling, video_engine.py:184) across a *window_s* second
    window, face-cropping each (whole-frame fallback). Full-res decode, no
    JPEG/256px loss. Returns None only when no frame decodes.
    """
    import cv2  # type: ignore
    detector = _load_face_detector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or fps <= 0:
            fps = 25.0
        nwin = max(1, int(round(window_s * fps)))
        if total > 0:
            nwin = min(nwin, int(total))
        idxs = np.linspace(0, max(nwin - 1, 0), n, dtype=int)
        tensors: list[np.ndarray] = []
        last: np.ndarray | None = None
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            t = _frame_face_chw(fr, detector, size) if (ok and fr is not None) else None
            if t is None:
                if last is None:
                    return None
                t = last
            else:
                last = t
            tensors.append(t)
        return np.transpose(np.stack(tensors), (1, 0, 2, 3)).astype(np.float32)
    finally:
        cap.release()


def _havic_preprocess(video_path: Path, wav: Path | None, workdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Media file -> (audio_fbank (1024, 128), visual (3, 16, 224, 224)).

    Faithful reproduction of the upstream HAVIC input construction: a single
    3.2 s analysis window from t=0 (reference segments clips to 3.2 s),
    * 16 uniformly-timed full-res FIRST-FACE crops at scale=1.3 resized to
    224 (whole-frame fallback when no face) — the prime saturation fix;
    * the kaldi log-fbank (-mean) of the MATCHED 3.2 s audio window,
    interpolated to 1024 frames and normalised by the reference
    (-6.9960, 3.1205) constants.
    Raises ValueError on undecodable inputs — callers convert that to a
    'failed' CheckResult.
    """
    dur = probe(video_path).duration_s
    window_s = min(dur, _HAVIC_WINDOW_S) if dur and dur > 0 else _HAVIC_WINDOW_S
    # Preferred: full-res face-crop window over the 3.2 s window. If the video
    # cannot be opened for the face path (or decode fails), degrade to the
    # whole-frame extract_frames fallback so short/scoped callers still score.
    visual = _havic_visual_window(video_path, window_s)
    if visual is None:
        frames = extract_frames(video_path, workdir / "xm_havic_frames", n=16, jpeg_q=9)
        if not frames:
            raise ValueError("no frames extracted for HAVIC visual stream")
        visual = _havic_visual(frames)
        if visual is None:
            raise ValueError("HAVIC visual frames unreadable")
    pcm = _load_pcm(wav) if wav is not None else None
    if pcm is None or pcm.size == 0:
        raise ValueError("no decodable PCM for HAVIC audio stream")
    win_n = int(round(window_s * 16000))
    if pcm.size > win_n:
        pcm = pcm[:win_n]          # match the 3.2 s visual window
    fbank = _kaldi_fbank(pcm - pcm.mean())  # reference: segment - segment.mean()
    fbank = (fbank - _HAVIC_FBANK_MEAN) / _HAVIC_FBANK_STD
    return fbank.astype(np.float32), visual


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
