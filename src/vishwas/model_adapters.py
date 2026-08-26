"""Model-weights inference adapter registry.

Each weight family (EFFORT, DeMamba, Fake-Mamba, AASIST, SSL-audio, HAVIC,
IMAGE_FACE) maps env-var -> Adapter with:
  preprocess(raw) -> model input
  extract_prob(model_output) -> float | None
  load(path) -> object | None   (never raises; returns None on any failure)

The capabilities call ``resolve(env_name)`` to get the adapter, then
``run_check(adapter, weight_path, raw_input)`` for the (status, signals, notes)
tuple.  Missing torch or an unreadable path yields status 'unavailable' and the
capability emits its existing CheckResult verbatim — no regression.

Arch-aware loading (Phase 1 / B0): gated families with a registered arch spec
(aasist/effort/havic) load via :func:`_arch_aware_load`, which builds the
network skeleton, verifies key-set coverage of the checkpoint's state dict,
and returns a READY callable wrapper only when both succeed.  A loaded-but-not
applicable checkpoint yields None plus ``last_reason = 'weight file loaded
but architecture unavailable'`` on the adapter — never a half-loaded model.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .device import resolve_device
from .model_archs import get_arch
from .model_archs.base import ArchNotImplementedError, ArchSpec


# ---------------------------------------------------------------------------
# Mel-spectrogram helper (numpy-only; no librosa/scipy assumed)
# ---------------------------------------------------------------------------

def compute_mel(
    waveform: np.ndarray,
    *,
    sr: int = 16000,
    n_mels: int = 128,
    max_frames: int = 2000,
    frame_len_ms: int = 25,
    hop_ms: int = 10,
) -> np.ndarray:
    """1-D waveform -> (n_frames, n_mels) log-mel spectrogram.

    - Triangular filterbank built from mel-scale boundaries.
    - Padding/cropping to exactly *max_frames* rows (zero-pad tail or crop head).
    - Returns shape (rows<=max_frames, cols==n_mels), finite, non-negative.
    """
    if waveform.ndim != 1:
        waveform = waveform.squeeze()
    # Ensure float32, normalise to [-1, 1] scale
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.max() > 1.0:
        wf = wf / (wf.max() + 1e-12)

    win_size = int(sr * frame_len_ms / 1000)
    hop = max(1, int(sr * hop_ms / 1000))
    if len(wf) < win_size:
        wf = np.pad(wf, (0, win_size - len(wf)))

    n_frames_avail = (len(wf) - win_size) // hop + 1
    n_frames = min(n_frames_avail, max_frames)
    if n_frames <= 0:
        return np.zeros((1, n_mels), dtype=np.float32)

    # Short-time FFT
    window = np.hanning(win_size).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(wf, win_size)[::hop][:n_frames]
    stft = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    n_bins = stft.shape[1]  # win_size//2 + 1

    # Triangular mel filterbank
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = _hz_to_mel(0)
    mel_max = _hz_to_mel(sr / 2)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor(hz_pts * n_bins / (sr / 2)).astype(int).clip(0, n_bins)

    filters = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = bin_pts[i], bin_pts[i + 1], bin_pts[i + 2]
        if mid <= lo:
            continue
        for j in range(lo, mid):
            if mid > lo:
                filters[i, j] = (j - lo) / (mid - lo)
        for j in range(mid, hi):
            if hi > mid:
                filters[i, j] = (hi - j) / (hi - mid)

    mel_spec = stft @ filters.T
    mel_spec = np.log(mel_spec + 1e-9)
    # Pad or crop to max_frames rows
    if mel_spec.shape[0] < max_frames:
        pad = np.zeros((max_frames - mel_spec.shape[0], n_mels), dtype=np.float32)
        mel_spec = np.vstack([mel_spec, pad])
    else:
        mel_spec = mel_spec[:max_frames]
    return mel_spec.astype(np.float32)


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

#: Single-slot holder so loaders (e.g. _arch_aware_load) can hand a failure
#: reason back to Adapter.load() without changing any loader signature.
_pending_load_reason: list[str | None] = [None]

#: Sentinel distinguishing "arch argument omitted" from "arch=None" in
#: _arch_aware_load (None means 'no arch for this family').
_ARCH_UNSET = object()


@dataclass
class Adapter:
    """One entry in the ADAPTERS registry."""
    env_name: str
    family: str  # 'image' | 'audio' | 'video' | 'face'
    preprocess: Callable[[Any], Any]
    extract_prob: Callable[[Any], float | None] = field(default=None)  # type: ignore
    _load: Callable[[str], Any] = field(default=None, repr=False)     # type: ignore
    last_reason: str | None = field(default=None, repr=False)
    """Records WHY the most recent load() returned None (e.g. the arch seam's
    'weight file loaded but architecture unavailable'); None until set."""

    def __post_init__(self):
        if self._load is None:
            self._load = _default_load

    # -- public API ----------------------------------------------------------
    def load(self, path: str) -> Any:
        """Load weights; return None on ANY failure. Never raises.

        When the loader records a failure reason (see ``_arch_aware_load``),
        it is adopted into ``self.last_reason`` so callers can surface WHY the
        weights were unusable (e.g. 'weight file loaded but architecture
        unavailable') instead of a generic 'not provisioned' note.
        """
        _pending_load_reason[0] = None
        try:
            obj = self._load(path)
        except Exception:
            obj = None
        if obj is None and _pending_load_reason[0]:
            self.last_reason = _pending_load_reason[0]
        return obj

    def run(self, weight_path: str, raw_input: Any) -> tuple[str, dict, str]:
        """Full check pipeline. Returns (status, signals, notes)."""
        model = self.load(weight_path)
        if model is None:
            return "unavailable", {"missing_dependency": "model-weights"}, \
                   f"{self.env_name} weights not provisioned or unreadable"
        try:
            processed = self.preprocess(raw_input)
            out = _call_model(model, processed)
            prob = self.extract_prob(out) if self.extract_prob else _auto_extract(out)
            if prob is None:
                return "failed", {}, "adapter could not extract probability from model output"
            return "ok", {"prob_deepfake": round(prob, 3)}, "model inference succeeded"
        except Exception as e:
            return "degraded", {"error_class": type(e).__name__}, \
                   f"inference error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_load(path: str) -> Any:
    """torch.load with map_location=resolve_device(); returns None on any failure."""
    if not path or not os.path.exists(path):
        return None
    try:
        import torch  # type: ignore
        return torch.load(path, map_location=resolve_device(), weights_only=False)
    except ImportError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Arch-aware loading seam (Phase 1 / B0)
# ---------------------------------------------------------------------------

#: env var -> arch family registered in vishwas.model_archs
_ARCH_FAMILIES = {
    # 2026-08-26 SWAP to proven arch: the VISHWAS_AASIST slot now resolves to
    # 'aasist3' (Spectra-AASIST3, Apache-2.0) instead of the old 'aasist'
    # (HABLA_WavLM_AASIST) checkpoint, which was rejected/inverted at serving
    # (bonafide prob_deepfake 1.0). Track C independently re-scored Spectra-
    # AASIST3 at AUC 0.9967 / EER 0.70% on the official ASVspoof2019-LA eval
    # (bonafide p_spoof median 0.000, spoof 0.991). Old 'aasist' arch module is
    # retained in the registry as rollback; see model_archs/aasist3.py header.
    "VISHWAS_AASIST_WEIGHTS": "aasist3",
    "VISHWAS_EFFORT_WEIGHTS": "effort",
    "VISHWAS_HAVIC_WEIGHTS": "havic",
    # RawBMamba fills the 'fakemamba' family (see model_archs/fakemamba.py).
    "VISHWAS_FAKEMAMBA_WEIGHTS": "fakemamba",
    # XLSR-Mamba-LA (MIT) — see model_archs/xlsrmamba.py.
    "VISHWAS_XLSRMAMBA_WEIGHTS": "xlsrmamba",
    # SPAI (CVPR'25 spectral AI-image detector, Apache-2.0) fills the image
    # face-check slot; see model_archs/spai.py + _spai/PROVENANCE.md.
    "VISHWAS_IMAGE_FACE_WEIGHTS": "spai",
}

ARCH_UNAVAILABLE_REASON = "weight file loaded but architecture unavailable"


def _extract_state_dict(raw: Any) -> Any:
    """Pull the weight tensor-dict out of a checkpoint payload.

    Handles flat state-dicts/OrderedDicts, and top-level training wrappers
    (e.g. AASIST's ``{'model_state': ..., 'optimizer_state': ...}``).
    Returns None when no tensor-dict payload is recognisable.
    """
    if isinstance(raw, dict):
        for k in ("model_state", "state_dict", "model", "net"):
            v = raw.get(k)
            if isinstance(v, dict):
                return v
        return raw
    return None


class ArchModelWrapper:
    """READY callable returned by :func:`_arch_aware_load` once the skeleton
    was built AND the checkpoint applied cleanly.

    Exposes ``.predict(x) -> float`` (and ``.score``) so the existing
    ``_call_model`` duck-typing and ``is_usable_model`` keep working; the
    float is the ArchSpec's calibrated [0,1] posterior.
    """

    def __init__(self, model: Any, spec: ArchSpec):
        self.model = model
        self.spec = spec

    def predict(self, x: Any) -> float:
        return self.spec.score(self.model, x)

    def score(self, x: Any) -> float:
        return self.spec.score(self.model, x)


def _arch_aware_load(path: str, family: str | None = None, *,
                     env_name: str | None = None,
                     raw_load: Callable[[str], Any] | None = None,
                     arch: Any = _ARCH_UNSET) -> Any:
    """Load-order seam for gated model families (plan Phase 1 / B0).

    (a) env var unset/missing -> None (behavior unchanged, no reason recorded);
    (b) env set, arch module importable AND ``apply_state`` True -> READY
        :class:`ArchModelWrapper` (callable, .predict/.score);
    (c) otherwise -> None + ``ARCH_UNAVAILABLE_REASON`` handed back through
        ``_pending_load_reason`` so ``Adapter.load()`` records it on
        ``adapter.last_reason`` (the notes channel capabilities surface).

    *raw_load* overrides the checkpoint reader (tests inject fakes); *arch*
    overrides the registry lookup (pass an ArchSpec instance, or None to force
    'no arch'; omit to consult the registry). Existing heuristic fallbacks are
    untouched either way.
    """
    env = env_name if env_name is not None else (family or "").upper()
    if env:
        p_env = os.environ.get(env)
        if not p_env or not os.path.exists(p_env):
            return None  # (a) env gate — unchanged behaviour
    if not path or not os.path.exists(path):
        return None
    loader = raw_load if raw_load is not None else _default_load
    try:
        raw = loader(path)
    except Exception:
        return None
    if raw is None:
        return None
    # Already a usable inference object (full-model checkpoint) — legacy path.
    if is_usable_model(raw):
        return raw
    sd = _extract_state_dict(raw)
    if sd is None:
        return None
    if arch is _ARCH_UNSET:
        fam = family or _ARCH_FAMILIES.get(env, "")
        spec = get_arch(fam) if fam else None
    else:
        spec = arch
    if spec is None:
        _pending_load_reason[0] = ARCH_UNAVAILABLE_REASON
        return None  # (c) no arch for this family
    try:
        model = spec.build()
    except ArchNotImplementedError:
        _pending_load_reason[0] = ARCH_UNAVAILABLE_REASON
        return None  # (c) stub: arch module present, build() not yet vendored
    except Exception:
        _pending_load_reason[0] = ARCH_UNAVAILABLE_REASON
        return None
    try:
        ok = bool(spec.apply_state(model, sd))
    except Exception:
        ok = False
    if not ok:
        _pending_load_reason[0] = ARCH_UNAVAILABLE_REASON
        return None  # (c) shape/key mismatch — never half-load
    return ArchModelWrapper(model, spec)  # (b) READY


def _call_model(model: Any, processed: Any) -> Any:
    """Call .predict, .forward, or __call__ — duck-typed, never raises here."""
    if hasattr(model, "predict"):
        return model.predict(processed)
    if hasattr(model, "forward"):
        import torch
        with torch.no_grad():
            return model.forward(torch.as_tensor(processed) if not isinstance(processed, (list, str, bytes)) else processed)
    if callable(model):
        return model(processed)
    raise TypeError(f"model object of type {type(model).__name__} has no callable interface")


def is_usable_model(obj: Any) -> bool:
    """True iff obj can actually run inference (predict/forward/__call__).

    Bare state-dict / tensor-container checkpoints (torch.save(model.state_dict()))
    are NOT usable without the architecture class — treat them as unusable so
    callers emit their normal 'unavailable' result rather than failing mid-scan.
    """
    if obj is None:
        return False
    return hasattr(obj, "predict") or hasattr(obj, "forward") or callable(obj)


def _sigmoid(x: float) -> float:
    if x > 60:
        return 1.0
    if x < -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _to_float(val: Any) -> float | None:
    """First element of tensor/list/array -> float; clamp/sigmoid as needed."""
    try:
        if hasattr(val, "item"):  # torch.Tensor scalar
            val = val.item()
        if hasattr(val, "squeeze") and callable(getattr(val, "squeeze")):
            arr = val.squeeze()
            if getattr(arr, "ndim", 0) == 0:
                val = float(arr)
            else:
                val = np.asarray(arr).flatten()[0]
        if isinstance(val, (list, tuple)):
            val = val[0] if val else 0.0
        f = float(val)
        if f > 1.0 or f < 0.0:
            f = _sigmoid(f)  # out-of-probability-range outputs are treated as logits
        return max(0.0, min(1.0, f))
    except Exception:
        return None


def _auto_extract(output: Any) -> float | None:
    return _to_float(output)


# ---------------------------------------------------------------------------
# Preprocess functions per family
# ---------------------------------------------------------------------------

def _img_resize_chw(frame_or_img: Any, size: int = 224) -> Any:
    """BGR HWC uint8 -> CHW float32 [0,1]. Accepts np array or Path-like."""
    arr = _ensure_ndarray(frame_or_img)
    if arr is None:
        return np.zeros((3, size, size), dtype=np.float32)
    # Center-crop square then resize
    h, w = arr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = arr[y0:y0 + s, x0:x0 + s]
    try:
        import cv2  # type: ignore
        resized = cv2.resize(crop, (size, size)).astype(np.float32) / 255.0
    except Exception:
        # numpy bilinear fallback
        resized = _np_resize(crop, size).astype(np.float32) / 255.0
    return np.transpose(resized, (2, 0, 1))


def _np_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Naive nearest-neighbour resize (fallback when cv2 absent)."""
    h, w = img.shape[:2]
    idx = (np.arange(size) * h // size).astype(int)
    idx2 = (np.arange(size) * w // size).astype(int)
    return img[idx[:, None], idx2[None, :], :]


def _ensure_ndarray(obj: Any) -> np.ndarray | None:
    """Accepts np.ndarray, Path, str path, or raw bytes; returns HWC uint8 array."""
    if isinstance(obj, np.ndarray):
        return obj if obj.ndim >= 2 else None
    # Path or str path to image file
    path_str = None
    if hasattr(obj, "__fspath__"):
        path_str = os.fspath(obj)
    elif isinstance(obj, str):
        path_str = obj
    if path_str and os.path.isfile(path_str):
        try:
            import cv2  # type: ignore
            im = cv2.imread(path_str, cv2.IMREAD_COLOR)
            return im
        except Exception:
            pass
    return None


def _wavform_from_path(path: Any) -> np.ndarray:
    """Read WAV -> 1-D float32 waveform [-1,1]. Falls back to zeros."""
    try:
        import struct
        raw = open(os.fspath(path) if hasattr(path, "__fspath__") else path, "rb").read()
        if raw[:4] != b"RIFF":
            return np.zeros(16000)
        i = 12
        while i + 8 <= len(raw):
            cid = raw[i:i + 4]
            sz = struct.unpack("<I", raw[i + 4:i + 8])[0]
            if cid == b"data":
                body = raw[i + 8:i + 8 + sz]
                samples = struct.unpack(f"<{len(body) // 2}h", body[:len(body) // 2 * 2])
                return np.asarray(samples, dtype=np.float32) / 32768.0
            i += 8 + sz
    except Exception:
        pass
    return np.zeros(16000, dtype=np.float32)


def _seq_tensor(frame_list: list[Any], size: int = 224) -> np.ndarray:
    """List of frames -> (N, 3, size, size) float32 tensor."""
    tensors = []
    for fr in frame_list[:16]:
        t = _img_resize_chw(fr, size)
        tensors.append(t)
    if not tensors:
        return np.zeros((1, 3, size, size), dtype=np.float32)
    return np.stack(tensors).astype(np.float32)


def _waveform_preprocess(raw: Any) -> np.ndarray:
    """1-D waveform float32 (or wav path) -> model-ready 1-D array."""
    if isinstance(raw, np.ndarray) and raw.ndim == 1:
        return raw.astype(np.float32)
    wf = _wavform_from_path(raw)
    return wf


def _mel_aasist_preprocess(raw: Any) -> np.ndarray:
    """WAV path or waveform -> 2-D (<=2000, 128) log-mel matrix."""
    wf = _waveform_preprocess(raw)
    return compute_mel(wf)


def _mel_ssl_preprocess(raw: Any) -> np.ndarray:
    """Same mel pipeline, different head expectation (SSL complement)."""
    wf = _waveform_preprocess(raw)
    return compute_mel(wf)


def _video_frame_preprocess(raw: Any) -> np.ndarray:
    """Frame list or video path -> (N, 3, 224, 224) sequence tensor."""
    if isinstance(raw, (list, tuple)):
        return _seq_tensor(list(raw))
    # single path — wrap in singleton list
    return _seq_tensor([raw])


def _spai_preprocess(raw):
    """Pass-through for the SPAI heavy gate: spai.ArchSpec.score() performs its
    own decode + [0,1] + pad-to-224 preprocessing faithfully to upstream
    `spai infer`, so the adapter does no reshaping here."""
    return raw


def _face_crops_preprocess(raw: Any) -> np.ndarray:
    """Face crop(s) -> (N, 3, 112, 112) embedding-style input."""
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3:
            return _img_resize_chw(raw, 112)[None]
        if raw.ndim == 4:
            batch = [_img_resize_chw(f, 112) for f in raw]
            return np.stack(batch).astype(np.float32)
    arr = _ensure_ndarray(raw)
    if arr is None:
        return np.zeros((1, 3, 112, 112), dtype=np.float32)
    return _img_resize_chw(arr, 112)[None]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, Adapter] = {
    "VISHWAS_EFFORT_WEIGHTS": Adapter(
        env_name="VISHWAS_EFFORT_WEIGHTS",
        family="image",
        preprocess=_img_resize_chw,
        extract_prob=_auto_extract,
        # Phase 1 Task 1.3: route through the arch-aware seam so a provisioned
        # checkpoint loads as a READY ArchModelWrapper (.predict/.score ->
        # calibrated fake/AIGC posterior). The arch (model_archs.effort) is a
        # CLIP-style ViT-L/14 with OrthAlign self-attn; _img_resize_chw feeds
        # (3,224,224) [0,1] CHW which score() ImageNet-normalises. Env unset /
        # arch unavailable -> None + last_reason, T2 falls back unchanged.
        _load=lambda p: _arch_aware_load(
            p, "effort", env_name="VISHWAS_EFFORT_WEIGHTS"),
    ),
    "VISHWAS_DEMAMBA_WEIGHTS": Adapter(
        env_name="VISHWAS_DEMAMBA_WEIGHTS",
        family="video",
        preprocess=_video_frame_preprocess,
        extract_prob=_auto_extract,
    ),
    "VISHWAS_FAKEMAMBA_WEIGHTS": Adapter(
        env_name="VISHWAS_FAKEMAMBA_WEIGHTS",
        family="audio",
        preprocess=_waveform_preprocess,
        extract_prob=_auto_extract,
        # 2026-08-23: family filled by RawBMamba (the one Mamba-family detector
        # with obtainable weights) routed through the arch-aware seam; see
        # model_archs/fakemamba.py. Env unset -> None + last_reason, heuristic
        # fallbacks unchanged.
        _load=lambda p: _arch_aware_load(
            p, "fakemamba", env_name="VISHWAS_FAKEMAMBA_WEIGHTS"),
    ),
    "VISHWAS_AASIST_WEIGHTS": Adapter(
        env_name="VISHWAS_AASIST_WEIGHTS",
        family="audio",
        # Spectra-AASIST3 wants a RAW 16 kHz waveform, not a mel matrix. The
        # learned arch (model_archs.aasist3) pre-emphasises + windows to its
        # 64,600-sample window inside score(); _waveform_preprocess just
        # decodes to PCM.
        preprocess=_waveform_preprocess,
        extract_prob=_auto_extract,
        # Phase 1 Task 1.2 + 2026-08-26: route through the arch-aware seam so
        # a provisioned checkpoint loads as a READY ArchModelWrapper (.predict/
        # .score -> calibrated spoof posterior). Family now 'aasist3'
        # (Spectra-AASIST3, proven AUC 0.9967) — see _ARCH_FAMILIES comment.
        # Env unset / arch unavailable -> None + last_reason, and T2 falls
        # back to its heuristic features unchanged.
        _load=lambda p: _arch_aware_load(
            p, "aasist3", env_name="VISHWAS_AASIST_WEIGHTS"),
    ),
    "VISHWAS_XLSRMAMBA_WEIGHTS": Adapter(
        env_name="VISHWAS_XLSRMAMBA_WEIGHTS",
        family="audio",
        # XLSR-Mamba-LA (MIT, arXiv 2411.10027) wants a RAW 16 kHz waveform;
        # the learned arch (model_archs.xlsrmamba) repeat-tiles to the trained
        # 66800-sample window inside score(); _waveform_preprocess just decodes
        # to PCM. 2026-08-24: routed through the arch-aware seam so a
        # provisioned checkpoint loads as a READY ArchModelWrapper (.predict/
        # .score -> calibrated spoof posterior). Env unset / arch unavailable
        # -> None + last_reason, sibling detectors unchanged. Replaces
        # RawBMamba as the Mamba-slot primary (that gate stays as fallback).
        preprocess=_waveform_preprocess,
        extract_prob=_auto_extract,
        _load=lambda p: _arch_aware_load(
            p, "xlsrmamba", env_name="VISHWAS_XLSRMAMBA_WEIGHTS"),
    ),
    "VISHWAS_SSL_AUDIO_WEIGHTS": Adapter(
        env_name="VISHWAS_SSL_AUDIO_WEIGHTS",
        family="audio",
        preprocess=_mel_ssl_preprocess,
        extract_prob=_auto_extract,
    ),
    "VISHWAS_HAVIC_WEIGHTS": Adapter(
        env_name="VISHWAS_HAVIC_WEIGHTS",
        family="video",
        preprocess=_video_frame_preprocess,
        extract_prob=_auto_extract,
        # Phase 1 Task 1.4: route through the arch-aware seam so a provisioned
        # checkpoint loads as a READY ArchModelWrapper whose .predict((audio,
        # video)) -> inconsistency posterior in [0,1] (HavicArch.score).
        # cross_modal supplies its own reference-faithful (audio, video)
        # preprocessing tuple; this adapter's preprocess stays the generic
        # frame-sequence helper for run_check() compatibility. Env unset /
        # arch unavailable -> None + last_reason, and the cross-modal stage
        # falls back to its heuristic AV probe unchanged.
        _load=lambda p: _arch_aware_load(
            p, "havic", env_name="VISHWAS_HAVIC_WEIGHTS"),
    ),
    "VISHWAS_IMAGE_FACE_WEIGHTS": Adapter(
        env_name="VISHWAS_IMAGE_FACE_WEIGHTS",
        family="face",
        # SPAI heavy gate (CVPR'25 spectral AI-image detector): preprocess is a
        # pass-through — spai.ArchSpec.score() decodes + normalises + pads, then
        # runs upstream-faithful inference. Wired through the arch-aware seam so
        # a provisioned checkpoint loads as a READY ArchModelWrapper (.predict ->
        # calibrated p_fake). Env unset / arch or weights unavailable -> None +
        # last_reason; the freqband heuristic fallback carries unchanged. Before
        # this change this slot fell back to VISHWAS_EFFORT_WEIGHTS (the
        # overfitting chameleon gate).
        preprocess=_spai_preprocess,
        extract_prob=_auto_extract,
        _load=lambda p: _arch_aware_load(
            p, "spai", env_name="VISHWAS_IMAGE_FACE_WEIGHTS"),
    ),
}


def resolve(env_name: str) -> Adapter | None:
    """Return the Adapter for *env_name*, or None if unregistered."""
    return ADAPTERS.get(env_name)


def run_check(adapter: Adapter, weight_path: str, raw_input: Any) -> tuple[str, dict, str]:
    """Convenience wrapper: adapter.run(weight_path, raw_input)."""
    return adapter.run(weight_path, raw_input)
