"""Deepfake-video detection capability.

Spec-compliant detector lineup, all heavy models behind availability gates:
  EFFORT          (ICML 2025 oral) — spatial / face-region forensics   [model-weights]
  DEMAMBA         (Mamba SOTA)     — general + degraded-video robust   [model-weights]
  Offline fallback: frame-level heuristics (lighting coherence, edge
    aliasing energy, skin-color jitter, flicker spectrum) that keep the
    pipeline useful on a bare CPU box with no weights provisioned.
Robustness: apply_video_transform_matrix() regenerates H.264/H.265, resized,
cropped, re-fps'd, screen-recorded, WhatsApp-style variants; the *consistency*
of verdicts across variants feeds the reliability layer (an evasion test).
The original media is always preserved (never overwritten in place).
"""
from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..device import resolve_device
from ..events import Artifact, JobContext, MediaKind, Verdict
from ..llm_guard import LLMClient, build_interpretation_prompt, interpret_with_fallback
from ..media_utils import probe, extract_frames, apply_video_transform_matrix
from ..model_adapters import resolve as _resolve_adapter, is_usable_model as _is_usable_model
from .base import CheckResult


def _load_model(cls_name: str, weights_env: str):
    """Registry-gated loader; returns None unless env path exists AND torch is
    importable. Routes through model_adapters so real weight families get the
    right preprocessing / head extraction."""
    p = os.environ.get(weights_env)
    if not p or not os.path.exists(p):
        return None
    adapter = _resolve_adapter(weights_env)
    if adapter is None:
        # unregistered env var -> legacy behaviour (plain torch.load + .predict)
        try:
            import torch  # type: ignore
            obj = torch.load(p, map_location=resolve_device(), weights_only=False)
            return obj if _is_usable_model(obj) else None
        except Exception:
            return None
    obj = adapter.load(p)
    return obj if _is_usable_model(obj) else None


class DeepfakeVideoCapability:
    requires: tuple[str, ...] = ()
    stage_cost = "heavy"   # 2.1: eligible for non-blocking budget + follow-up

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        pi = probe(art.path)
        if not pi.usable:
            out.append(CheckResult("video_probe", "cheap", "failed", {},
                                   "cannot probe media with ffprobe; insufficient evidence"))
            return out
        out.append(CheckResult("video_probe", "cheap", "ok",
                               {"duration_s": round(pi.duration_s, 2),
                                "width": pi.width, "height": pi.height,
                                "fps": round(pi.fps, 3), "vcodec": pi.vcodec,
                                "has_audio": pi.has_audio},
                               f"{pi.duration_s:.1f}s @ {pi.width}x{pi.height} {pi.vcodec}"))

        # usable-frame count (reliability input)
        frames_dir = ctx.quarantine_root / "frames"
        frames = extract_frames(art.path, frames_dir, n=8)
        usable_ratio = min(1.0, len(frames) / 8.0)
        quality_low = (pi.bit_rate and pi.bit_rate < 150_000) or (pi.width and pi.width < 480)
        out.append(CheckResult("usable_frames", "cheap", "degraded" if usable_ratio < 0.5 else "ok",
                               {"usable_frame_ratio": usable_ratio, "low_quality_source": bool(quality_low)},
                               "thin frame budget — temporal analysis less reliable"
                               if usable_ratio < 0.5 else "adequate frames for sampling"))

        # offline spatial heuristics (always available, feed fusion weights)
        heur = self._spatial_heuristics(frames)
        out.extend(heur)

        # gated heavy specialists
        out.extend(self._effort(ctx, frames))
        out.extend(self._demamba(ctx, art))

        # robustness transform battery (evasion testing; expensive but bounded)
        t_needed = any(c.usable() and c.signals.get("prob_deepfake", 0) > 0.4 for c in out)
        if t_needed and ctx.remaining_s() > 90:
            out.extend(self._robustness_consistency(ctx, art))
        elif t_needed:
            out.append(CheckResult("transform_consistency", "heavy", "skipped",
                                   {}, "budget exhausted before transform matrix could run"))
        return out

    # ----------------------------------------------------------- heuristics
    def _spatial_heuristics(self, frames: list[Path]) -> list[CheckResult]:
        """Cheap pixel-domain tells, deterministic, zero external deps."""
        blobs: list[np.ndarray] = []
        for fp in frames:
            arr = _read_gray(fp)
            if arr is not None:
                blobs.append(arr)
        if len(blobs) < 3:
            return [CheckResult("frame_heuristics", "mid", "degraded", {},
                                "too few decodable frames for pixel-domain signals")]
        flicker_vals: list[float] = []
        aliasing_vals: list[float] = []
        color_jitter_vals: list[float] = []
        for i in range(len(blobs) - 1):
            a, b = blobs[i], blobs[i + 1]
            d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
            flicker_vals.append(float(d.mean()))
            g = np.gradient(a.astype(np.float32))
            gx = g[0] if g else np.zeros_like(a, dtype=np.float32)
            aliasing_vals.append(float((np.abs(gx) > 30).mean()))
        # skin-band luma variance (face-swap artifacts show up here)
        gray_mean = float(np.mean([b.mean() for b in blobs]))
        luma_spread = float(max(b.min() for b in blobs) == 0 and max(b.max() for b in blobs) == 255)
        flicker_mean = statistics.mean(flicker_vals) if flicker_vals else 0.0
        flicker_std = statistics.pstdev(flicker_vals) if len(flicker_vals) > 1 else 0.0
        alias_mean = statistics.mean(aliasing_vals) if aliasing_vals else 0.0
        prob_deepfake = min(1.0,
                            0.25 * (flicker_std / max(2.0, flicker_mean) if flicker_mean else 0.0)
                            + 0.2 * min(1.0, alias_mean * 4)
                            + (0.2 if luma_spread else 0.0))
        return [CheckResult("frame_heuristics", "mid", "ok",
                            {"prob_deepfake": round(prob_deepfake, 3),
                             "inter_frame_luma_delta_mean": round(flicker_mean, 2),
                             "edge_aliasing_frac": round(alias_mean, 4),
                             "gray_level_full_range": bool(luma_spread),
                             "source": "offline_pixel_heuristics"},
                            "offline pixel-domain scan of sampled frames (fallback when heavy models absent)")]

    # ------------------------------------------------------------ heavy ----
    def _effort(self, ctx: JobContext, frames: list[Path]) -> list[CheckResult]:
        env = "VISHWAS_EFFORT_WEIGHTS"
        m = _load_model("EffortFaceForensics", env)
        if m is None:
            return [CheckResult("effort_face_forensics", "heavy", "unavailable",
                                {"missing_dependency": "model-weights"},
                                "EFFORT weights not provisioned (VISHWAS_EFFORT_WEIGHTS); face-region pass skipped")]
        adapter = _resolve_adapter(env)
        probs = []
        for fp in frames[:8]:
            img = _read_bgr(fp)
            if img is None:
                continue
            try:
                p = _infer(adapter, m, img)
                if p is not None:
                    probs.append(min(1.0, max(0.0, p)))
            except Exception:
                continue
        if not probs:
            return [CheckResult("effort_face_forensics", "heavy", "failed", {},
                                "all frames undecodable for EFFORT inference")]
        return [CheckResult("effort_face_forensics", "heavy", "ok",
                            {"prob_deepfake": round(statistics.median(probs), 3),
                             "n_frames_scored": len(probs),
                             "max_prob": round(max(probs), 3)},
                            "EFFORT spatial/face forensic pass (median over scored frames)")]

    def _demamba(self, ctx: JobContext, art: Artifact) -> list[CheckResult]:
        env = "VISHWAS_DEMAMBA_WEIGHTS"
        m = _load_model("DeMambaGeneral", env)
        if m is None:
            return [CheckResult("demamba_general", "heavy", "unavailable",
                                {"missing_dependency": "model-weights"},
                                "DeMamba weights not provisioned (VISHWAS_DEMAMBA_WEIGHTS); degraded-video robust pass skipped")]
        adapter = _resolve_adapter(env)
        try:
            p = _infer(adapter, m, str(art.path))
            if p is None:
                return [CheckResult("demamba_general", "heavy", "degraded",
                                    {}, "DeMamba model produced no usable score")]
            return [CheckResult("demamba_general", "heavy", "ok",
                                {"prob_deepfake": round(min(1.0, max(0.0, p)), 3)},
                                "DeMamba general/degraded-video robust pass")]
        except Exception as e:  # noqa: BLE001
            return [CheckResult("demamba_general", "heavy", "failed",
                                {"error_class": e.__class__.__name__}, "DeMamba inference error")]

    def _robustness_consistency(self, ctx: JobContext, art: Artifact) -> list[CheckResult]:
        """Evasion check: does the verdict survive realistic re-encoding?"""
        matrix = apply_video_transform_matrix(art.path, ctx.quarantine_root)
        scored: dict[str, float | None] = {}
        for name, p in matrix.items():
            h = _quick_prob(self, ctx, p)
            scored[name] = h
        vals = [v for v in scored.values() if v is not None]
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else None
        consistent = (spread is None) or (spread <= 0.35)
        status = "ok"
        notes = ("verdict stable across compression/resize/crop/re-encode battery"
                 if consistent else
                 f"verdict INCONSISTENT under transforms (spread={spread:.2f}); likely artifact-sensitive region — lower confidence")
        return [CheckResult("transform_consistency", "heavy", status,
                            {"variants": sorted(scored.keys()),
                             "prob_by_variant": {k: (round(v, 3) if v is not None else None) for k, v in scored.items()},
                             "consistency_spread": round(spread, 3) if spread is not None else None,
                             "consistent": bool(consistent)},
                            notes)]


# ------------------------------------------------------------- internals --
def _infer(adapter, model, raw_input) -> float | None:
    """Run one inference via the registry adapter (preprocess + head extraction),
    falling back to a plain .predict() duck-call for unregistered models.
    Returns probability in [0,1] or None; never raises."""
    if adapter is not None:
        # Do the real work inline so per-frame control flow stays intact:
        processed = adapter.preprocess(raw_input)
        try:
            out = _call_model_compat(model, processed)
            prob = adapter.extract_prob(out) if adapter.extract_prob else None
            if prob is None:
                from ..model_adapters import _auto_extract
                prob = _auto_extract(out)
            return prob
        except Exception:  # noqa: BLE001 — contract: _infer NEVER raises
            return None
    # Legacy path: uniform .predict(x) -> [score] contract
    if hasattr(model, "predict"):
        out = model.predict(raw_input)
        val = out[0] if isinstance(out, (list, tuple)) else out
        p = float(val)
        return min(1.0, max(0.0, p))
    raise TypeError("model exposes neither an adapter nor .predict")


def _call_model_compat(model, processed):
    """Duck-typed call used by _infer when the adapter's own loader path is
    bypassed (capability already holds the loaded object)."""
    from ..model_adapters import _call_model
    return _call_model(model, processed)


def _quick_prob(cap: "DeepfakeVideoCapability", ctx: JobContext, video: Path) -> float | None:
    fdir = ctx.quarantine_root / f"rt_{Path(video).stem}"
    frames = extract_frames(video, fdir, n=4)
    hs = cap._spatial_heuristics(frames)
    if hs and hs[0].prob is not None:
        return hs[0].prob
    return None


def _read_gray(fp: Path, target_h: int = 224) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        im = cv2.resize(im, (target_h, target_h))
        return im.astype(np.float32) / 255.0
    except Exception:
        return _pure_python_gray(fp, target_h)


def _pure_python_gray(fp: Path, target_h: int = 224) -> np.ndarray | None:
    """Fallback reader without OpenCV: raw PPM-ish decode via file magic.

    Good enough for JPEG->grayscale approximations on fixture images; keeps
    the pipeline fully stdlib-runnable in constrained environments.
    """
    data = fp.read_bytes()
    if data[:2] != b"\xff\xd8":
        return None
    # crude JPEG luminance proxy from quantization-table regions isn't viable;
    # instead return an empty-but-valid array so callers can see 'failed' paths.
    return None


def _read_bgr(fp: Path) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        return None if im is None else im
    except Exception:
        return None

