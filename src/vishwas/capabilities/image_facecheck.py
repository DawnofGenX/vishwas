"""Image face-check capability: still-photo face-manipulation forensics.

WhatsApp users often send static photos of "relatives"/"officials" asking for
money or confirmation. Pipeline:
  T0 cheap : magic-byte verify (already done upstream) + basic integrity scan
             (exif sanity, resolution, per-band luma stats)
  T1 heavy : gated face-forensics model (EFFORT weights reuse; env override
             VISHWAS_IMAGE_FACE_WEIGHTS allowed to point at a dedicated one)
Offline fallback: frequency-domain anomaly scoring (spectral discontinuity in
the mid-bands where swap-paste seams live) so a bare CPU box returns something
real rather than nothing.
"""
from __future__ import annotations

import logging
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from ..device import resolve_device
from ..events import Artifact, JobContext, MediaKind
from ..model_adapters import resolve as _resolve_adapter, _call_model as _call_model_compat, _auto_extract, is_usable_model as _is_usable_model
from .base import CheckResult

log = logging.getLogger("vishwas.image_facecheck")

IMAGE_KINDS = {MediaKind.PNG, MediaKind.JPEG, MediaKind.WEBP, MediaKind.GIF,
               MediaKind.TIFF, MediaKind.HEIC}

# 2026-08-27 image false-clean fix: NYUAD 3-class AI-image detector runs under the
# isolated .venv-ambient tree (transformers 5.14.1 + torch 2.13) because the
# webhook's docling-python/transformers-5.15 stack cannot import any ViT (same
# GEN_EMAIL/circular-import failure as audio Wav2Vec2Model). Complements SPAI
# (spectral) which scores some flux AI images ~0.0. Verified discriminates on our
# corpus (flux_00/01/04 caught at 0.98-0.999; real photos 0.002-0.04).
_NYUAD_HELPER = str(Path(__file__).resolve().parents[3] / "scripts" / "image_score_nyuad.py")
_VENV_AMBIENT_PY = "/home/hermes/.venv-ambient/bin/python"
_NYUAD_SUBPROC_TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# GENIMAGE — the THIRD image signal (2026-10-22)
# ---------------------------------------------------------------------------
# The image channel had exactly TWO independent signals (SPAI and NYUAD) and
# they could not separate the two groups that matter: 5 genuine photos
# flagged by NYUAD alone, and 9 AI images caught by NYUAD alone. Both groups
# carry the SAME signature (SPAI low, NYUAD high), so no rule over
# (SPAI, NYUAD) can separate them — the fusion layer's own comment says a
# THIRD signal is required.
#
# The upstream EFFORT project (YZY-stack/Effort-AIGI-Detection) ships three
# checkpoints of the same architecture: chameleon (face/spatial), ffpp (face
# forensics, IN USE for video) and genimage (GENERAL AIGC image detection,
# trained on GenImage/SD-v1.4). genimage was NOT on this host; it was
# fetched from the upstream public Drive folder and hash-verified against the
# manifest (sha256_first12 7c32ceb4e66d).
#
# MEASURED on the raw 60-image corpus through the production path:
#     genimage AUC 0.9211  (AI median 0.9970, real median 0.4334)
# and it separates the two groups SPAI+NYUAD could not:
#     real photos NYUAD-only (5): genimage 0.3813 .. 0.9991
#     AI images  NYUAD-only (9): genimage 0.9442 .. 0.9999
#     best threshold 0.9855 -> catches 8/9 AI, flags 1/5 real
#
# IT IS AN IMAGE DETECTOR, NOT A VIDEO DETECTOR. Measured on the operator's
# AI VIDEO frames it is INVERTED (AI 0.037/0.107/0.098 vs real
# 0.225/0.064/0.277) — h264 compression plus the adapter's center-crop on a
# wide frame destroy the signal. So it is wired into the IMAGE channel only
# and must never be used to score video frames.
#
# The checkpoint uses the SAME architecture as the EFFORT gate (CLIP ViT-L/14
# + OrthAlign), so it loads through the same adapter/spec — but under its OWN
# env var, because the video channel's VISHWAS_EFFORT_WEIGHTS must keep
# pointing at the ffpp checkpoint.
_GENIMAGE_DEFAULT = "/opt/vishwas/models/effort/genimage/effort_genimage.pth"
_GENIMAGE_ENV = "VISHWAS_EFFORT_GENIMAGE_WEIGHTS"


def _genimage_path() -> Path:
    return Path(os.environ.get(_GENIMAGE_ENV, _GENIMAGE_DEFAULT))


# ⚠️ THE MODEL CACHE (2026-10-22). The genimage checkpoint used to be
# rebuilt and reloaded for EVERY image (the `spec.build()` + `torch.load()`
# were inside `_genimage_prob`). That is the same defect PR #28 fixed in
# the AUDIO lane, where aasist3 reloaded once per crop and pushed the
# audio lane past its budget.
#
# The cache is keyed on the checkpoint PATH (a different checkpoint must
# never serve a stale model), it is BOUNDED, and a FAILED load is NOT
# cached (a checkpoint provisioned later must be picked up, not
# permanently poisoned by an earlier failure).
_GENIMAGE_MODEL_CACHE: dict = {}
_GENIMAGE_MODEL_CACHE_MAX = 4


def _genimage_model(path: Path):
    """Build the genimage model ONCE per checkpoint path and reuse it.

    ⚠️ THIS WAS A PER-CALL MODEL BUILD (2026-10-22). `_genimage_prob`
    called `spec.build()` and `torch.load()` INSIDE itself, so EVERY image
    reloaded the checkpoint. That is the same defect PR #28 fixed in the
    AUDIO lane (where aasist3 reloaded once per crop and pushed the lane
    past its budget) — the image lane had the identical shape, unfixed.

    The cache is keyed on the checkpoint PATH (so a different checkpoint
    never serves a stale model), it is BOUNDED (one entry per distinct
    path, capped), and a FAILED load is NOT cached — a checkpoint that is
    provisioned later must be picked up, not permanently poisoned by the
    failure that preceded it.
    """
    key = str(path)
    cached = _GENIMAGE_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    from ..model_archs.effort import get_arch
    spec = get_arch()
    model = spec.build()
    import torch

    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if not spec.apply_state(model, sd):
        return None
    model.eval()
    dev = resolve_device()
    model.to(dev)
    # ⚠️ THE DEVICE IS STORED SEPARATELY. The arch's `_EffortNet` is a
    # plain nn.Module wrapper and has no `.device` attribute, so the
    # device the model was moved to must be remembered alongside it —
    # `next(model.parameters()).device` would work, but keeping the
    # resolved device is explicit and avoids a parameter scan per image.
    entry = (model, dev)
    if len(_GENIMAGE_MODEL_CACHE) >= _GENIMAGE_MODEL_CACHE_MAX:
        _GENIMAGE_MODEL_CACHE.clear()
    _GENIMAGE_MODEL_CACHE[key] = entry
    return entry


def _genimage_prob(img: np.ndarray | None) -> float | None:
    """Score one image with the genimage (general-AIGC) EFFORT checkpoint.

    Returns p_fake in [0,1], or None if the checkpoint or the model is not
    available. NEVER fabricates a score: an unavailable checkpoint must
    degrade the channel to an honest 'unavailable' check, not to a fake
    'clean' reading.

    The same adapter/spec as the EFFORT gate is used (identical architecture),
    but the weights come from the genimage checkpoint and the model is built
    from the image (NOT the face crop — genimage is a whole-image detector).
    """
    if img is None:
        return None
    path = _genimage_path()
    if not path.exists():
        return None
    try:
        entry = _genimage_model(path)
        if entry is None:
            return None
        model, dev = entry
        import torch

        chw = _preprocess_chw(img)
        x = torch.from_numpy(chw).unsqueeze(0).to(dev)
        with torch.no_grad():
            logits = model(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            p = torch.softmax(logits.float(), dim=-1)[0, 1].item()
        return min(1.0, max(0.0, float(p)))
    except Exception as exc:  # noqa: BLE001
        log.warning("genimage scoring failed (%s: %s) — third image signal unavailable",
                    type(exc).__name__, exc)
        # A FAILURE IS NOT CACHED: the checkpoint may be provisioned later,
        # and a cached failure would be permanent.
        return None


def _preprocess_chw(img: np.ndarray) -> np.ndarray:
    """BGR HWC uint8 -> CHW float32, CLIP-normalised, 224x224.

    Mirrors the EFFORT gate's own preprocessing: center-crop square, resize to
    224, BGR->RGB, then CLIP's normalisation (mean [0.48145466, 0.4578275,
    0.40821073], std [0.26862954, 0.26130258, 0.27577711]).
    """
    import cv2

    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    crop = img[y0:y0 + s, x0:x0 + s]
    resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    return ((chw - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


def _subprocess_nyuad_score(path: Path, timeout_s: float = _NYUAD_SUBPROC_TIMEOUT_S) -> float | None:
    """Score one image via the .venv-ambient NYUAD helper. Returns p_fake in [0,1]
    or None on any failure (never fabricates). MUST run with a CLEAN PYTHONPATH so
    .venv-ambient's own transformers/torch win (the webhook's inherited
    docling-python/pylibs shadow them with the broken transformers-5.15 stack).

    Mirrors deepfake_audio._subprocess_aasist3_score (2026-10-21 hardening): every
    failure mode is logged loudly rather than swallowed — a silently dead
    interpreter (the .venv-ambient tree is an out-of-tree dependency that can break
    on its own) must be visible, not inferred from a suspicious absence of signal.
    A None return downgrades the channel to an honest 'unavailable' check; the
    caller (analyze) never invents a score from it.
    """
    import copy
    import subprocess
    venv_env = copy.deepcopy(dict(os.environ))
    venv_env.pop("PYTHONPATH", None)  # let .venv-ambient site-packages resolve
    if not os.path.exists(_VENV_AMBIENT_PY):
        log.warning(
            "nyuad helper interpreter missing: %s — NYUAD second image signal "
            "unavailable (SPAI carries the image channel). Repair with: "
            "uv python install 3.11 && see deploy/vishwas-secrets.env",
            _VENV_AMBIENT_PY,
        )
        return None
    if not os.path.exists(_NYUAD_HELPER):
        log.warning("nyuad helper script missing: %s — NYUAD unavailable", _NYUAD_HELPER)
        return None
    try:
        r = subprocess.run(
            [_VENV_AMBIENT_PY, _NYUAD_HELPER, str(path)],
            capture_output=True, text=True, timeout=timeout_s, env=venv_env)
    except Exception as exc:  # noqa: BLE001
        log.warning("nyuad helper failed to execute (%s: %s) — NYUAD unavailable",
                    type(exc).__name__, exc)
        return None
    if r.returncode != 0:
        log.warning("nyuad helper exited %s — NYUAD unavailable (stderr: %s)",
                    r.returncode, (r.stderr or "").strip()[:200])
        return None
    for line in r.stdout.splitlines():
        x = line.strip()
        if x.startswith("#") or not x:
            continue
        try:
            p = float(x)
            return min(1.0, max(0.0, p))
        except ValueError:
            continue
    log.warning("nyuad helper produced no parseable posterior — NYUAD unavailable")
    return None


def _load_image(p: Path, side: int = 512) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            return None
        return cv2.resize(im, (side, side))
    except Exception:
        return None


def _load_original_image(p: Path) -> np.ndarray | None:
    """Full-resolution BGR decode (no resize). SPAI's heavy gate is any-resolution
    spectral and needs the ORIGINAL pixels; the fixed 512 resize collapses it."""
    try:
        import cv2  # type: ignore
        return cv2.imread(str(p), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _spectral_band_anomaly(img: np.ndarray) -> float:
    """Mid-frequency energy ratio proxy. Swap/blend seams and GAN artifacts
    typically push energy into specific mid bands; real photos have a smoother
    falloff. Bounded heuristic -> 0..1."""
    gray = img.mean(axis=2).astype(np.float32)
    F = np.abs(np.fft.fft2(gray - gray.mean()))
    h, w = F.shape
    cy, cx = h // 2, w // 2
    # annuli around DC
    def ring(r_in: int, r_out: int) -> float:
        yy, xx = np.mgrid[0:h, 0:w]
        dy, dx = yy - cy, xx - cx
        m = (dy * dy + dx * dx >= r_in ** 2) & (dy * dy + dx * dx < r_out ** 2)
        return float(F[m].mean()) if m.any() else 0.0
    low = ring(4, 64)
    mid = ring(64, 128)
    high = ring(128, 256)
    denom = (low + mid + high) or 1.0
    ratio = mid / denom                     # typical real photo ~0.25-0.4
    anomaly = abs(ratio - 0.32) / 0.32      # deviation from expected band mix
    return round(min(1.0, anomaly), 3)


class ImageFaceCheckCapability:
    requires: tuple[str, ...] = ()

    # QR evidence for photographed gov IDs (gap fix 2026-08-25): the QR
    # verifier lives in GovDocumentCapability, which photo'd IDs never reach
    # (they route here). When the filename carries gov hints, borrow that
    # check so aadhaar.jpg etc. get qr_payload_check evidence too.
    _gov_hints = None

    def _qr_evidence_for_gov_image(self, art: Artifact) -> list[CheckResult]:
        if ImageFaceCheckCapability._gov_hints is None:
            from ..router import _GOV_HINTS
            ImageFaceCheckCapability._gov_hints = _GOV_HINTS
        name = (art.original_filename or "").lower()
        if not ImageFaceCheckCapability._gov_hints.search(name):
            return []
        try:
            from .gov_document import GovDocumentCapability
            return GovDocumentCapability()._qr_payload_checks(art)
        except Exception:  # noqa: BLE001 — QR is additive evidence, never fatal
            return []

    def analyze(self, art: Artifact, ctx: JobContext) -> list[CheckResult]:
        out: list[CheckResult] = []
        kind = art.verified_kind
        if kind is None or kind not in IMAGE_KINDS:
            out.append(CheckResult("image_check", "cheap", "skipped", {},
                                   "verified content is not an image"))
            return out
        out.append(CheckResult("image_integrity", "cheap", "ok",
                               {"format": kind.value,
                                "size_bytes": art.size_bytes,
                                "sha256": art.sha256},
                               "container-level integrity ok"))
        img = _load_image(art.path)
        if img is None:
            out.append(CheckResult("image_face_forensics", "mid", "degraded", {},
                                   "could not decode pixel data with available decoders"))
            out.extend(self._qr_evidence_for_gov_image(art))
            return out
        anomaly = _spectral_band_anomaly(img)
        out.append(CheckResult("frequency_band_analysis", "mid", "ok",
                               {"prob_deepfake": round(min(1.0, 0.35 + 0.4 * anomaly), 3),
                                "band_anomaly": anomaly,
                                "source": "offline_frequency_heuristics"},
                               "offline frequency-band scan (fallback when face models absent)"))
        # Gated learned model — VISHWAS_IMAGE_FACE_WEIGHTS accepts SPAI or the
        # fine-tuning pipeline's metadata-tagged EfficientNet-B0 checkpoint.
        # EFFORT fallback below is backward-compat ONLY when the image-specific
        # env is unset.
        # SPAI is ANY-RESOLUTION SPECTRAL: feeding it the fixed 512x512 resize
        # (img, used above for the freqband heuristic) collapsed its detection —
        # it needs the ORIGINAL-resolution photo. Load the full-res image for the
        # heavy gate; the freqband heuristic keeps the 512 path unchanged.
        heavy_img = _load_original_image(art.path)
        if heavy_img is None:
            heavy_img = img
        env, _p = _resolved_weights_env()
        adapter = _resolve_adapter(env) if env else None
        if env == "VISHWAS_IMAGE_FACE_WEIGHTS" and adapter is None:
            # Belt-and-braces: if the dedicated image adapter ever fails to
            # register, DO NOT silently fall through to the EFFORT chameleon gate
            # (the overfit that motivated this fix).
            out.append(CheckResult("image_face_forensics", "heavy", "unavailable",
                                   {"missing_dependency": "model-weights"},
                                   "image-forensics gate not provisioned; frequency heuristic carries"))
            out.extend(self._qr_evidence_for_gov_image(art))
            return out
        m = _load_model()
        if m is None:
            out.append(CheckResult("image_face_forensics", "heavy", "unavailable",
                                   {"missing_dependency": "model-weights"},
                                   "face-forensics weights not provisioned; frequency heuristic carries"))
        else:
            try:
                p = _infer_face(adapter, m, heavy_img)
                if p is None:
                    out.append(CheckResult("image_face_forensics", "heavy", "degraded",
                                           {}, "face-forensics model produced no usable score"))
                    return out
                out.append(CheckResult("image_face_forensics", "heavy", "ok",
                                       {"prob_deepfake": round(min(1.0, max(0.0, p)), 3)},
                                       "learned face-manipulation detector pass"))
            except Exception as e:  # noqa: BLE001
                out.append(CheckResult("image_face_forensics", "heavy", "failed",
                                       {"error_class": e.__class__.__name__}, "inference error"))
        # NYUAD 3-class AI-image detector (second, INDEPENDENT signal) — runs under
        # .venv-ambient (the webhook's transformers-5.15 stack can't import ViT).
        # Complements SPAI: catches flux AI that SPAI scores ~0.0. If unavailable
        # (no .venv-ambient), honest-unavailable — never fabricates a score.
        _nyuad_p = None
        if os.path.exists(_VENV_AMBIENT_PY) and os.path.exists(_NYUAD_HELPER):
            _nyuad_p = _subprocess_nyuad_score(art.path)
        if _nyuad_p is None:
            out.append(CheckResult("nyuad_image_detector", "heavy", "unavailable",
                                   {"missing_dependency": "subprocess-env"},
                                   "NYUAD AI-image detector unavailable (no .venv-ambient); SPAI carries"))
        else:
            out.append(CheckResult("nyuad_image_detector", "heavy", "ok",
                                   {"prob_deepfake": round(_nyuad_p, 3),
                                    "source": "nyuad.subprocess(.venv-ambient)"},
                                   "NYUAD 3-class AI-image detector pass"))
        # GENIMAGE — the THIRD independent image signal (2026-10-22).
        # Measured on the raw 60-image corpus: AUC 0.9211 (AI median 0.9970,
        # real median 0.4334) — the best single signal on this corpus, and the
        # one that separates the two groups SPAI+NYUAD could not (5 real
        # photos flagged by NYUAD alone vs 9 AI images caught by NYUAD alone,
        # all with the SAME SPAI-low/NYUAD-high signature).
        # It runs on the FULL-RESOLUTION image, not the 512x512 resize used
        # by the freqband heuristic.
        _gen_p = _genimage_prob(heavy_img)
        if _gen_p is None:
            out.append(CheckResult("genimage_image_detector", "heavy", "unavailable",
                                   {"missing_dependency": "model-weights"},
                                   "genimage general-AIGC detector unavailable "
                                   "(checkpoint not provisioned); SPAI+NYUAD carry the image channel"))
        else:
            out.append(CheckResult("genimage_image_detector", "heavy", "ok",
                                   {"prob_deepfake": round(_gen_p, 3),
                                    "source": "effort.genimage"},
                                   "EFFORT genimage (GenImage/SD-v1.4) general-AIGC pass — "
                                   "third independent image signal"))
        out.extend(self._qr_evidence_for_gov_image(art))
        return out


def _resolved_weights_env() -> tuple[str | None, str | None]:
    """Return (env_name, weight_path) honouring the documented override order:
    VISHWAS_IMAGE_FACE_WEIGHTS (SPAI heavy gate) first; else fall back to
    VISHWAS_EFFORT_WEIGHTS for backward-compat ONLY when the image-specific env
    is unset."""
    p = os.environ.get("VISHWAS_IMAGE_FACE_WEIGHTS")
    if p:
        return "VISHWAS_IMAGE_FACE_WEIGHTS", p
    p = os.environ.get("VISHWAS_EFFORT_WEIGHTS")
    if p:
        return "VISHWAS_EFFORT_WEIGHTS", p
    return None, None


def _load_model():
    """Registry-gated loader; returns None unless the resolved env path exists AND
    torch is importable. Routes through model_adapters (face or EFFORT family)."""
    env, p = _resolved_weights_env()
    if not p or not os.path.exists(p):
        return None
    adapter = _resolve_adapter(env) if env else None
    if adapter is not None:
        obj = adapter.load(p)
        return obj if _is_usable_model(obj) else None
    try:
        import torch  # type: ignore
        obj = torch.load(p, map_location=resolve_device(), weights_only=False)
        return obj if _is_usable_model(obj) else None
    except Exception:
        return None


def _infer_face(adapter, model, img: np.ndarray) -> float | None:
    """One face-forensics inference via the registry (or legacy .predict).
    [0,1] or None; never raises."""
    if adapter is not None:
        processed = adapter.preprocess(img)
        out = _call_model_compat(model, processed)
        prob = adapter.extract_prob(out) if adapter.extract_prob else None
        if prob is None:
            prob = _auto_extract(out)
        return prob
    if hasattr(model, "predict"):
        out = model.predict(img)
        val = out[0] if isinstance(out, (list, tuple)) else out
        return min(1.0, max(0.0, float(val)))
    return None
