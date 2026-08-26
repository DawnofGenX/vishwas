"""Spectra-AASIST3 architecture spec (audio anti-spoofing).

VENDORED / DROP-IN PROVENANCE
=============================
Checkpoint: /opt/verisafe/models/aasist3/model.safetensors
  HF card : lab260/Spectra-AASIST3 (Apache-2.0)
  Evidence: independently re-scored by the Speech Anti-Spoofing Arena —
            EER 0.97% ASVspoof2019-LA (71,237 trials), 4.38% 2021-LA,
            1.20% In-The-Wild; sha-pinned scores.txt in-repo.
  Net def : model_archs/_spectra/model.py (verbatim from the repo, imports
            re-homed). SSL frontend = HF Wav2Vec2 XLS-R-300m; weights for
            it live in _spectra/wav2vec2-xls-r-300m/ (config + safetensors
            fetched from facebook/wav2vec2-xls-r-300m, MIT).
  Bridge  : MLPBridge(1024 -> 128); backend KANAASIST (KANLinear head,
            nb_samp 64400 in d_args but wrapper windows to 64,600 — we
            follow the WRAPPER's documented preprocessing, which is what
            produced the Arena numbers).

INPUT CONTRACT (replicates the Arena wrapper exactly):
  16 kHz mono float32 -> preemphasis 0.97 over the full waveform ->
  deterministic first-64,600-sample window (tile-repeat if shorter) ->
  forward -> logits[:, 1] == BONA FIDE logit.
POLARITY INVERSION (X5): downstream expects SPOOF posterior;
  p_spoof = 1 - sigmoid(logit_bonafide).

WHY THIS SWAP (see aasist.py header): the previous HABLA_WavLM_AASIST
checkpoint is degenerate — no input-dependent decision function at eval
mode (19-probe GPU post-mortem, commit 35411b0). Spectra-AASIST3 is the
documented replacement with permissive license and pinned third-party
evidence. Old checkpoint kept as rollback in deploy/vishwas-secrets.env.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base import ArchSpec
from ..device import resolve_device

# Vendor path so `_spectra.model` can import its sibling modules and find
# the local wav2vec2-xls-r-300m dir without network access.
_VENDOR_DIR = Path(__file__).resolve().parent / "_spectra"
if str(_VENDOR_DIR.parent.parent) not in sys.path:
    pass  # package-relative import below avoids sys.path games

_INPUT_SAMPLES = 64_600
_PREEMPHASIS = 0.97

#: memoised combined wav2vec2 snapshot dirs, keyed by the weights dir. The
#: weights area holds ONLY pytorch_model.bin; the XLS-R-300m config + preproc
#: live in the vendored net tree. from_pretrained needs all three in one dir or
#: it instantiates a base-768 default-config shell that can't carry the
#: 1024-dim checkpoint (Track C combined them; we do the same, in a temp dir,
#: never writing to the weights area).
_w2v_snapshots: dict[str, str] = {}


def _ensure_w2v_snapshot(w2v_dir: Path) -> Path:
    """Return a complete wav2vec2 dir (config + preproc + bin) for from_pretrained."""
    key = str(w2v_dir)
    if key in _w2v_snapshots:
        return Path(_w2v_snapshots[key])
    vendored = _VENDOR_DIR / "wav2vec2-xls-r-300m"
    has_config = (w2v_dir / "config.json").exists()
    if has_config:
        _w2v_snapshots[key] = key  # complete already
        return w2v_dir
    out = Path(tempfile.mkdtemp(prefix="vishwas_w2v_"))
    for f in ("config.json", "preprocessor_config.json"):
        c = vendored / f
        if c.exists():
            (out / f).write_bytes(c.read_bytes())
    bin_src = w2v_dir / "pytorch_model.bin"
    if bin_src.exists():
        try:
            os.symlink(bin_src, out / "pytorch_model.bin")
        except OSError:
            shutil.copy(bin_src, out / "pytorch_model.bin")
    _w2v_snapshots[key] = str(out)
    return out


def _load_net_class():
    """Import the vendored net definition, pointed at local wav2vec2."""
    import importlib.util
    spec_path = _VENDOR_DIR / "model.py"
    spec = importlib.util.spec_from_file_location("vishwas_model_archs__spectra_model", spec_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vishwas_model_archs__spectra_model"] = mod
    # Patch Wav2Vec2Model.from_pretrained default: resolve to LOCAL snapshot
    # so init never hits the network (offline-first rule).
    spec.loader.exec_module(mod)
    return mod


def _preprocess_waveform(wav: np.ndarray | torch.Tensor) -> torch.Tensor:
    """16 kHz mono waveform -> (1, 64600) float32 tensor, Arena pipeline."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float().numpy()
    wav = np.ascontiguousarray(wav, dtype=np.float32).reshape(-1)
    # preemphasis 0.97 on the FULL waveform (y[n] = x[n] - 0.97 x[n-1])
    if wav.size > 1:
        out = np.empty_like(wav)
        out[0] = wav[0]
        np.subtract(wav[1:], _PREEMPHASIS * wav[:-1], out=out[1:])
        wav = out
    if wav.size < _INPUT_SAMPLES:
        reps = _INPUT_SAMPLES // max(wav.size, 1) + 1
        wav = np.tile(wav, reps)[:_INPUT_SAMPLES]
    else:
        wav = wav[:_INPUT_SAMPLES]
    return torch.from_numpy(wav).unsqueeze(0)  # (1, 64600)


class SpectraAASIST3Spec(ArchSpec):
    name = "aasist3"
    weight_env = "VISHWAS_AASIST_WEIGHTS"
    implemented = True

    def build(self) -> Any:
        mod = _load_net_class()
        # Point the vendored encoder at the LOCAL wav2vec2 snapshot (kept OUT of
        # git, in the weights area, not the repo tree).
        w2v_dir = Path(os.environ.get(
            "VISHWAS_SPECTRA_W2V_DIR",
            "/opt/verisafe/models/aasist3/wav2vec2-xls-r-300m"))
        orig_from_pretrained = mod.Wav2Vec2Model.from_pretrained

        def _local_from_pretrained(name_or_path=None, *a, **kw):
            kw.setdefault("local_files_only", True)
            # transformers >= 5.0 rejects gradient_checkpointing= from_pretrained
            # kwargs (no external score_eval shim needed). Production inference
            # runs eval-only, so the flag is always False here anyway.
            kw.pop("gradient_checkpointing", None)
            # Combine the weights-dir bin with the vendored config (see
            # _ensure_w2v_snapshot) so the correct XLS-R-300m shell is built.
            return orig_from_pretrained(
                str(_ensure_w2v_snapshot(w2v_dir)), *a, **kw)

        mod.Wav2Vec2Model.from_pretrained = staticmethod(_local_from_pretrained)
        try:
            net = mod.SpectraAASIST3()
        finally:
            mod.Wav2Vec2Model.from_pretrained = orig_from_pretrained
        # Apply the checkpoint HERE — build() must return a loaded model, not a
        # randomly-initialised one. Weights live at
        # /opt/verisafe/models/aasist3/model.safetensors (path from weight_env).
        from safetensors.torch import load_file
        ck_path = os.environ.get(self.weight_env, "")
        if ck_path and os.path.exists(ck_path):
            sd = load_file(ck_path)
            missing, unexpected = net.load_state_dict(sd, strict=False)
            n_missing = len(missing)
            if n_missing > 0.05 * len(sd) or n_missing:
                # hard-fail on ANY missing key: 1022/1022 expected
                raise RuntimeError(
                    f"aasist3 checkpoint coverage incomplete: {n_missing} missing "
                    f"(first: {missing[:3]}), {len(unexpected)} unexpected")
            net.eval()
        net.to(resolve_device())
        net.eval()
        return net

    def score(self, model: Any, x: Any) -> float:
        """SPOOF posterior in [0,1]. Checkpoint emits bona-fide logit at
        index 1 (X5): p_spoof = 1 - sigmoid(logits[0, 1])."""
        wav = _preprocess_waveform(x)
        inner = getattr(model, "model", model)  # unwrap ArchModelWrapper
        try:
            dev = next(inner.parameters()).device
            wav = wav.to(dev)
        except (StopIteration, AttributeError):
            pass
        was_training = getattr(inner, "training", False)
        if hasattr(inner, "eval"):
            inner.eval()
        with torch.no_grad():
            logits = inner(wav)
        if was_training and hasattr(inner, "train"):
            inner.train()
        return float(1.0 - F.softmax(logits, dim=-1)[0, 1].item())


def get_arch() -> SpectraAASIST3Spec:
    """Registry hook consumed by ``vishwas.model_archs.get_arch`` (family
    ``'aasist3'``) and by ``vishwas.model_adapters._arch_aware_load``."""
    return SpectraAASIST3Spec()
