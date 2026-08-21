"""Hermetic tests for the model-weights inference adapter registry (D1).

Zero network, no real weights: synthetic numpy arrays, stub model objects,
temp dirs. Covers brief cases a-h:
  a. resolve() -> adapter for each of the 7 env names; unknown -> None
  b. stub .predict(x)->[0.7] -> adapter yields 0.7
  c. stub returning logits [2.5] -> sigmoid-normalised, in (0.5, 1)
  d. stub raising mid-inference -> status 'degraded'/'failed', no escape
  e. missing weight path -> loader None -> caller emits 'unavailable'
  f. mel helper: 44.1 kHz sine >=0.5s -> (rows<=2000, cols==128), finite,
     peak near the tone's mel band
  g. multi-crop aggregation: 3 crops -> MEDIAN (not mean/max)
  h. torch-absent simulation -> loader None, graceful
"""
from __future__ import annotations

import math
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from verisafe import model_adapters as ma
from verisafe.model_adapters import ADAPTERS, Adapter, compute_mel, resolve, run_check
from verisafe.capabilities import deepfake_audio as daudio
from verisafe.capabilities import deepfake_video as dvideo
from verisafe.capabilities import image_facecheck as iface

SEVEN_ENVS = [
    "VERISAFE_EFFORT_WEIGHTS",
    "VERISAFE_DEMAMBA_WEIGHTS",
    "VERISAFE_FAKEMAMBA_WEIGHTS",
    "VERISAFE_AASIST_WEIGHTS",
    "VERISAFE_SSL_AUDIO_WEIGHTS",
    "VERISAFE_HAVIC_WEIGHTS",
    "VERISAFE_IMAGE_FACE_WEIGHTS",
]


# ---------------------------------------------------------------- fixtures --
class StubPredict:
    """Duck-typed model: .predict(x) -> [next score]."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def predict(self, x):
        s = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return [s]


class RaisingModel:
    def predict(self, x):
        raise ValueError("simulated mid-inference failure")


@pytest.fixture
def weights_file(tmp_path):
    p = tmp_path / "stub_weights.bin"
    p.write_bytes(b"\x00\x01stub-not-a-real-weight-file")
    return str(p)


# ------------------------------------------------------------------- a ----
def test_resolve_all_seven_and_unknown():
    families = {
        "VERISAFE_EFFORT_WEIGHTS": "image",
        "VERISAFE_DEMAMBA_WEIGHTS": "video",
        "VERISAFE_FAKEMAMBA_WEIGHTS": "audio",
        "VERISAFE_AASIST_WEIGHTS": "audio",
        "VERISAFE_SSL_AUDIO_WEIGHTS": "audio",
        "VERISAFE_HAVIC_WEIGHTS": "video",
        "VERISAFE_IMAGE_FACE_WEIGHTS": "face",
    }
    assert sorted(ADAPTERS.keys()) == sorted(SEVEN_ENVS)
    for env in SEVEN_ENVS:
        ad = resolve(env)
        assert isinstance(ad, Adapter)
        assert ad.env_name == env
        assert ad.family == families[env]
        callable(ad.preprocess)
    assert resolve("VERISAFE_DOES_NOT_EXIST_WEIGHTS") is None
    assert resolve("") is None


def _adapter_with_loader(env: str, stub) -> Adapter:
    """Non-mutating copy of a registry adapter with a stub loader injected."""
    return replace(resolve(env), _load=lambda p: stub)


# ------------------------------------------------------------------- b ----
def test_predict_list_score_passthrough(weights_file):
    ad = _adapter_with_loader("VERISAFE_EFFORT_WEIGHTS", StubPredict([0.7]))
    status, signals, _notes = run_check(ad, weights_file, np.zeros((224, 224, 3), np.uint8))
    assert status == "ok"
    assert signals["prob_deepfake"] == 0.7
    assert isinstance(ad, Adapter)


# ------------------------------------------------------------------- c ----
def test_logit_sigmoid_normalization(weights_file):
    ad = _adapter_with_loader("VERISAFE_FAKEMAMBA_WEIGHTS", StubPredict([2.5]))  # raw logit
    status, signals, _notes = run_check(ad, weights_file, np.zeros(16000, np.float32))
    assert status == "ok"
    p = signals["prob_deepfake"]
    expected = round(min(1.0, max(0.0, 1.0 / (1.0 + math.exp(-2.5)))), 3)  # ~0.924
    assert 0.5 < p < 1.0
    assert p == expected


def test_auto_extract_scalar_and_array():
    assert ma._auto_extract(0.3) == 0.3
    assert ma._auto_extract(np.array([0.9])) == 0.9
    assert ma._auto_extract(np.array(1.5)) == pytest.approx(1.0 / (1 + math.exp(-1.5)), abs=1e-3)
    assert ma._auto_extract([-0.5]) == pytest.approx(1.0 / (1 + math.exp(0.5)), abs=1e-3)
    assert ma._auto_extract(3.0) == pytest.approx(1.0 / (1 + math.exp(-3.0)), abs=1e-3)
    assert ma._auto_extract([-3.0]) == pytest.approx(1.0 / (1 + math.exp(3.0)), abs=1e-3)
    assert ma._auto_extract(["garbage"]) is None


# ------------------------------------------------------------------- d ----
def test_raising_model_degraded_no_escape(weights_file):
    ad = _adapter_with_loader("VERISAFE_AASIST_WEIGHTS", RaisingModel())
    status, signals, notes = run_check(ad, weights_file, np.zeros(16000, np.float32))
    assert status in ("degraded", "failed")
    assert "ValueError" in signals.get("error_class", "")
    assert "RaisingModel" in notes or "inference error" in notes


# ------------------------------------------------------------------- e ----
def test_missing_path_loader_none_and_unavailable(tmp_path):
    missing = str(tmp_path / "does_not_exist.pth")
    for env in SEVEN_ENVS:
        ad = resolve(env)
        assert ad.load(missing) is None
        status, signals, _notes = run_check(ad, missing, np.zeros(8, np.float32))
        assert status == "unavailable"
        assert signals == {"missing_dependency": "model-weights"}


def test_capability_loaders_return_none_without_env(monkeypatch):
    for env in SEVEN_ENVS:
        monkeypatch.delenv(env, raising=False)
    assert dvideo._load_model("EffortFaceForensics", "VERISAFE_EFFORT_WEIGHTS") is None
    assert daudio._load_weights("VERISAFE_AASIST_WEIGHTS") is None
    assert iface._load_model() is None


def test_video_effort_unavailable_result_verbatim(monkeypatch):
    monkeypatch.delenv("VERISAFE_EFFORT_WEIGHTS", raising=False)
    cap = dvideo.DeepfakeVideoCapability()
    res = cap._effort(types.SimpleNamespace(), [])
    assert len(res) == 1
    r = res[0]
    assert r.name == "effort_face_forensics"
    assert r.cost == "heavy"
    assert r.status == "unavailable"
    assert r.signals == {"missing_dependency": "model-weights"}
    assert "EFFORT weights not provisioned (VERISAFE_EFFORT_WEIGHTS)" in r.notes


def test_audio_ssl_unavailable_result_verbatim(monkeypatch, tmp_path):
    monkeypatch.delenv("VERISAFE_SSL_AUDIO_WEIGHTS", raising=False)
    cap = daudio.DeepfakeAudioCapability()
    res = cap._ssl_detector(types.SimpleNamespace(), tmp_path / "aud.wav")
    assert res[0].name == "ssl_audio_detector"
    assert res[0].status == "unavailable"
    assert res[0].signals == {"missing_dependency": "model-weights"}
    assert "SSL-complement weights not provisioned" in res[0].notes


def test_audio_multicrop_unavailable_result_verbatim(monkeypatch, tmp_path):
    monkeypatch.delenv("VERISAFE_AASIST_WEIGHTS", raising=False)
    cap = daudio.DeepfakeAudioCapability()
    ctx = types.SimpleNamespace(quarantine_root=tmp_path)
    res = cap._multi_crop("aasist", "VERISAFE_AASIST_WEIGHTS", 3, ctx)
    assert res[0].name == "aasist_detector"
    assert res[0].status == "unavailable"
    assert "AASIST weights not provisioned (VERISAFE_AASIST_WEIGHTS); skipped" in res[0].notes


# ------------------------------------------------------------------- f ----
def test_mel_helper_shape_finite_peak_band():
    sr = 44100
    dur_s = 0.6
    t = np.arange(int(sr * dur_s)) / sr
    freq = 1000.0
    wave = (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)
    spec = compute_mel(wave, sr=sr, n_mels=128)
    assert spec.ndim == 2
    assert spec.shape[1] == 128
    assert spec.shape[0] <= 2000
    assert np.all(np.isfinite(spec))
    # peak should sit in the 1 kHz mel band: fraction = log10(1+f/700)/log10(1+(sr/2)/700)
    frac = math.log10(1 + freq / 700.0) / math.log10(1 + (sr / 2) / 700.0)
    expected_row = int(round(frac * 128))
    peak_row = int(np.unravel_index(np.argmax(spec), spec.shape)[1])
    assert abs(peak_row - expected_row) <= 30, \
        f"peak at mel row {peak_row}, expected ~{expected_row} (frac={frac:.2f})"


def test_mel_padded_to_max_rows_on_short_input():
    short = np.zeros(4000, dtype=np.float32)  # < one second
    spec = compute_mel(short, sr=16000)
    assert spec.shape == (2000, 128)
    assert np.all(np.isfinite(spec))


# ------------------------------------------------------------------- g ----
def test_multi_crop_aggregation_is_median(monkeypatch, tmp_path):
    stub = StubPredict([0.2, 0.8, 0.6])
    fake_crops = [tmp_path / f"w{i}.wav" for i in range(3)]
    monkeypatch.setattr(daudio, "_load_weights", lambda env: stub)
    monkeypatch.setattr(
        daudio, "_crop_windows",
        lambda outdir, n=3, win_s=3.0: fake_crops[:n],
    )
    monkeypatch.setenv("VERISAFE_AASIST_WEIGHTS", str(tmp_path / "x"))
    cap = daudio.DeepfakeAudioCapability()
    ctx = types.SimpleNamespace(quarantine_root=tmp_path)
    res = cap._multi_crop("aasist", "VERISAFE_AASIST_WEIGHTS", 3, ctx)
    r = res[0]
    assert r.status == "ok"
    scores = [0.2, 0.8, 0.6]
    median = sorted(scores)[len(scores) // 2]          # 0.6
    mean = sum(scores) / len(scores)                     # 0.5333...
    assert r.signals["prob_deepfake"] == 0.6
    assert r.signals["prob_deepfake"] != round(mean, 3)  # NOT the mean
    assert r.signals["prob_deepfake"] != max(scores)     # NOT the max
    assert r.signals["n_crops_scored"] == 3
    assert r.signals["max_prob"] == 0.8
    del median  # referenced for clarity only


# ------------------------------------------------------------------- h ----
def test_torch_absent_simulation_graceful(weights_file, monkeypatch):
    # Poison sys.modules so ANY `import torch` raises ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    for env in SEVEN_ENVS:
        ad = resolve(env)
        assert ad.load(weights_file) is None
        status, signals, _notes = run_check(ad, weights_file, np.zeros(8, np.float32))
        assert status == "unavailable"
        assert signals == {"missing_dependency": "model-weights"}
    # Legacy capability loaders must also degrade to None, never raise.
    monkeypatch.setenv("VERISAFE_EFFORT_WEIGHTS", weights_file)
    assert dvideo._load_model("EffortFaceForensics", "VERISAFE_EFFORT_WEIGHTS") is None
    monkeypatch.setenv("VERISAFE_AASIST_WEIGHTS", weights_file)
    assert daudio._load_weights("VERISAFE_AASIST_WEIGHTS") is None
    monkeypatch.setenv("VERISAFE_IMAGE_FACE_WEIGHTS", weights_file)
    assert iface._load_model() is None


# ------------------------------------------------------- preprocessor sanity
def test_image_preprocess_chw_normalized():
    ad = resolve("VERISAFE_EFFORT_WEIGHTS")
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    chw = ad.preprocess(img)
    assert chw.shape == (3, 224, 224)
    assert chw.dtype == np.float32
    assert 0.0 <= float(chw.min()) <= 1.0
    assert float(chw.max()) <= 1.0


def test_face_preprocess_112_batch():
    ad = resolve("VERISAFE_IMAGE_FACE_WEIGHTS")
    crop = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    batch = ad.preprocess(crop)
    assert batch.shape == (1, 3, 112, 112)
    assert batch.dtype == np.float32


def test_video_sequence_preprocess_batches_frames():
    ad = resolve("VERISAFE_HAVIC_WEIGHTS")
    frames = [(np.random.rand(240, 320, 3) * 255).astype(np.uint8) for _ in range(5)]
    seq = ad.preprocess(frames)
    assert seq.shape == (5, 3, 224, 224)
    assert seq.dtype == np.float32


# ------------------------------------------------- regression: state-dict --
def test_state_dict_checkpoint_degrades_to_unavailable(weights_file, monkeypatch):
    """A bare state-dict checkpoint (torch.save(model.state_dict())) loads as a
    dict/OrderedDict — no predict/forward/__call__ — so capability loaders must
    return None and emit their normal 'unavailable' result instead of crashing
    mid-scan. Payload built with pickle+numpy so the test holds whether or not
    a working torch is importable."""
    import pickle
    sd = {"conv1.weight": np.zeros((3, 3, 8, 8), dtype=np.float32),
          "head.bias": np.array([0.5], dtype=np.float32)}
    sd_path = Path(weights_file).parent / "state_dict_only.pt"
    with open(sd_path, "wb") as f:
        pickle.dump(sd, f)

    monkeypatch.setenv("VERISAFE_EFFORT_WEIGHTS", str(sd_path))
    m = dvideo._load_model("EffortFaceForensics", "VERISAFE_EFFORT_WEIGHTS")
    assert m is None  # unusable payload -> treated as not provisioned

    # _infer must never raise even if handed the raw dict directly
    ad = resolve("VERISAFE_EFFORT_WEIGHTS")
    frame = np.random.rand(224, 224, 3).astype(np.uint8)
    p = dvideo._infer(ad, sd, frame)
    assert p is None

    # audio + face loaders apply the same gate
    monkeypatch.setenv("VERISAFE_AASIST_WEIGHTS", str(sd_path))
    assert daudio._load_weights("VERISAFE_AASIST_WEIGHTS") is None
    monkeypatch.setenv("VERISAFE_IMAGE_FACE_WEIGHTS", str(sd_path))
    assert iface._load_model() is None


def test_is_usable_model_matrix():
    class HasPredict:
        def predict(self, x):
            return [0.5]

    class HasForward:
        def forward(self, x):
            return x

    assert ma.is_usable_model(HasPredict()) is True
    assert ma.is_usable_model(HasForward()) is True
    assert ma.is_usable_model(lambda x: x) is True
    assert ma.is_usable_model({"a": 1}) is False
    assert ma.is_usable_model(None) is False


# ------------------------------- 1.1 (B0): arch-loading seam -------------
class FakeTinyModel:
    """Torch-free stand-in for a constructed network skeleton."""

    def __init__(self, n_params=2):
        self.state = {"p0": np.zeros(2, dtype=np.float32),
                      "p1": np.ones(1, dtype=np.float32)}

    def load_state_dict(self, sd, strict=True):
        if strict and set(sd) != set(self.state):
            raise RuntimeError("strict mismatch")
        missing = set(self.state) - set(sd)
        bad = {k: v for k, v in sd.items() if k in self.state and len(v) != len(self.state[k])}
        self.missing, self.shape_bad = missing, bad
        for k, v in sd.items():
            if k in self.state and len(v) == len(self.state[k]):
                self.state[k] = np.asarray(v, dtype=np.float32)
        return types.SimpleNamespace(missing=list(missing), unexpected_keys=list(set(sd) - set(self.state)))


def _make_fake_arch(name="aasist", weight_env="VERISAFE_AASIST_WEIGHTS",
                    apply_ok=True, score_val=0.7):
    """A tiny ArchSpec subclass satisfying the 1.1 contract (no real net)."""

    class FakeArch(ma.ArchSpec):
        implemented = apply_ok

        def build(self):
            if not self.implemented:
                raise ma.ArchNotImplementedError(f"{self.name} architecture not vendored yet")
            return FakeTinyModel()

        def apply_state(self, model, sd):
            out = model.load_state_dict(sd, strict=False)
            covered = max(0, len(model.state) - len(out.missing)) / len(model.state)
            return covered >= 0.95 and not out.missing and not getattr(model, "shape_bad", {})

        def score(self, model, x):
            return score_val

    return FakeArch()


def test_archspec_contract_defaults():
    """Base class documents name/weight_env slots and typed unimplemented error."""
    spec = ma.ArchSpec(name="demo", weight_env="VERISAFE_DEMO_WEIGHTS")
    assert spec.name == "demo"
    assert spec.weight_env == "VERISAFE_DEMO_WEIGHTS"
    with pytest.raises(ma.ArchNotImplementedError):
        spec.build()
    assert isinstance(ma.ArchNotImplementedError, type)
    assert issubclass(ma.ArchNotImplementedError, Exception)


def test_get_arch_lazy_registry(monkeypatch):
    from verisafe.model_archs import get_arch

    # effort / havic are still honest stubs (tasks 1.3 / 1.4 pending):
    # resolvable specs whose build() MUST raise — never silently work.
    for fam, env in (("effort", "VERISAFE_EFFORT_WEIGHTS"),
                     ("havic", "VERISAFE_HAVIC_WEIGHTS")):
        got = get_arch(fam)
        assert isinstance(got, ma.ArchSpec)
        assert got.name == fam and got.weight_env == env
        assert getattr(got, "implemented", False) is False
        with pytest.raises(ma.ArchNotImplementedError):
            got.build()

    # aasist was vendored in task 1.2. Under REAL torch the spec resolves with
    # implemented=True; under the hermetic stub-torch tree the module import
    # fails (no torch.nn) and the registry honestly degrades to None.
    try:
        import torch.nn  # noqa: F401
        _real_torch = True
    except Exception:
        _real_torch = False
    aasist = get_arch("aasist")
    if _real_torch:
        assert isinstance(aasist, ma.ArchSpec)
        assert aasist.name == "aasist" and aasist.weight_env == "VERISAFE_AASIST_WEIGHTS"
        assert aasist.implemented is True
        # Vendored build() overrides the base stub. (Not invoked here: building
        # the 315M-param WavLM-Large front-end costs ~15 s on this CPU.)
        assert aasist.build.__func__ is not ma.ArchSpec.build
    else:
        assert aasist is None  # importable-but-broken -> honest None

    assert get_arch("unknown_family_xyz") is None


def test_env_unset_loader_returns_none(tmp_path, monkeypatch):
    """(a) env var unset -> None: seam short-circuits before any load (behavior unchanged)."""
    p = tmp_path / "x.pth"
    p.write_bytes(b"stub")
    for e in ("VERISAFE_AASIST_WEIGHTS", "VERISAFE_EFFORT_WEIGHTS", "VERISAFE_HAVIC_WEIGHTS"):
        monkeypatch.delenv(e, raising=False)
    assert ma._arch_aware_load(str(p), "aasist", env_name="VERISAFE_AASIST_WEIGHTS") is None
    assert ma._arch_aware_load(str(p), "effort", env_name="VERISAFE_EFFORT_WEIGHTS") is None
    assert ma._arch_aware_load(str(p), "havic", env_name="VERISAFE_HAVIC_WEIGHTS") is None


def test_arch_ready_wrapper_returns_callable_score(weights_file, monkeypatch):
    """(b) env set + arch importable AND apply_state True -> ready callable object.

    Mirrors what existing passing tests assert: run_check yields status 'ok'
    with signals['prob_deepfake'] == the calibrated score.
    """
    monkeypatch.setenv("VERISAFE_AASIST_WEIGHTS", weights_file)
    fake_sd = {"p0": np.zeros(2, dtype=np.float32), "p1": np.ones(1, dtype=np.float32)}
    loader = lambda path: ma._arch_aware_load(
        path, "aasist", env_name="VERISAFE_AASIST_WEIGHTS",
        raw_load=lambda p: fake_sd,
        arch=_make_fake_arch(apply_ok=True, score_val=0.7),
    )
    ad = replace(resolve("VERISAFE_AASIST_WEIGHTS"), _load=loader)
    m = ad.load(weights_file)
    assert m is not None
    # READY callable: duck-typing via .score AND .predict so _call_model works
    assert callable(m.score)
    assert hasattr(m, "predict") or callable(m)
    assert 0.0 <= float(m.score(np.zeros((8, 128)))) <= 1.0
    status, signals, notes = run_check(ad, weights_file, np.zeros(16000, np.float32))
    assert status == "ok"
    assert signals["prob_deepfake"] == 0.7
    assert "model inference succeeded" in notes


def test_arch_unavailable_reason_and_no_half_load(weights_file, monkeypatch):
    """(c) env set but arch unavailable (None) OR apply_state False -> None + reason."""
    # (c-i) get_arch returns None
    monkeypatch.setenv("VERISAFE_EFFORT_WEIGHTS", weights_file)
    loader_none = lambda path: ma._arch_aware_load(
        path, "effort", env_name="VERISAFE_EFFORT_WEIGHTS",
        raw_load=lambda p: {"k": 1},
        arch=None,
    )
    ad = replace(resolve("VERISAFE_EFFORT_WEIGHTS"), _load=loader_none)
    assert ad.load(weights_file) is None
    assert ad.last_reason == "weight file loaded but architecture unavailable"

    # (c-ii) arch present but apply_state False (coverage below threshold)
    loader_bad = lambda path: ma._arch_aware_load(
        path, "effort", env_name="VERISAFE_EFFORT_WEIGHTS",
        raw_load=lambda p: {"unrelated_key_only": np.zeros(3)},
        arch=_make_fake_arch(name="effort", apply_ok=True, score_val=0.1),
    )
    ad2 = replace(resolve("VERISAFE_EFFORT_WEIGHTS"), _load=loader_bad)
    assert ad2.load(weights_file) is None
    assert ad2.last_reason == "weight file loaded but architecture unavailable"

    # (c-iii) arch not yet implemented (stub) -> same honest reason, never half-load
    monkeypatch.delenv("VERISAFE_HAVIC_WEIGHTS", raising=False)
    monkeypatch.setenv("VERISAFE_HAVIC_WEIGHTS", weights_file)
    loader_stub = lambda path: ma._arch_aware_load(
        path, "havic", env_name="VERISAFE_HAVIC_WEIGHTS",
        raw_load=lambda p: {"k": 1},
    )
    ad3 = replace(resolve("VERISAFE_HAVIC_WEIGHTS"), _load=loader_stub)
    assert ad3.load(weights_file) is None
    assert ad3.last_reason == "weight file loaded but architecture unavailable"


def test_is_usable_model_wrapper_vs_raw_statedict(weights_file, monkeypatch):
    """(iv) is_usable_model(True) on the arch wrapper vs (False) on raw state-dict."""
    monkeypatch.setenv("VERISAFE_AASIST_WEIGHTS", weights_file)
    fake_sd = {"p0": np.zeros(2), "p1": np.ones(1)}
    m = ma._arch_aware_load(
        weights_file, "aasist", env_name="VERISAFE_AASIST_WEIGHTS",
        raw_load=lambda p: fake_sd,
        arch=_make_fake_arch(apply_ok=True, score_val=0.3),
    )
    assert ma.is_usable_model(m) is True
    assert ma.is_usable_model(fake_sd) is False
