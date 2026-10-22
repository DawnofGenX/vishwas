"""THE IMAGE CHANNEL MUST NOT RELOAD THE CHECKPOINT FOR EVERY IMAGE.

2026-10-22. The aasist3 audio lane had exactly this defect — the
checkpoint was reloaded once per CROP, which cost 3 x 63 s and pushed the
audio lane past its budget (PR #28 fixed it).

**THE IMAGE LANE HAS THE IDENTICAL DEFECT, UNFIXED.** Both image gates
build their model and load their checkpoint on EVERY call:

  * `image_facecheck._genimage_prob()` — `spec.build()` + `torch.load()`
    INSIDE the function, so it runs once per image;
  * the SPAI path — `_load_model()` inside the per-image path.

Measured cost: a 60-image corpus pass spends minutes reloading the same
checkpoint. On the SERVING path that is one image at a time, so it is
not a per-job cost — but it is a per-image cost, and it is the same
shape of defect that broke the audio lane.

WHAT THIS PINS
--------------
  1. the genimage model is built ONCE and reused, not per call;
  2. the checkpoint file is read ONCE, not per call;
  3. the cache is keyed on the checkpoint PATH, so switching
     checkpoints (a different env var) does not serve a stale model;
  4. a failed load is NOT cached, so a checkpoint that appears later is
     picked up rather than being permanently poisoned;
  5. the cache is bounded (it must not grow without limit).
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
IMG = REPO / "src" / "vishwas" / "capabilities" / "image_facecheck.py"


def _src() -> str:
    if not IMG.exists():
        import pytest
        pytest.skip("the image capability is not present")
    return IMG.read_text()


def test_the_genimage_model_is_built_once_not_per_call():
    """`spec.build()` must not be inside the per-image function."""
    src = _src()
    m = re.search(r"def _genimage_prob\(.*?\n(?=\ndef |\nclass )", src, re.S)
    assert m, "the genimage scorer is gone"
    body = m.group(0)
    assert "spec.build()" not in body, (
        "_genimage_prob builds the model on EVERY call, so the checkpoint is "
        "reloaded for every image — the same defect PR #28 fixed in the "
        "audio lane")
    assert "torch.load(" not in body, (
        "_genimage_prob reads the checkpoint file on EVERY call")


def test_a_cache_exists_and_is_keyed_on_the_checkpoint_path():
    """A cache keyed on the PATH, so a different checkpoint is not served
    from a stale entry."""
    src = _src()
    assert re.search(r"_GENIMAGE_MODEL_CACHE|_genimage_model_cache|"
                     r"lru_cache|_MODEL_CACHE", src), (
        "the image capability has no model cache, so every image rebuilds "
        "and reloads the checkpoint")
    # The key must DERIVE from the path. The shape is
    # `key = str(path)` then `cache[key]`, so the check is that the path
    # feeds the key — not that the literal `cache[path]` appears.
    assert re.search(r"key\s*=\s*str\(path\)", src), (
        "the model cache is not keyed on the checkpoint path, so a "
        "different checkpoint would serve a stale model")
    assert re.search(r"_GENIMAGE_MODEL_CACHE\[key\]\s*=", src), (
        "the model cache is not written under the path-derived key")


def test_a_failed_load_is_not_cached():
    """A load that fails must not poison the cache — the checkpoint may
    appear later (that is how a deployment provisions weights), and a
    cached failure would be permanent."""
    src = _src()
    assert re.search(r"cache\.pop\(|del cache\[|except.*cache", src, re.S), (
        "a FAILED load appears to be cached, so a checkpoint that is "
        "provisioned later would never be picked up")


def test_the_cache_is_bounded():
    """It must not grow without limit."""
    src = _src()
    assert re.search(r"if len\(.*cache.*\)\s*>|MAX_CACHE|_CACHE_MAX", src), (
        "the model cache is unbounded — it would grow one entry per "
        "distinct checkpoint path for the life of the process")
