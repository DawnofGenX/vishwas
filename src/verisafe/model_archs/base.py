"""Architecture-spec contract for learned-model families (roadmap Phase 1, B0).

Each gated family (aasist / effort / havic) lives in its own sibling module in
this package and exposes exactly ONE ``ArchSpec`` subclass.  The adapter
registry (:func:`verisafe.model_adapters._arch_aware_load`) consults
:func:`verisafe.model_archs.get_arch` LAZILY at load time.

Honesty rule (no half-loads, ever):
  * ``build()`` constructs the network skeleton.  Until a concrete family is
    vendored (tasks 1.2/1.3/1.4) it raises :class:`ArchNotImplementedError`;
    the adapter converts that to ``(None, reason)`` — it never returns a
    partially-populated model.
  * ``apply_state()`` performs a ``strict=False`` load but compares key-set
    coverage: if >5% of expected keys are missing OR shape-mismatched it
    returns False and records which keys failed (``last_apply``).
  * ``score()`` returns a calibrated posterior in [0, 1] (softmax probability
    of the positive/deepfake class, or a documented monotone mapping).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ArchNotImplementedError(RuntimeError):
    """Typed internal error: family arch module is present but its concrete
    network (``build()`` / ``score()``) is not yet implemented.  Callers must
    convert this to ``(None, reason)`` — NEVER half-load."""


class ArchSpec:
    """Contract shared by all gated-model architectures.

    Subclasses must set the class attributes ``name`` (e.g. ``"aasist"``) and
    ``weight_env`` (e.g. ``"VERISAFE_AASIST_WEIGHTS"``).  Instances may also be
    constructed positionally: ``ArchSpec(name=..., weight_env=...)``.
    """

    name: str = ""
    weight_env: str = ""

    def __init__(self, name: Optional[str] = None, weight_env: Optional[str] = None):
        if name:
            self.name = name
        if weight_env:
            self.weight_env = weight_env

    # -- contract -------------------------------------------------------------
    def build(self) -> Any:
        """Construct the network skeleton (no weights).

        Base behaviour: NOT YET IMPLEMENTED — raises
        :class:`ArchNotImplementedError`.  Concrete families override this
        (tasks 1.2/1.3/1.4); stubs deliberately inherit it so a stub can
        never silently pretend to work.
        """
        raise ArchNotImplementedError(
            f"{self.name or '<unnamed>'} architecture not vendored yet"
        )

    def apply_state(self, model: Any, sd: Dict[str, Any]) -> bool:
        """Load *sd* into *model* with ``strict=False`` and verify key coverage.

        Returns False (and records the failing keys in ``self.last_apply``)
        when the payload is not a non-empty dict, the model exposes no
        ``load_state_dict``, the load raises (torch surfaces size mismatches
        even under strict=False), or >5% of expected keys are missing or
        shape-mismatched.  Never leaves a half-loaded model in a usable state:
        callers check the return value and discard the model on False.
        """
        if not isinstance(sd, dict) or not sd:
            self.last_apply = {"ok": False, "error": "state-dict payload missing or empty"}
            return False
        load_fn = getattr(model, "load_state_dict", None)
        if load_fn is None:
            self.last_apply = {"ok": False, "error": "model has no load_state_dict"}
            return False
        try:
            out = load_fn(sd, strict=False)
        except Exception as exc:  # includes torch size-mismatch RuntimeError
            self.last_apply = {
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            }
            return False
        missing: List[str] = list(getattr(out, "missing_keys", None) or [])
        unexpected: List[str] = list(getattr(out, "unexpected_keys", None) or [])
        # Expected key universe ≈ what the model asked for (given + missing).
        denom = max(1, len(missing) + len(unexpected))
        frac_missing = len(missing) / denom
        frac_unexpected = len(unexpected) / denom
        ok = frac_missing <= 0.05 and frac_unexpected <= 0.05
        self.last_apply = {
            "ok": ok,
            "frac_missing": round(frac_missing, 3),
            "frac_unexpected": round(frac_unexpected, 3),
            "missing": missing[:25],
            "unexpected": unexpected[:25],
        }
        return ok

    def score(self, model: Any, x: Any) -> float:
        """Calibrated posterior in [0, 1]: softmax probability of the positive
        (deepfake/spoof) class, or a documented monotone mapping of logits.

        Base behaviour: NOT YET IMPLEMENTED (same honesty rule as build()).
        """
        raise ArchNotImplementedError(
            f"{self.name or '<unnamed>'} scoring not vendored yet"
        )
