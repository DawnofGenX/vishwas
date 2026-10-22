"""THE genimage VETO TABLE MUST BE RE-DERIVABLE, AND IT MUST NOT DRIFT.

2026-10-22. The `_image_clean` comment in `src/vishwas/fusion.py` carried
a table of genimage veto thresholds that justified the SHIPPED threshold
(0.9855) and the SHIPPED shape (veto-only). It cited
`tools/eval_image_arbitration.py` as its source — but that tool sweeps NO
threshold and implements NO "clear" shape. **The table was ASSERTED, not
reproducible.**

`tools/sweep_genimage_veto.py` now re-derives it by WRAPPING the real
shipped `_image_clean`. The VETO-ONLY column reproduces exactly; the
VETO+CLEAR column did not (it claimed fakeTRUST 5/30 at 0.9855 where the
measurement gives 3/30, and it named three "newly let through" images when
only one is).

WHAT THIS PINS
--------------
  1. the sweep tool EXISTS and is the tool the comment cites;
  2. the VETO-ONLY column in the shipped comment matches the MEASURED
     values (so the number in the comment cannot drift away from reality
     without this test failing);
  3. the shipped threshold is the one the measurement supports;
  4. the shipped shape is veto-only, NOT veto+clear — the clear shape is
     rejected, and the reason (it lets a high-NYUAD AI image through) is
     pinned to the ONE image that is actually affected.
"""
from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
FUSION = REPO / "src" / "vishwas" / "fusion.py"
SWEEP = REPO / "tools" / "sweep_genimage_veto.py"


def _src(path: pathlib.Path) -> str:
    if not path.exists():
        import pytest
        pytest.skip(f"{path.name} is not present")
    return path.read_text()



def _code_only(src: str) -> str:
    """The source with docstrings and `#` comments stripped.

    Needed because the tool's own docstring LEGITIMATELY mentions the
    rule it deliberately does not implement — the check must be on CODE.

    Uses `ast` to find every docstring node and blank its lines out.
    A naive line-based stripper gets this wrong: a docstring's closing
    quotes often sit on a line of their OWN, and a stripper that treats
    that line as "still inside the docstring" swallows the ENTIRE rest of
    the file. Then the check passes VACUOUSLY — which is exactly what
    happened the first time this test was written.
    """
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    lines = src.split("\n")
    doc_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                doc_nodes.append(body[0])
    for node in doc_nodes:
        for ln in range(node.lineno - 1, (node.end_lineno or node.lineno)):
            lines[ln] = ""
    return "\n".join(lines)


def test_the_sweep_tool_exists_and_is_what_the_comment_cites():
    """The table's source tool must exist and the comment must cite it."""
    src = _src(FUSION)
    assert SWEEP.exists(), (
        "tools/sweep_genimage_veto.py is missing — the genimage veto table "
        "in the _image_clean comment is not re-derivable")
    assert "tools/sweep_genimage_veto.py" in src, (
        "the _image_clean comment does not cite the sweep tool that "
        "re-derives its table")
    # and it must no longer cite the tool that does NOT sweep
    assert "tools/eval_image_arbitration.py). The" not in src, (
        "the genimage veto table still cites tools/eval_image_arbitration.py, "
        "which sweeps no threshold and implements no 'clear' shape")


def test_the_sweep_tool_wraps_the_real_policy_rather_than_reimplementing_it():
    """A re-implemented base policy gave realFA 11/30 instead of 12/30.

    ⇒ NEVER RE-IMPLEMENT A POLICY TO MEASURE IT; WRAP IT.
    """
    src = _src(SWEEP)
    assert "real_clean" in src, (
        "the sweep tool does not capture the real shipped _image_clean")
    assert re.search(r"real_clean\s*=\s*F\._image_clean", src), (
        "the sweep tool does not capture the REAL shipped _image_clean "
        "before monkey-patching it")
    assert "F._image_clean = original" in src, (
        "the sweep tool does not restore the real _image_clean afterwards")
    # It must not re-implement the SPAI rule. The check is on CODE, not
    # prose — the tool's docstring legitimately MENTIONS the rule it
    # deliberately does not implement.
    code = _code_only(src)
    assert "eff <= 0.45" not in code, (
        "the sweep tool re-implements the SPAI rule instead of wrapping the "
        "real policy — that is what produced the 11/30 discrepancy")
    assert "0.45" not in code, (
        "the sweep tool contains a hardcoded SPAI threshold, so it is "
        "re-implementing the policy rather than wrapping the real one")


def test_the_shipped_threshold_is_the_measured_one():
    """0.9855 is the threshold the measurement supports."""
    src = _src(FUSION)
    m = re.search(r"_GENIMAGE_VETO_T\s*=\s*([0-9.]+)", src)
    assert m, "_GENIMAGE_VETO_T is not defined"
    assert float(m.group(1)) == 0.9855, (
        f"the shipped genimage veto threshold is {m.group(1)}, but the "
        "measured sweep supports 0.9855 (fakeTRUST 2/30, realFALSEALARM "
        "12/30) — re-derive it with tools/sweep_genimage_veto.py before "
        "changing it")


def test_the_shipped_shape_is_veto_only_and_the_clear_shape_is_rejected():
    """The clear shape must stay rejected, with the RIGHT reason.

    The old comment named three images as "newly let through" when only
    ONE is: a low-SPAI image. gigagan_07 reads SPAI 1.0000 and
    stable_diffusion_3_03 reads SPAI 0.5967, so SPAI already denies them.
    """
    src = _src(FUSION)
    i = src.find("THE THRESHOLD (0.9855)")
    assert i != -1, "the genimage veto table is gone from the comment"
    seg = src[i:i + 6000]
    assert "VETO ONLY" in seg, (
        "the comment no longer states that genimage is a VETO ONLY signal")
    # the rejection must name the ONE affected image
    assert "midjourney_v6.1_04" in seg, (
        "the comment must name the one image the clear shape actually lets "
        "through (midjourney_v6.1_04)")
    # And it must NOT claim MORE THAN ONE image is newly let through. The
    # old comment said "the three AI images it newly let through"; only
    # midjourney_v6.1_04 is. Match on the COUNT claim, not one exact
    # phrasing, so the check cannot be defeated by re-wording.
    assert not re.search(
        r"(three|3)\s+AI\s+images?\s+it\s+newly\s+lets?\s+through", seg), (
        "the comment still claims THREE images are newly let through by the "
        "clear shape; only midjourney_v6.1_04 is (the other two read SPAI "
        "1.0000 and 0.5967, so SPAI already denies them)")
    # and the count claim must be exactly one. The comment wraps across
    # lines, so match on WHITESPACE, not literal spaces.
    m = re.search(r"the\s+(\w+)\s+AI\s+images?\s+it\s+newly\s+lets?\s+"
                  r"through", seg)
    assert m, ("the comment does not state how many images the clear shape "
               "newly lets through")
    assert m.group(1).lower() in ("one", "1"), (
        f"the comment claims {m.group(1)} images are newly let through by "
        "the clear shape; only midjourney_v6.1_04 is (the other two read "
        "SPAI 1.0000 and 0.5967, so SPAI already denies them)")
    # The images that CANNOT be newly let through must not be named as
    # AFFECTED. The comment may legitimately MENTION them while explaining
    # why they are not affected, so the check is on the phrase that names
    # them as newly-let-through, not on the bare image name.
    for img in ("gigagan_07", "stable_diffusion_3_03"):
        bad = re.search(
            rf"newly\s+lets?\s+through\s*\([^)]*{img}", seg)
        assert not bad, (
            f"the comment names {img} as newly let through by the clear "
            "shape, but its SPAI reading already denies it")


def test_the_veto_only_column_in_the_comment_matches_the_measurement():
    """The shipped comment's veto-only column must equal the measured one.

    Measured (tools/sweep_genimage_veto.py, real FusionEngine, 60 images):
        0.9000 -> fakeTRUST  0/30  realFA 14/30
        0.9300 -> fakeTRUST  1/30  realFA 13/30
        0.9400 -> fakeTRUST  2/30  realFA 13/30
        0.9500 -> fakeTRUST  2/30  realFA 12/30
        0.9700 -> fakeTRUST  2/30  realFA 12/30
        0.9855 -> fakeTRUST  2/30  realFA 12/30
        0.9950 -> fakeTRUST  2/30  realFA 12/30
    """
    src = _src(FUSION)
    i = src.find("THE THRESHOLD (0.9855)")
    assert i != -1, "the genimage veto table is gone from the comment"
    seg = src[i:i + 6000]
    expected = {
        "0.9000": (0, 14), "0.9300": (1, 13), "0.9400": (2, 13),
        "0.9500": (2, 12), "0.9700": (2, 12), "0.9855": (2, 12),
        "0.9950": (2, 12),
    }
    for th, (ft, fa) in expected.items():
        pat = re.compile(
            rf"{re.escape(th)}\s+{ft}/30\s+{fa}/30\s+"
            rf"\d+/30\s+\d+/30")
        assert pat.search(seg), (
            f"the veto-only row for threshold {th} in the comment does not "
            f"match the measured ({ft}/30 fakeTRUST, {fa}/30 realFA). "
            "Re-derive with tools/sweep_genimage_veto.py.")
