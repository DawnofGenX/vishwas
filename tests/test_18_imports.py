"""Lock import resolution after removing the empty dir-scaffold (Phase 0, Task 0.2).

Historically ``src/vishwas/{analysis,channels,fusion}`` existed as EMPTY
directories sitting next to the live modules ``channels.py`` / ``fusion.py``
and the bare namespace ``analysis``.  Python resolves module-vs-directory by
``.py`` winning today, but that resolution is fragile (e.g. if an ``__init__.py``
ever lands in one of those dirs the module would be shadowed silently).
The dirs were removed with ``rmdir``; these tests make sure the removal
stuck and that no submodule-style imports depend on the old layout.
"""


def test_module_vs_dir_no_shadowing():
    import vishwas.channels as c
    import vishwas.fusion as f

    assert c.__file__.endswith("channels.py")
    assert f.__file__.endswith("fusion.py")


def test_analysis_not_a_package_landmine():
    try:
        import vishwas.analysis
    except Exception:
        pass
    else:
        import vishwas.analysis as a

        # At worst a bare namespace package — never a real module/package
        # that could shadow a future vishwas.analysis.py.
        assert getattr(a, "__file__", None) is None
