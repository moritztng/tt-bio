"""The lever census must not latch its wrap flag on a half-imported module.

`tt_bio.tenstorrent` imports ttnn on line 6 and defines AdaLN on line 9619, so for the
duration of its own body it sits in sys.modules with ttnn present and AdaLN absent. The
census hook thread polls every 3 s; a poll landing in that window used to latch
`_census_wrapped`, raise AttributeError, have it swallowed by `tick()`, and leave all seven
wrap levers counting zero for the whole fold -- which the ladder comparator reads as seven
levers going dark, not as seven levers nobody counted.
"""
import sys
import types
import importlib.util
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture()
def lc():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import lever_census
        yield lever_census
    finally:
        sys.path.remove(str(SCRIPTS))


def _half_imported(initializing):
    """A stand-in for tt_bio.tenstorrent mid-body: ttnn done, AdaLN not defined yet."""
    m = types.ModuleType("tt_bio.tenstorrent")
    m.__spec__ = importlib.util.spec_from_loader("tt_bio.tenstorrent", loader=None)
    m.__spec__._initializing = initializing
    return m


def test_install_wraps_does_not_latch_on_a_partially_imported_module(lc, monkeypatch):
    half = _half_imported(True)
    monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", half)
    monkeypatch.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
    monkeypatch.setattr(lc, "WRAPS_INSTALLED", False)

    lc._install_wraps()

    assert not getattr(half, "_census_wrapped", False), (
        "the wrap flag latched while the module body was still executing, so the next tick "
        "returns at the guard and no wrapper is ever installed")
    assert not lc.WRAPS_INSTALLED


def test_an_uninstalled_wrap_lever_reports_unmeasured_not_dark(lc, monkeypatch):
    """0/0 is 'the fold never reached it'. Nothing counting must not borrow that reading."""
    monkeypatch.setitem(sys.modules, "tt_bio.tenstorrent", _half_imported(False))
    monkeypatch.setattr(lc, "WRAPS_INSTALLED", False)

    rows = lc._snapshot_process()
    wrap = [f for f, _m, _a, _c, how in lc.LEVERS
            if how == "wrap" and _m == "tt_bio.tenstorrent"]
    assert wrap, "no wrap-counted lever on tt_bio.tenstorrent to check"
    for flag in wrap:
        r = rows[flag]
        assert r["served"] is None and r["declined"] is None, (
            f"{flag} reports {r['served']}/{r['declined']} with no wrapper installed; "
            "release_gate turns 0/0 into frac 0.0, which is the reading for a dark lever")
