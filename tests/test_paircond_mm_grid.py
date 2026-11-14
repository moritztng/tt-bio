"""The diffusion pair-conditioning matmuls never force ``core_grid``.

Forcing ``core_grid`` on those N*N-row projections selects ttnn's 1D in1-multicast matmul, which
deadlocks intermittently: protenix-v1's 4-tile projection hung 17 of 24 folds with it
(perf/pxv1/hang_watcher_capture.txt) and protenix-v2's 8-tile one hung qb2 chip 3 within 50 calls
of perf/spd/nocdeadlock/paircond_repro.py, against 1000/1000 clean without it. The earlier gate
kept the grid above 4 tiles; that is what left protenix-v2 exposed.

Host-only: no device and no checkpoint load.
"""
import pytest

from tt_bio.protenix import paircond_mm_kw


@pytest.mark.parametrize("force,expect", [(None, False), ("1", True)])
def test_core_grid_only_on_request(monkeypatch, force, expect):
    """Ask the real helper, not a copy of its logic."""
    if force is None:
        monkeypatch.delenv("PROTENIX_PAIRCOND_MM_FORCE_GRID", raising=False)
    else:
        monkeypatch.setenv("PROTENIX_PAIRCOND_MM_FORCE_GRID", force)
    assert ("core_grid" in paircond_mm_kw(None, None)) is expect


def test_every_paircond_projection_goes_through_the_helper():
    """No forced core_grid may survive inside _diffusion_pair_cond.

    The first version of the v1 fix patched only `linear_no_bias_z` on the unblocked path and left
    four other projections in the same function forcing the grid, including the SAME projection on
    the blocked path. This asserts on the source so that regression cannot come back quietly.
    """
    import inspect

    from tt_bio.protenix import Protenix

    src = inspect.getsource(Protenix._diffusion_pair_cond)
    assert "core_grid=CORE_GRID_MAIN" not in src, (
        "a pair-cond projection forces core_grid directly instead of going through paircond_mm_kw()")
    n_lin, n_kw = src.count("ttnn.linear("), src.count("paircond_mm_kw(")
    assert n_lin >= 3 and n_kw == n_lin, (
        f"{n_lin} ttnn.linear calls but {n_kw} paircond_mm_kw calls in _diffusion_pair_cond")
