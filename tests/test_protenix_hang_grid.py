"""protenix-v2 folds of 500-507 tokens must not run on a 13x10 grid (GitHub issue #9).

In that window the fp32 pair-cond readback never returns: the host spins in ttnn.to_torch
while the chip stays alive, so the fold hangs instead of failing. 11x10 completes every
time. The window is narrow, 496 and 512 are clean, and 11x10 has no measured advantage
anywhere else, so the workaround has to stay narrow too: every other size keeps all 130
cores. These are the properties that pin that down without a device.
"""
import contextlib

import pytest
import torch

import tt_bio.protenix as P
import tt_bio.tenstorrent as T

G13 = (T.COMPUTE_GRID_X_13, T.COMPUTE_GRID_Y)
G11 = (T.COMPUTE_GRID_X_11, T.COMPUTE_GRID_Y)


@pytest.fixture
def on_grid(monkeypatch):
    """Present a given active compute grid, and put the module back after."""
    @contextlib.contextmanager
    def _use(grid):
        old_c, old_g = T.CORE_GRID_MAIN, T.COMPUTE_GRID_MAIN
        T.CORE_GRID_MAIN = __import__("ttnn").CoreGrid(y=grid[1], x=grid[0])
        T.COMPUTE_GRID_MAIN = grid
        try:
            yield
        finally:
            T.CORE_GRID_MAIN, T.COMPUTE_GRID_MAIN = old_c, old_g
    monkeypatch.delenv("TT_BIO_FORCE_GRID", raising=False)
    return _use


def feats(n_tokens, atoms_per_token=4):
    return {"atom_to_token_idx": torch.arange(n_tokens).repeat_interleave(atoms_per_token)}


def grid_inside(n_tokens):
    """The grid a fold of n_tokens actually runs on."""
    with P._fold_grid(feats(n_tokens)):
        return T.COMPUTE_GRID_MAIN


@pytest.mark.parametrize("n_tokens", [500, 501, 505, 506, 507])
def test_window_folds_drop_to_11x10(on_grid, n_tokens):
    with on_grid(G13):
        assert grid_inside(n_tokens) == G11
        assert T.COMPUTE_GRID_MAIN == G13, "the grid must be restored after the fold"


@pytest.mark.parametrize("n_tokens", [128, 496, 499, 508, 512, 640])
def test_every_other_size_keeps_the_full_grid(on_grid, n_tokens):
    with on_grid(G13):
        assert grid_inside(n_tokens) == G13


def test_the_window_is_exactly_the_measured_one():
    assert P.HANG_GRID_TOKEN_WINDOW == (500, 507)


def test_an_11x10_card_is_untouched(on_grid):
    with on_grid(G11):
        assert grid_inside(506) == G11


def test_an_explicit_force_grid_wins(on_grid, monkeypatch):
    # TT_BIO_FORCE_GRID is how issue #9 was diagnosed in the first place; a pin set by
    # hand must survive the fold, or a bisect against the workaround cannot be run.
    monkeypatch.setenv("TT_BIO_FORCE_GRID", "13,10")
    with on_grid(G13):
        assert grid_inside(506) == G13


def test_compute_grid_restores_after_a_failed_fold(on_grid):
    with on_grid(G13):
        with pytest.raises(RuntimeError):
            with T.compute_grid(*G11):
                assert T.COMPUTE_GRID_MAIN == G11
                raise RuntimeError("fold died")
        assert T.COMPUTE_GRID_MAIN == G13


def test_compute_grid_clears_the_program_config_caches(on_grid):
    # The caches are keyed by shape, not by grid, so a stale entry would hand an 11x10
    # fold a 13x10 program config -- the failure mode this workaround exists to avoid.
    with on_grid(G13):
        T._sdpa_program_config.cache_clear()
        T._sdpa_program_config(128, 128)
        assert T._sdpa_program_config.cache_info().currsize == 1
        with T.compute_grid(*G11):
            assert T._sdpa_program_config.cache_info().currsize == 0
            cfg = T._sdpa_program_config(128, 128)
            got = cfg.compute_with_storage_grid_size
            assert (got.x, got.y) == G11
        assert T._sdpa_program_config.cache_info().currsize == 0


def test_the_release_gate_folds_a_rung_inside_the_window():
    """A hang class that closes has to stay closed: the size-ladder arm folds 506 tokens
    on every release, so the workaround going away flips that rung's lever census."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "release_gate.py"
    spec = importlib.util.spec_from_file_location("release_gate_for_hang_grid", path)
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    lo, hi = P.HANG_GRID_TOKEN_WINDOW
    rungs = rg._size_ladder_rungs("protenix-v2", rg.SIZE_LADDER_RUNGS)
    assert [r for r in rungs if lo <= r <= hi], \
        f"no size-ladder rung inside the issue-#9 hang window {lo}-{hi}: {rungs}"
    # and it is protenix-v2's alone -- no other model pays for a protenix defect.
    assert not [r for r in rg._size_ladder_rungs("boltz2", rg.SIZE_LADDER_RUNGS)
                if lo <= r <= hi]
