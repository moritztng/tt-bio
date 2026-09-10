"""The pair row tile has to bound its transient on Blackhole too, not only on small grids.

esmfold2 at 1536 tokens died on ONE tensor: OuterProductMean's product, [B, L*32, L*32] bf16,
which is 2048*L^2 bytes -- 4.50 GiB at L=1536. The allocator refused it on both Blackhole
boxes (4831838208 B requested, 603979776 B per bank, largest free block 504102848 B on qb2
card 1 / p300c 2026-09-10, and the same to within 14336 B on qb1 card 0 / p150a). The op
already knows how to walk it in row blocks; `pair_row_tile` was returning 0 (single pass) for
every L because the budget was only ever set for grids smaller than Blackhole's.

The two things this pins down: the sizes that already fold keep the exact single pass they
folded with, and the sizes above them are bounded instead of growing with L^2.
"""
import importlib

import pytest

tenstorrent = pytest.importorskip("tt_bio.tenstorrent")

REFUSED_BYTES = 4831838208          # what the chip said no to, at L=1536
PAIR_TRANSIENT_BYTES = 2048         # (rows*32) x (L*32) bf16 -> 2048 * rows * L


def _bytes(L: int) -> int:
    rows = tenstorrent.pair_row_tile(L)
    return PAIR_TRANSIENT_BYTES * (rows or L) * L


@pytest.fixture(autouse=True)
def blackhole_baseline(monkeypatch):
    """The state after _apply_grid_thresholds declines to retune: no small-grid budget."""
    monkeypatch.setattr(tenstorrent, "SMALL_GRID_PAIR_TILE_AREA", 0)
    monkeypatch.setattr(tenstorrent, "SMALL_GRID_SEQ_TILE", 0)


@pytest.mark.parametrize("L", [128, 256, 512, 768, 1024])
def test_sizes_already_folded_keep_their_single_pass(L):
    # Every size the Blackhole ladder has walked is unchanged, so this cannot regress a
    # shipped fold's arithmetic or its wall clock.
    assert tenstorrent.pair_row_tile(L) == 0


@pytest.mark.parametrize("L", [1056, 1152, 1300, 1312, 1408, 1536])
def test_sizes_above_the_measured_top_are_blocked_and_bounded(L):
    rows = tenstorrent.pair_row_tile(L)
    assert rows and rows < L, f"L={L} still asks for a single pass"
    assert rows % 32 == 0, f"L={L} rows={rows} is not tile-aligned"
    # Bounded, not merely smaller: the transient stays at the measured single-pass top
    # instead of tracking L^2.
    assert _bytes(L) <= PAIR_TRANSIENT_BYTES * tenstorrent.BH_PAIR_TILE_AREA


def test_the_refused_allocation_is_the_one_this_removes():
    # Negative control on the whole change: with no budget the L=1536 transient is exactly
    # the byte count the allocator refused, and with the budget it is under half of it.
    assert PAIR_TRANSIENT_BYTES * 1536 * 1536 == REFUSED_BYTES
    assert _bytes(1536) < REFUSED_BYTES / 2


def test_a_small_grid_budget_still_wins():
    # Wormhole's fitted budget must not be overridden by the Blackhole baseline: it is
    # smaller on purpose, and a part with less L1 has to block sooner.
    tenstorrent.SMALL_GRID_PAIR_TILE_AREA = 65536
    try:
        assert tenstorrent.pair_row_tile(1024) == 64
    finally:
        tenstorrent.SMALL_GRID_PAIR_TILE_AREA = 0


def test_the_budget_is_a_module_constant_not_a_literal_in_the_helper():
    src = importlib.import_module("inspect").getsource(tenstorrent.pair_row_tile)
    assert "BH_PAIR_TILE_AREA" in src and "1024 * 1024" not in src
