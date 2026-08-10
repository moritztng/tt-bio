"""The trimul L1 chunk-width budget must price the batch dimension.

`_trimul_chunk_size` widens the triangle-multiplication's hidden-channel chunk while the
working set it holds in L1 stays inside `TRIANGLE_MULT_L1_CHUNK_BUDGET`. Every tensor that
loop keeps in L1 is `[batch, chunk, seq, seq]`, so the budget has to see the batch. It did
not, and ESMFold2's confidence head arrives at the trimul with one pair copy per diffusion
sample: at 117 aa the batch-blind budget widened the chunk from 32 to 128, and at
`--diffusion_samples 5` the channel loop's input matmul threw "statically allocated circular
buffers in program 500 clash with L1 buffers". Both esmfold2 and esmfold2-fast failed
release_gate.py on it; the same target at 1 sample folded fine, which is what hid it.

Host-only — no device, no network. The grid is pinned because the budget scales with core
count, so an unpinned test would assert different widths on Wormhole and Blackhole.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

GRID = (13, 10)  # Blackhole p150a, the grid the budget was measured on
SEQ_L1 = 128     # 117 aa padded to a tile multiple: the release-gate target, on the L1 path
HIDDEN = 128     # trimul latent width for ESMFold2 and Protenix-v2


@pytest.fixture
def grid(monkeypatch):
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", GRID)
    monkeypatch.setattr(T, "_FAST_MODE", False)


def _budget() -> float:
    return T.TRIANGLE_MULT_L1_CHUNK_BUDGET * GRID[0] * GRID[1] / (T.COMPUTE_GRID_X_13 * 10)


@pytest.mark.parametrize("batch", [1, 2, 4, 5, 8, 16, 32])
@pytest.mark.parametrize("seq", [32, 64, 128, 224, 320, 352])
def test_working_set_stays_inside_the_budget(grid, batch, seq):
    """The invariant the missing batch term broke: what the loop holds must fit."""
    c = T._trimul_chunk_size(seq, HIDDEN, batch)
    assert batch * c * seq * seq <= _budget() or c == T.TRIANGLE_MULT_CHUNK_SIZE


def test_batch_one_is_unchanged(grid):
    """Single-sample folds keep the width the perf work measured — 32 -> 128 at 117 aa."""
    assert T._trimul_chunk_size(SEQ_L1, HIDDEN, 1) == 128
    assert T._trimul_chunk_size(SEQ_L1, HIDDEN) == 128  # batch defaults to 1


def test_the_failing_case_narrows(grid):
    """release_gate.py's esmfold2 leg: 117 aa at --diffusion_samples 5 must not widen to 128."""
    assert T._trimul_chunk_size(SEQ_L1, HIDDEN, 5) == 64


@pytest.mark.parametrize("seq", [32, 64, 128, 224, 320, 352])
def test_width_never_grows_with_batch(grid, seq):
    """More samples can only ever narrow the chunk, never widen it."""
    widths = [T._trimul_chunk_size(seq, HIDDEN, b) for b in (1, 2, 4, 5, 8, 16, 32)]
    assert widths == sorted(widths, reverse=True)
    assert min(widths) >= T.TRIANGLE_MULT_CHUNK_SIZE


@pytest.mark.parametrize("batch", [1, 5, 32])
def test_dram_path_is_untouched(grid, batch):
    """Above the L1 ceiling the chunks live in DRAM and the width never moved off 32."""
    seq = T._trimul_l1_max_seq() + 32
    assert T._trimul_chunk_size(seq, HIDDEN, batch) == T.TRIANGLE_MULT_CHUNK_SIZE


@pytest.mark.parametrize("batch", [1, 5])
def test_width_always_divides_the_hidden_channels(grid, batch):
    """A width that does not divide `hidden` would drop or duplicate channels."""
    for seq in (32, 64, 128, 224, 320):
        assert HIDDEN % T._trimul_chunk_size(seq, HIDDEN, batch) == 0
