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

def _pre_fix_width(seq_len: int, hidden: int) -> int:
    """The budget as it was before the batch term — the reference for the no-op claim below."""
    if seq_len > T._trimul_l1_max_seq():
        return T.TRIANGLE_MULT_CHUNK_SIZE
    gx, gy = T.COMPUTE_GRID_MAIN
    budget = T.TRIANGLE_MULT_L1_CHUNK_BUDGET * gx * gy / (T.COMPUTE_GRID_X_13 * 10)
    c = T.TRIANGLE_MULT_CHUNK_SIZE
    while hidden % (c * 2) == 0 and (c * 2) * seq_len * seq_len <= budget:
        c *= 2
    return c


@pytest.mark.parametrize("grid", [(13, 10), (11, 10), (8, 8), (7, 7)])
@pytest.mark.parametrize("fast", [False, True])
def test_batch_one_is_a_no_op_on_every_grid(monkeypatch, grid, fast):
    """Single-sample folds must resolve to the EXACT width the pre-fix budget picked.

    Every model that passed before this fix folds its trunk at batch 1, so this is what makes
    the change bit-exact for them: not a numerical argument, but the same width, hence the same
    program, hence the same computation. Swept over the grids tt-bio runs on (Blackhole 13x10,
    p150a 11x10, Wormhole 8x8, and a small grid), both precision modes, every hidden width and
    sequence length up to 1568 — 3136 combinations, no mismatch.
    """
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", grid)
    monkeypatch.setattr(T, "_FAST_MODE", fast)
    for hidden in (32, 64, 96, 128, 192, 256, 384, 512):
        for seq in range(32, 1600, 32):
            want = _pre_fix_width(seq, hidden)
            assert T._trimul_chunk_size(seq, hidden, 1) == want, (grid, fast, hidden, seq)
            assert T._trimul_chunk_size(seq, hidden) == want, (grid, fast, hidden, seq)


@pytest.fixture
def clean_memos():
    """The clash memos are module-level process state; every test gets them empty."""
    saved_clash = dict(T._TRIMUL_CHUNK_CLASH)
    saved_dram = set(T._TRIMUL_DRAM_SHAPES)
    T._TRIMUL_CHUNK_CLASH.clear()
    T._TRIMUL_DRAM_SHAPES.clear()
    yield
    T._TRIMUL_CHUNK_CLASH.clear()
    T._TRIMUL_CHUNK_CLASH.update(saved_clash)
    T._TRIMUL_DRAM_SHAPES.clear()
    T._TRIMUL_DRAM_SHAPES.update(saved_dram)


def test_empty_memo_is_a_no_op_on_the_incident_shape(grid, clean_memos):
    """Neutrality: with nothing recorded, the issue-11 shape keeps its 0.6.3 width.

    (140 tokens, hidden 256) is the fold Taylor reported; the budget alone picks 256
    there, and an untouched memo must not move it.
    """
    assert T.COMPUTE_GRID_MAIN == GRID
    assert T._trimul_chunk_size(140, 256, 1) == _pre_fix_width(140, 256) == 256


def test_recorded_clash_clamps_below_it(grid, clean_memos):
    """The retry's whole mechanism: a recorded clash narrows the next pick, floored at 32."""
    T._record_trimul_clash(140, 256, 1, 256)
    assert T._trimul_chunk_size(140, 256, 1) == 128
    T._record_trimul_clash(140, 256, 1, 128)
    assert T._trimul_chunk_size(140, 256, 1) == 64
    T._record_trimul_clash(140, 256, 1, T.TRIANGLE_MULT_CHUNK_SIZE)
    assert T._trimul_chunk_size(140, 256, 1) == T.TRIANGLE_MULT_CHUNK_SIZE


def test_clash_memo_keeps_the_minimum(grid, clean_memos):
    """The narrowest observed clash is the binding one, regardless of record order."""
    T._record_trimul_clash(140, 256, 1, 256)
    T._record_trimul_clash(140, 256, 1, 64)
    T._record_trimul_clash(140, 256, 1, 128)
    assert T._TRIMUL_CHUNK_CLASH[T._trimul_chunk_key(140, 256, 1)] == 64
    assert T._trimul_chunk_size(140, 256, 1) == 32


def test_memo_is_keyed_per_shape_and_grid(monkeypatch, clean_memos):
    """A clash on one grid or shape says nothing about another: no cross-talk."""
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (11, 10))
    T._record_trimul_clash(140, 256, 1, 256)
    assert T._trimul_chunk_size(140, 256, 1) == 128      # same key: clamped
    assert T._trimul_chunk_size(140, 128, 1) == 128      # other hidden: untouched
    assert T._trimul_chunk_size(224, 256, 1) == 64       # other seq: untouched
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", (13, 10))
    assert T._trimul_chunk_size(140, 256, 1) == 256      # other grid: untouched


def test_dram_shapes_force_the_dram_config(grid, clean_memos):
    """The terminal escape: a shape that clashes at the minimum width leaves L1."""
    assert T._triangle_mul_memory_config(140) is T.ttnn.L1_MEMORY_CONFIG
    T._TRIMUL_DRAM_SHAPES.add(140)
    assert T._triangle_mul_memory_config(140) is T.ttnn.DRAM_MEMORY_CONFIG
