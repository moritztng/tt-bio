"""A refused L1 score block loses one row; it does not retire L1 for the whole shape class.

Triangle attention's fp32-softmax tail keeps its score block L1-resident when the block fits a
per-core byte budget. The block is sized against the idle device, so at large token counts the
sharded softmax can still refuse its circular buffers around a block that allocated legally, and
that refusal is the only signal there is.

Retiring the shape class on it was RF3's whole 768 -> 1024 aa scaling cliff: 434 of 435 triangle
attentions per recycle took the interleaved DRAM tail at 1024 aa where 768 aa runs every block
sharded, and the trunk's local log-log exponent went 2.77 -> 3.63 across that one interval. A 2-row
block is accepted at the same shape and is worth 1.264x on the trunk (52.468 -> 41.508 s/recycle,
perf/rf3/verify_backoff.sh on qb2 card 2), bit-identical to the interleaved tail on both trunk
outputs with an A/A control at 0.0 (perf/rf3/fp32_l1_backoff_bits.py).

Host-only: no device, no fold. The refusal itself needs hardware and lives in that harness.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

N_HEADS = 4
CORES = 64          # _FP32_SOFTMAX_L1_GRID is 8x8


def shape(tokens: int) -> tuple[int, int]:
    """(per_row, height_per_row) for a [S, n_heads, S, S] fp32 score tensor."""
    height_per_row = N_HEADS * tokens
    return height_per_row * tokens * 4, height_per_row


@pytest.fixture(autouse=True)
def clean_caps():
    saved = dict(T._FP32_SOFTMAX_L1_ROW_CAP)
    T._FP32_SOFTMAX_L1_ROW_CAP.clear()
    yield
    T._FP32_SOFTMAX_L1_ROW_CAP.clear()
    T._FP32_SOFTMAX_L1_ROW_CAP.update(saved)


def test_grid_and_budget_are_the_calibration_these_numbers_came_from():
    assert T._FP32_SOFTMAX_L1_GRID == (8, 8)
    assert T._FP32_SOFTMAX_L1_BYTES_PER_CORE == 768 << 10


@pytest.mark.parametrize("tokens,rows", [(512, 12), (768, 4), (1024, 3)])
def test_shipped_block_sizes(tokens, rows):
    """The measured sizes: 43 blocks per call at 512 aa, 192 at 768, 342 at 1024."""
    per_row, height_per_row = shape(tokens)
    assert T._fp32_softmax_l1_rows(per_row, height_per_row) == rows


def test_a_refusal_at_1024_narrows_to_two_rows_and_not_to_nothing():
    """The regression this file exists for. Retiring instead costs 1.264x on the RF3 trunk."""
    per_row, height_per_row = shape(1024)
    key = (height_per_row, 1024)
    T._fp32_softmax_l1_narrow(key, 3)
    assert T._FP32_SOFTMAX_L1_ROW_CAP[key] == 2
    assert T._fp32_softmax_l1_rows(per_row, height_per_row,
                                  T._FP32_SOFTMAX_L1_ROW_CAP.get(key)) == 2


def test_narrowing_walks_down_to_retirement_and_stays_there():
    per_row, height_per_row = shape(1024)
    key = (height_per_row, 1024)
    seen = []
    for _ in range(4):
        cap = T._FP32_SOFTMAX_L1_ROW_CAP.get(key)
        rows = T._fp32_softmax_l1_rows(per_row, height_per_row, cap)
        seen.append(rows)
        if not rows:
            break
        T._fp32_softmax_l1_narrow(key, rows)
    assert seen == [3, 2, 1, 0]


def test_a_later_refusal_cannot_widen_a_class_back():
    """Blocks arrive in whatever order the caller slices them, so the cap only ever tightens."""
    key = (4096, 1024)
    T._fp32_softmax_l1_narrow(key, 2)
    T._fp32_softmax_l1_narrow(key, 8)
    assert T._FP32_SOFTMAX_L1_ROW_CAP[key] == 1


def test_a_retired_class_takes_the_interleaved_tail():
    per_row, height_per_row = shape(1024)
    assert T._fp32_softmax_l1_rows(per_row, height_per_row, 0) == 0


def test_the_cap_still_respects_the_shard_divisibility_walk():
    """At 768 aa a row block must be even: height_per_row is 3072 and a shard needs cores*32."""
    per_row, height_per_row = shape(768)
    assert height_per_row * 3 % (CORES * 32) != 0
    assert T._fp32_softmax_l1_rows(per_row, height_per_row, 3) == 2


def test_zero_budget_means_no_shard_at_all():
    per_row, height_per_row = shape(1024)
    saved = T._FP32_SOFTMAX_L1_BYTES_PER_CORE
    T._FP32_SOFTMAX_L1_BYTES_PER_CORE = 0
    try:
        assert T._fp32_softmax_l1_rows(per_row, height_per_row) == 0
    finally:
        T._FP32_SOFTMAX_L1_BYTES_PER_CORE = saved


# --- S1: the shard's core count, where the tuned rectangle cannot divide any block -------------
#
# `_fp32_softmax_l1_rows` fixes the shard at 64 cores, so a block is legal only when
# `rows * n_heads * S` is a multiple of 2048. The budget affords rows proportional to S**-2 while
# that multiple grows with S, so above ~512 tokens the walk reaches 0 and the whole tail runs
# interleaved. `_fp32_softmax_l1_plan` keeps the tuned answer wherever it exists and only then
# lets the core count float.


@pytest.fixture(autouse=True)
def clean_plan_cache():
    saved_grid = T.COMPUTE_GRID_MAIN
    T._fp32_softmax_l1_plan.cache_clear()
    yield
    T.COMPUTE_GRID_MAIN = saved_grid
    T._fp32_softmax_l1_plan.cache_clear()


@pytest.mark.parametrize("tokens,rows", [(512, 12), (768, 4), (1024, 3)])
def test_the_plan_is_the_tuned_answer_wherever_the_rectangle_serves(tokens, rows):
    """Every size that is L1-resident today keeps byte for byte its block and its 64 cores."""
    per_row, height_per_row = shape(tokens)
    assert T._fp32_softmax_l1_plan(per_row, height_per_row) == (rows, CORES)


@pytest.mark.parametrize("tokens,rows,cores", [(544, 15, 102), (704, 10, 110), (832, 7, 104)])
def test_the_plan_lights_a_dark_size_with_a_legal_shard(tokens, rows, cores):
    per_row, height_per_row = shape(tokens)
    assert T._fp32_softmax_l1_rows(per_row, height_per_row) == 0
    assert T._fp32_softmax_l1_plan(per_row, height_per_row) == (rows, cores)
    # a height shard needs whole tile rows on every core, and every core under the budget
    assert rows * height_per_row % (cores * 32) == 0
    assert rows * per_row <= cores * T._FP32_SOFTMAX_L1_BYTES_PER_CORE


def test_the_plan_never_asks_for_more_cores_than_the_active_grid():
    """The 110 is a p150a measurement. On a Wormhole 8x8 the active grid IS 64 cores, and a plan
    asking for more hands `num_cores_to_corerangeset` a count no grid can hold."""
    for grid in ((13, 10), (11, 10), (8, 8)):
        T.COMPUTE_GRID_MAIN = grid
        T._fp32_softmax_l1_plan.cache_clear()
        for tokens in range(32, 1057, 32):
            per_row, height_per_row = shape(tokens)
            rows, cores = T._fp32_softmax_l1_plan(per_row, height_per_row)
            assert cores <= grid[0] * grid[1], (grid, tokens, rows, cores)
            if rows and cores != CORES:
                assert rows * height_per_row % (cores * 32) == 0, (grid, tokens, rows, cores)


def test_a_narrowed_cap_bounds_the_free_plan_too():
    """A refusal has to shrink the floating-core block as well, or the class never backs off."""
    per_row, height_per_row = shape(544)
    rows, _cores = T._fp32_softmax_l1_plan(per_row, height_per_row)
    assert rows == 15
    assert T._fp32_softmax_l1_plan(per_row, height_per_row, rows - 1)[0] < rows


def test_the_flag_off_leaves_the_shipped_behaviour_exactly():
    per_row, height_per_row = shape(544)
    saved = T._FP32_SOFTMAX_L1_ANY_CORES
    T._FP32_SOFTMAX_L1_ANY_CORES = False
    T._fp32_softmax_l1_plan.cache_clear()
    try:
        assert T._fp32_softmax_l1_plan(per_row, height_per_row) == (0, 0)
    finally:
        T._FP32_SOFTMAX_L1_ANY_CORES = saved
        T._fp32_softmax_l1_plan.cache_clear()
