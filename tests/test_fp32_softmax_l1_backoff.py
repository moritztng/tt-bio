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
    saved = dict(T._FP32_SOFTMAX_L1_ROW_CAP), dict(T._FP32_SOFTMAX_L1_FREE_ROW_CAP)
    saved_refusals = dict(T._FP32_SOFTMAX_L1_REFUSALS)
    for d in (T._FP32_SOFTMAX_L1_ROW_CAP, T._FP32_SOFTMAX_L1_FREE_ROW_CAP,
              T._FP32_SOFTMAX_L1_REFUSALS):
        d.clear()
    yield
    for d, keep in ((T._FP32_SOFTMAX_L1_ROW_CAP, saved[0]),
                    (T._FP32_SOFTMAX_L1_FREE_ROW_CAP, saved[1]),
                    (T._FP32_SOFTMAX_L1_REFUSALS, saved_refusals)):
        d.clear()
        d.update(keep)


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
    assert T._fp32_softmax_l1_plan(per_row, height_per_row, tokens) == (rows, CORES)


@pytest.mark.parametrize("tokens,rows,cores", [(544, 15, 102), (704, 10, 110), (832, 7, 104)])
def test_the_plan_lights_a_dark_size_with_a_legal_shard(tokens, rows, cores):
    per_row, height_per_row = shape(tokens)
    assert T._fp32_softmax_l1_rows(per_row, height_per_row) == 0
    assert T._fp32_softmax_l1_plan(per_row, height_per_row, tokens) == (rows, cores)
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
            rows, cores = T._fp32_softmax_l1_plan(per_row, height_per_row, tokens)
            assert cores <= grid[0] * grid[1], (grid, tokens, rows, cores)
            if rows and cores != CORES:
                assert rows * height_per_row % (cores * 32) == 0, (grid, tokens, rows, cores)


def test_a_narrowed_cap_bounds_the_free_plan_too():
    """A refusal has to shrink the floating-core block as well, or the class never backs off.

    The floating plan reads its OWN cap, so the refusal that shrinks it is one recorded against it.
    """
    per_row, height_per_row = shape(544)
    rows, _cores = T._fp32_softmax_l1_plan(per_row, height_per_row, 544)
    assert rows == 15
    assert T._fp32_softmax_l1_plan(per_row, height_per_row, 544,
                                   free_cap=rows - 1)[0] < rows


def test_the_flag_off_leaves_the_shipped_behaviour_exactly():
    per_row, height_per_row = shape(544)
    saved = T._FP32_SOFTMAX_L1_ANY_CORES
    T._FP32_SOFTMAX_L1_ANY_CORES = False
    T._fp32_softmax_l1_plan.cache_clear()
    try:
        assert T._fp32_softmax_l1_plan(per_row, height_per_row, 544) == (0, 0)
    finally:
        T._FP32_SOFTMAX_L1_ANY_CORES = saved
        T._fp32_softmax_l1_plan.cache_clear()


@pytest.mark.parametrize("tokens", [515, 546, 547, 1023])
def test_a_ragged_key_dim_gets_no_plan_at_all(tokens):
    """A width that is not whole tiles has no shard, so a block cap would be pure loop overhead:
    measured 0.786x at 2 heads, 0.928x at 4 and 0.910x at 8 on a 515-token key dim."""
    per_row, height_per_row = shape(tokens)
    assert T._fp32_softmax_l1_rows(per_row, height_per_row) == 0     # the tuned answer is dark too
    assert T._fp32_softmax_l1_plan(per_row, height_per_row, tokens) == (0, 0)


def test_a_ragged_width_gets_exactly_the_shipped_answer():
    """The free search must not plan at a width no shard can take. It would cap the block with
    nothing behind it, and the loop then pays a slice and a concat per block for no residency:
    measured 0.786x at 2 heads, 0.928x at 4 and 0.910x at 8 on a 515-token key dim
    (perf/fp32softmax/results/s1_op_bitexact.json).

    Where the tuned rectangle itself blocks a ragged width -- it does, at 16 four-head sizes
    between 132 and 1056 tokens -- that is the shipped behaviour and this asserts S1 leaves it
    alone, not that it is right.
    """
    for heads in (1, 2, 4, 8, 16):
        for tokens in range(33, 1057):
            if tokens % 32 == 0:
                continue
            hpr = heads * tokens
            per_row = hpr * tokens * 4
            rows = T._fp32_softmax_l1_rows(per_row, hpr)
            want = (rows, CORES) if rows else (0, 0)
            assert T._fp32_softmax_l1_plan(per_row, hpr, tokens) == want, (heads, tokens)


# --- S2: the free core count offered at every size, under the matmul-config constraint ---------
#
# `_fp32_softmax_l1_plan` with `_FP32_SOFTMAX_L1_FLOAT_CORES` on may replace a block the tuned 8x8
# rectangle CAN serve. The first version of that took the tallest block the byte budget affords and
# was a wash at 512 aa: 18 rows on 96 cores cut the block count 32.6 % and returned 1.013x, because
# `batch = rows * n_heads` went 48 -> 72 and `_batched_matmul_search` has no legal config there, so
# every q@k^T fell back to ttnn's own planner. The objective below is the tallest block that still
# admits a config, and the constraint applies only where the block being replaced had one to lose.
#
# These are host tests, so `_batched_matmul_config` has no device to read L1 from. It is pinned to
# the numbers this card actually reports -- 110 cores at 1532416 B unreserved -- because the plan
# is a pure function of them and the plans below are the ones the on-device sweep measured
# (perf/fp32softmax/results/s2_op_ab.json).

L1_UNRESERVED = 1532416         # ttnn.get_max_worker_l1_unreserved_size() on this p150a


@pytest.fixture
def s2_on(monkeypatch):
    monkeypatch.setattr(T, "_FP32_SOFTMAX_L1_FLOAT_CORES", True)
    monkeypatch.setattr(T, "_batched_matmul_config",
                        lambda batch, mt, kt, nt, elem, rung=0: T._batched_matmul_search(
                            batch, mt, kt, nt, elem, tuple(T.COMPUTE_GRID_MAIN), L1_UNRESERVED,
                            rung))
    T._fp32_softmax_l1_plan.cache_clear()
    yield
    T._fp32_softmax_l1_plan.cache_clear()


def bmm(tokens: int, heads: int = N_HEADS, head_dim: int = 32) -> tuple:
    """The (heads, Mt, head-dim tiles, key tiles, value tiles) the two score matmuls take."""
    t = -(-tokens // 32)
    return heads, t, -(-head_dim // 32), t, -(-head_dim // 32)


@pytest.mark.parametrize("heads,tokens,plan", [
    (4, 256, (81, 108)), (4, 512, (13, 104)), (4, 576, (13, 104)), (4, 640, (11, 110)),
    (4, 768, (9, 108)), (4, 1024, (3, 96)), (2, 512, (27, 108)), (2, 1024, (6, 96)),
    (8, 512, (6, 96)), (8, 576, (6, 108)),
])
def test_s2_plans_are_the_ones_the_sweep_measured(s2_on, heads, tokens, plan):
    T.COMPUTE_GRID_MAIN = (11, 10)
    T._fp32_softmax_l1_plan.cache_clear()
    hpr = heads * tokens
    per_row = hpr * tokens * 4
    assert T._fp32_softmax_l1_plan(per_row, hpr, tokens, None, bmm(tokens, heads)) == plan


def test_s2_never_returns_a_worse_plan_than_the_tuned_one(s2_on):
    """Taller, or the same height on more cores, or exactly the tuned answer. Never below it.

    The tie on height matters as much as the height: the same 3-row block at 1024 aa is 786432
    B/core on 64 and 524288 on 96, and it is the 64-core version that refuses its circular buffers
    and forces the 2-row backoff.
    """
    for grid in ((13, 10), (11, 10), (8, 8)):
        T.COMPUTE_GRID_MAIN = grid
        T._fp32_softmax_l1_plan.cache_clear()
        for heads in (1, 2, 4, 8, 16):
            for tokens in range(32, 1057, 32):
                hpr = heads * tokens
                per_row = hpr * tokens * 4
                tuned = T._fp32_softmax_l1_rows(per_row, hpr)
                rows, cores = T._fp32_softmax_l1_plan(per_row, hpr, tokens, None,
                                                      bmm(tokens, heads))
                assert cores <= grid[0] * grid[1], (grid, heads, tokens)
                if not tuned:
                    continue
                assert (rows, cores) >= (tuned, CORES), (grid, heads, tokens, tuned)
                assert rows * hpr % (cores * 32) == 0, (grid, heads, tokens)
                assert rows * per_row <= cores * T._FP32_SOFTMAX_L1_BYTES_PER_CORE


def test_s2_keeps_the_batched_matmul_config_wherever_the_tuned_block_had_one(s2_on):
    """The whole objective, asserted rather than described. A size whose tuned block has no config
    is unconstrained -- there is nothing to lose, which is why S1's decline count could grow 3.9x
    on the arm that won 1.214x."""
    T.COMPUTE_GRID_MAIN = (11, 10)
    T._fp32_softmax_l1_plan.cache_clear()
    moved = 0
    for heads in (1, 2, 4, 8, 16):
        for tokens in range(32, 1057, 32):
            hpr = heads * tokens
            per_row = hpr * tokens * 4
            tuned = T._fp32_softmax_l1_rows(per_row, hpr)
            if not tuned or not T._fp32_softmax_bmm_served(tuned, bmm(tokens, heads)):
                continue
            rows, cores = T._fp32_softmax_l1_plan(per_row, hpr, tokens, None, bmm(tokens, heads))
            assert T._fp32_softmax_bmm_served(rows, bmm(tokens, heads)), (heads, tokens, rows)
            moved += (rows, cores) != (tuned, CORES)
    assert moved > 20, moved      # the lever has to actually move something for this to mean much


def test_s2_off_ignores_the_matmul_shape_entirely():
    """The flag off is byte for byte what shipped, whatever `bmm` says."""
    assert not T._FP32_SOFTMAX_L1_FLOAT_CORES
    for heads in (2, 4, 8):
        for tokens in range(32, 1057, 32):
            hpr = heads * tokens
            per_row = hpr * tokens * 4
            rows = T._fp32_softmax_l1_rows(per_row, hpr)
            got = T._fp32_softmax_l1_plan(per_row, hpr, tokens, None, bmm(tokens, heads))
            if rows:
                assert got == (rows, CORES), (heads, tokens)


def test_a_dark_size_is_untouched_by_the_constraint(s2_on):
    """S1's own sizes have no tuned block and therefore no config to protect: same plan as main."""
    T.COMPUTE_GRID_MAIN = (11, 10)
    T._fp32_softmax_l1_plan.cache_clear()
    for tokens in (544, 704, 832, 960):
        hpr = N_HEADS * tokens
        per_row = hpr * tokens * 4
        assert T._fp32_softmax_l1_rows(per_row, hpr) == 0
        free = T._fp32_softmax_l1_free_rows(per_row, hpr, None, None)
        assert T._fp32_softmax_l1_plan(per_row, hpr, tokens, None, bmm(tokens)) == free


def test_retiring_a_floating_plan_falls_back_to_the_tuned_block_and_not_to_nothing():
    """The 768 aa cliff. Retiring the floating plan must not also drop the BLOCK CAP: an unblocked
    call materialises the whole fp32 score tensor -- 7.25 GB at 768 aa / 4 heads, measured 170.9 ms
    against 127.4 ms for the tuned 4-row block. So what the caller falls back to has to be the
    tuned row count, and at this size there IS one.

    And it has to be the FULL tuned row count. A floating refusal is recorded against the floating
    cap, so the block that ships is exactly the block that ships however far the walk descended.
    """
    per_row, height_per_row = shape(768)
    key = (height_per_row, 768)
    for rows in (9, 8, 7, 6, 5):
        T._fp32_softmax_l1_narrow(key, rows, free=True)
    assert T._FP32_SOFTMAX_L1_FREE_ROW_CAP[key] == 4
    assert key not in T._FP32_SOFTMAX_L1_ROW_CAP
    assert T._fp32_softmax_l1_rows(per_row, height_per_row,
                                   T._FP32_SOFTMAX_L1_ROW_CAP.get(key)) == 4


# --- the leash: how many rungs the floating walk gets, priced on what retirement lands on -------


def test_the_leash_at_768_reaches_every_rung_above_the_shipped_block(s2_on):
    """The plan is 9 rows on 108 cores and the shipped block is 4. A flat count of 2 retired the
    class at the second refusal and the fold read 1.0023x with 114 of 173 calls on the tuned block.
    The leash is the rungs above 4, so the walk gets all of them and none below.
    """
    T.COMPUTE_GRID_MAIN = (11, 10)
    T._fp32_softmax_l1_plan.cache_clear()
    per_row, height_per_row = shape(768)
    key = (height_per_row, 768)
    assert T._fp32_softmax_l1_plan(per_row, height_per_row, 768, None,
                                   bmm(768)) == (9, 108)
    seen = []
    while not T._fp32_softmax_free_spent(key, per_row, height_per_row):
        rows, cores = T._fp32_softmax_l1_plan(per_row, height_per_row, 768, None, bmm(768),
                                              T._FP32_SOFTMAX_L1_FREE_ROW_CAP.get(key))
        if cores == CORES:
            break
        seen.append((rows, cores))
        T._fp32_softmax_l1_narrow(key, rows, free=True)
    # every rung above the shipped 4 rows, then the shipped height itself on 96 cores instead of
    # 64 -- 524288 B/core against 786432, which is the shape the 1024 aa cell won on
    assert seen == [(9, 108), (8, 96), (7, 96), (6, 96), (5, 96), (4, 96)], seen
    # spent, and the plan is now the shipped block at its shipped height
    assert T._fp32_softmax_free_spent(key, per_row, height_per_row)
    assert T._fp32_softmax_l1_rows(per_row, height_per_row,
                                   T._FP32_SOFTMAX_L1_ROW_CAP.get(key)) == 4


def test_the_leash_stays_the_measured_count_where_retirement_lands_on_nothing(s2_on):
    """2 heads / 960 tokens: no tuned block, so the baseline is ONE unblocked call and a refused
    rung multiplies the block count instead of dividing it. Measured 0.295x with 16 of 29 blocks
    resident, and the cap of 2 is what stops it. The leash must not grow here.
    """
    T.COMPUTE_GRID_MAIN = (11, 10)
    T._fp32_softmax_l1_plan.cache_clear()
    hpr = 2 * 960
    per_row = hpr * 960 * 4
    key = (hpr, 960)
    assert T._fp32_softmax_l1_rows(per_row, hpr) == 0          # nothing to fall back to
    assert T._fp32_softmax_l1_plan(per_row, hpr, 960, None, bmm(960, 2)) == (11, 110)
    for n in (1, 2):
        T._FP32_SOFTMAX_L1_REFUSALS[key] = n
        T._fp32_softmax_l1_narrow(key, 11, free=True)
        assert T._fp32_softmax_free_spent(key, per_row, hpr) == (n >= 2)


def test_a_floating_refusal_never_narrows_the_shipped_block(s2_on):
    """The two caps are separate because a refusal is a property of the shard that refused: at
    1024 aa the sharded softmax refuses 3 rows on 64 cores and accepts the same 3 on 96. Charging
    a floating refusal to the tuned cap would shrink the block retirement has to land on.
    """
    T.COMPUTE_GRID_MAIN = (11, 10)
    T._fp32_softmax_l1_plan.cache_clear()
    for tokens, tuned in ((512, 12), (768, 4), (1024, 3)):
        per_row, height_per_row = shape(tokens)
        key = (height_per_row, tokens)
        for rows in range(20, 0, -1):
            T._fp32_softmax_l1_narrow(key, rows, free=True)
        assert key not in T._FP32_SOFTMAX_L1_ROW_CAP, tokens
        assert T._fp32_softmax_l1_rows(per_row, height_per_row) == tuned
        assert T._fp32_softmax_l1_plan(per_row, height_per_row, tokens, None, bmm(tokens),
                                       T._FP32_SOFTMAX_L1_FREE_ROW_CAP.get(key)) == (tuned, CORES)


def test_a_tuned_refusal_still_backs_the_tuned_block_off_one_row(s2_on):
    """The 1024 aa backoff is unchanged: a refusal by the tuned shard narrows the tuned cap."""
    per_row, height_per_row = shape(1024)
    key = (height_per_row, 1024)
    T._fp32_softmax_l1_narrow(key, 3)
    assert T._FP32_SOFTMAX_L1_ROW_CAP[key] == 2
    assert key not in T._FP32_SOFTMAX_L1_FREE_ROW_CAP
    assert T._fp32_softmax_l1_rows(per_row, height_per_row,
                                   T._FP32_SOFTMAX_L1_ROW_CAP.get(key)) == 2
