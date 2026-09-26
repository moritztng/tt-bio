"""The taped trimul's L1 residency budget: inert by default, and it refuses what it must.

`_triangle_mul_memory_config` refused L1 outright whenever `ops.taping()` was true, so the same
fold at the same size took L1 for inference and DRAM for a gradient round.
`TT_BIO_TRIMUL_TAPED_L1` prices the loop's own peak instead. The lever is release-gated and OFF,
so the first thing this file asserts is that nothing moves while it is off.

Host-only: no device, no network. The grid and the per-bank capacity are pinned because the
budget scales with both, so an unpinned test would assert a different verdict on every part.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

GRID = (13, 10)          # Blackhole p150a
BANK = 1_461_760         # what the L1 allocator reports per bank on that part
HIDDEN = 128             # AF2 / BindCraft 2 pair width
BC2_SEQ = 288            # BindCraft 2's bucketed token axis


@pytest.fixture
def card(monkeypatch):
    monkeypatch.setattr(T, "COMPUTE_GRID_MAIN", GRID)
    monkeypatch.setattr(T, "_FAST_MODE", False)
    monkeypatch.setattr(T, "_IS_SMALL_GRID", False)
    monkeypatch.setattr(T, "_l1_bank_bytes", lambda: BANK)
    T._TRIMUL_TAPED_L1_CLASH.clear()
    T._TRIMUL_DRAM_SHAPES.clear()
    yield
    T._TRIMUL_TAPED_L1_CLASH.clear()
    T._TRIMUL_DRAM_SHAPES.clear()


@pytest.fixture
def taping(monkeypatch):
    """A tape open, spelled the way `_triangle_mul_memory_config` asks the question."""
    from tt_bio import ops
    monkeypatch.setattr(ops, "taping", lambda: True)


@pytest.fixture
def lever(monkeypatch):
    monkeypatch.setattr(T, "_TRIMUL_TAPED_L1", True)
    monkeypatch.setattr(T, "_TRIMUL_TAPED_L1_SHARE", 0.5)


def test_off_is_the_shipped_refusal(card, taping):
    """Default OFF: a tape sends every shape to DRAM, exactly as it did before the lever."""
    assert not T._TRIMUL_TAPED_L1
    for seq in (128, BC2_SEQ, 352, 384, 512):
        assert T._triangle_mul_memory_config(seq, HIDDEN, 1) is T.ttnn.DRAM_MEMORY_CONFIG


def test_inference_never_asks_the_taped_budget(card, lever):
    """With no tape the gate is the sequence threshold and nothing else, lever or no lever."""
    assert T._triangle_mul_memory_config(BC2_SEQ, HIDDEN, 1) is T.ttnn.L1_MEMORY_CONFIG
    assert T._triangle_mul_memory_config(512, HIDDEN, 1) is T.ttnn.DRAM_MEMORY_CONFIG


def test_a_caller_that_cannot_name_the_width_keeps_the_refusal(card, lever, taping):
    """`MiniTriangularUpdate` and every other unpriced site is byte-identical to today."""
    assert T._triangle_mul_memory_config(BC2_SEQ) is T.ttnn.DRAM_MEMORY_CONFIG
    assert T._triangle_mul_memory_config(BC2_SEQ, HIDDEN, 1) is T.ttnn.L1_MEMORY_CONFIG


@pytest.mark.parametrize("seq,want_l1", [(BC2_SEQ, True), (352, True), (512, False)])
def test_the_budget_table(card, lever, seq, want_l1):
    """The table in state/perf10/SHARED-KERNELS.md, asserted rather than transcribed.

    512 is the row that matters: six chunk-multiples at width 32 is 96.00 MB against 181.23 MiB
    of banks, 53.0 %, over the 50 % share -- so the budget refuses it on bytes and not only
    because `TRIANGLE_MULT_L1_MAX_SEQ` would.
    """
    assert T._trimul_taped_l1_fits(seq, HIDDEN, 1) is want_l1


def test_the_budget_prices_the_batch(card, lever):
    """A confidence head arrives with one pair copy per diffusion sample; the budget sees it."""
    assert T._trimul_taped_l1_fits(BC2_SEQ, HIDDEN, 1)
    assert not T._trimul_taped_l1_fits(BC2_SEQ, HIDDEN, 8)


def test_the_sequence_threshold_is_still_the_outer_gate(card, lever, taping):
    """384 fits the budget and is still DRAM, because 384 > TRIANGLE_MULT_L1_MAX_SEQ = 352."""
    assert T._trimul_taped_l1_fits(384, HIDDEN, 1)
    assert T._triangle_mul_memory_config(384, HIDDEN, 1) is T.ttnn.DRAM_MEMORY_CONFIG


def test_a_recorded_clash_demotes_that_shape_and_only_that_shape(card, lever, taping):
    """The clash is the test. A shape that threw takes DRAM; its neighbours do not."""
    assert T._triangle_mul_memory_config(BC2_SEQ, HIDDEN, 1) is T.ttnn.L1_MEMORY_CONFIG
    T._TRIMUL_TAPED_L1_CLASH.add(T._trimul_chunk_key(BC2_SEQ, HIDDEN, 1))
    assert T._triangle_mul_memory_config(BC2_SEQ, HIDDEN, 1) is T.ttnn.DRAM_MEMORY_CONFIG
    assert T._triangle_mul_memory_config(BC2_SEQ, HIDDEN, 2) is T.ttnn.L1_MEMORY_CONFIG
    assert T._triangle_mul_memory_config(320, HIDDEN, 1) is T.ttnn.L1_MEMORY_CONFIG


def test_a_taped_clash_does_not_cost_inference_its_l1(card, lever):
    """Tape-only. `_TRIMUL_DRAM_SHAPES` is the gate that demotes both; this one must not.

    No `taping` fixture here on purpose: this is the fold that runs after the gradient round in
    the same process, and it keeps the residency it has always had.
    """
    T._TRIMUL_TAPED_L1_CLASH.add(T._trimul_chunk_key(BC2_SEQ, HIDDEN, 1))
    assert T._triangle_mul_memory_config(BC2_SEQ, HIDDEN, 1) is T.ttnn.L1_MEMORY_CONFIG


def test_one_width_walk(card):
    """`_trimul_chunk_size` and the taped budget read the same area and the same budget."""
    budget = T._trimul_l1_chunk_budget()
    c = T._trimul_chunk_size(BC2_SEQ, HIDDEN, 1)
    assert T._trimul_chunk_l1_area(BC2_SEQ, c, 1) <= budget
    assert T._trimul_chunk_l1_area(BC2_SEQ, c * 2, 1) > budget
