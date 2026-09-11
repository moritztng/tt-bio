"""Every matmul program config tt-bio issues is priced by ONE function.

Three sites plan reuse-multicast matmul circular buffers. They used to carry three
copies of the same arithmetic and two different budgets, and the copy that had drifted
was 201760 B per core more permissive than its siblings -- it priced against
`ttnn.get_max_worker_l1_unreserved_size()` (1532448, measured on qb1's p150a) with no
slack, where the other two price against the allocator's per-bank capacity (1461760, the
number `_l1_bank_bytes` records) and keep 128 KiB clear. A plan admitted by the loose site and refused by the allocator is the shape of
moritztng/tt-bio#14. These tests pin the single formula and the single budget.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ttnn = pytest.importorskip("ttnn")

import tt_bio.tenstorrent as T  # noqa: E402

BF16 = 2
BANK = 1461760          # Blackhole, what the L1 allocator reports per bank
UNRESERVED = 1532448    # qb1 p150a, measured: get_max_worker_l1_unreserved_size()


def test_pricer_reproduces_the_double_buffered_formula():
    """The shape the two 1D-multicast sites were fitted with, spelled out."""
    bw, obh, obw = 3, 5, 16
    tile = 1024 * BF16
    expect = (2 * bw * (obh + obw) * tile
              + obh * obw * (tile + 4096) + (128 << 10))
    assert T._matmul_cb_bytes(bw, obh, obw, BF16) == expect


def test_whole_k_plan_is_not_double_buffered():
    """`in0_block_w` == the whole of K means one block, so nothing to buffer against."""
    k, obh, obw = 7, 4, 2
    single = T._matmul_cb_bytes(k, obh, obw, BF16, buffered=False)
    double = T._matmul_cb_bytes(k, obh, obw, BF16)
    assert double - single == k * (obh + obw) * 1024 * BF16


def test_an_l1_resident_result_is_charged_to_the_plan():
    """ttnn allocates the result before the factory places a single CB, so it counts."""
    base = T._matmul_cb_bytes(1, 2, 2, BF16)
    with_out = T._matmul_cb_bytes(1, 2, 2, BF16, extra_tiles=10)
    assert with_out - base == 10 * 1024 * BF16


def test_slack_is_part_of_the_calibration_not_a_taste_margin():
    assert T._MATMUL_CB_SLACK == 128 << 10
    assert T._matmul_cb_bytes(0, 0, 0, BF16) == T._MATMUL_CB_SLACK


def test_the_gate_and_the_pricer_agree_at_every_rung():
    """_batched_matmul_search must decline exactly what the shared pricer prices over.

    The gate lived inline and could drift from the pricer again; this walks the real
    search and checks the config it returns is one the pricer admits.
    """
    T._batched_matmul_search.cache_clear()
    for rung in range(6):
        cfg = T._batched_matmul_search(batch=32, m_tiles=8, k_tiles=2, n_tiles=1,
                                       elem_bytes=BF16, grid=(13, 10), l1=BANK, rung=rung)
        if cfg is None:
            continue
        assert T._matmul_cb_bytes(cfg.in0_block_w, cfg.per_core_M, cfg.per_core_N,
                                  BF16) <= BANK
    T._batched_matmul_search.cache_clear()


def test_the_loose_budget_admitted_a_plan_the_allocator_refuses():
    """The window the unification closes, in one plan.

    A plan whose circular buffers land between the allocator's per-bank capacity and the
    unreserved number the loose gate used: it was admitted before and is refused now.
    Nothing on a p150a protenix-v2 fold at 704 tokens actually lands in that window --
    audited over 350 decisions, zero flipped -- so closing it is free there. It is not
    free on a part whose plan sits inside it, which is the point.
    """
    bw, obh, obw = 1, 20, 10
    need = T._matmul_cb_bytes(bw, obh, obw, BF16)
    assert need == 1482752
    assert need - T._MATMUL_CB_SLACK <= UNRESERVED, "the loose gate took this plan"
    assert need > BANK, "the shared budget refuses it"
    assert UNRESERVED + T._MATMUL_CB_SLACK - BANK == 201760, (
        "the two budgets differed by this much per core, and that is the whole regression")
