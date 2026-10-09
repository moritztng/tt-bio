"""The Blackhole Transition row height, pinned to the heights hardware actually ran.

`_BH_TRANSITION_L1_CHUNK_BYTES_PER_CORE` and `_BH_TRANSITION_CHUNK_ELEMS` are the only two numbers
between a shipped fold and a static-CB clash at `program.cpp:1052`. The ladder measured h=48/32/24
safe and bit-exact at W=512/768/1024 (c=128, hidden=512, 11x10 p300c) and every larger product
throwing, so those three heights are the contract; this pins the constants to them from the other
side, and pins the two guards that keep the rule from walking off its measurement.

Host-only: the derivation is arithmetic on the shape and the grid, so it needs no card. The height
expression is mirrored here the same way `test_transition_height_bands.py` mirrors the small-grid
one -- the engine's copy lives in a closure inside `Transition.__call__`.
"""
from __future__ import annotations

import pytest

from tt_bio import tenstorrent as T

C, HID = 128, 512      # the pair track: Boltz-2, BoltzGen, OpenFold3, Protenix-v2's 128 sibling
MSA_C, MSA_HID = 64, 256
P300C = (11, 10)       # 110 cores, the part every height below was measured on
P150A = (13, 10)       # 130 cores


def _height(W, c=C, hid=HID, grid=P300C, budget=None, b8=False):
    """`Transition.__call__`'s Blackhole height at this shape, mirrored. `b8`: the hidden is bfp8."""
    tile = lambda v: -(-int(v) // 32) * 32
    gx, gy = grid
    budget = T.TRANSITION_L1_CHUNK_BYTES_PER_CORE if budget is None else budget
    b = (T.TRANSITION_H_CHUNK_SIZE_BIG
         if W <= T.TRANSITION_H_CHUNK_BIG_MAX_W and c <= 256 else T.TRANSITION_H_CHUNK_SIZE)
    base = max(1, int(b * min(1.0, (1024 * 128) / (W * c))))
    if c > (T._BH_TRANSITION_L1_ROWS_MAX_C_B8 if b8 else T._BH_TRANSITION_L1_ROWS_MAX_C):
        return base
    hb = 1088 / 1024 if b8 else 2
    l1_rows = budget * gx * gy / (tile(W) * (2 * tile(c) + 2 * hb * tile(hid)))
    elems = T._BH_TRANSITION_CHUNK_ELEMS * 18 / (W * c * (2 + 8 * hb))
    return max(base, max(1, int(min(l1_rows, elems))))


def test_the_budget_is_the_ceil_of_the_measured_aggregate():
    # One h=48 block at W=512, c=128, hidden=512 in bf16 is this many bytes live across the grid.
    live = 2 * 48 * 512 * (C + 2 * HID)
    assert live == 56_623_104
    assert T._BH_TRANSITION_L1_CHUNK_BYTES_PER_CORE == -(-live // (P300C[0] * P300C[1]))
    assert T._BH_TRANSITION_L1_CHUNK_BYTES_PER_CORE == 514_756
    # The default in force on Blackhole is that number: _apply_grid_thresholds returns early there.
    assert T.TRANSITION_L1_CHUNK_BYTES_PER_CORE == T._BH_TRANSITION_L1_CHUNK_BYTES_PER_CORE
    assert T._TRANSITION_L1_ROWS is True


@pytest.mark.parametrize("W,h", [(320, 76), (512, 48), (768, 32), (1024, 24)])
def test_the_measured_heights(W, h):
    assert _height(W) == h


@pytest.mark.parametrize("W,h", [(512, 48), (768, 32), (1024, 24)])
def test_the_floor_of_the_same_aggregate_is_one_row_short(W, h):
    """Negative control for the ceil: the truncated budget misses every measured height.

    514,755 B/core is the same aggregate divided the other way, and it reads correct in prose. In
    code `int()` drops each height by one, which costs two extra row blocks at 768 aa and two at
    1024 aa. If someone "tidies" the constant down, this is what fails.
    """
    assert _height(W, budget=514_755) == h - 1


def test_a_wider_grid_stays_on_the_measured_extent():
    """130 cores would buy 56 rows at W=512 by bytes alone. Nothing has run 56 rows."""
    tile = lambda v: -(-int(v) // 32) * 32
    by_bytes = (T.TRANSITION_L1_CHUNK_BYTES_PER_CORE * P150A[0] * P150A[1]
                / (2 * tile(512) * (tile(C) + 2 * tile(HID))))
    assert int(by_bytes) == 56
    assert _height(512, grid=P150A) == 48
    assert _height(768, grid=P150A) == 32
    assert _height(1024, grid=P150A) == 24


@pytest.mark.parametrize("W", [1536, 2048, 3072])
def test_the_lever_is_worth_nothing_at_and_above_1536_tokens(W):
    """At c=128 the rule reduces to h*W = 24576, so above 1536 tokens it only undoes the ratio
    shrink and never exceeds the extent the base already sits at."""
    assert _height(W) * W == 24_576


def test_the_msa_track_gets_its_own_height_from_the_same_expression():
    """A quarter of the pair channel, so four times the rows: this is why a flat constant cannot
    express the lever. Verified on device at MSA depth 1024, bit-exact, digest 9aaff566b4888d96."""
    assert _height(512, c=MSA_C, hid=MSA_HID) == 96
    assert _height(768, c=MSA_C, hid=MSA_HID) == 64
    assert _height(1024, c=MSA_C, hid=MSA_HID) == 48
    # 96 rows at W=512, c=64 is 49,152 extent -- twice the c=128 extent -- and exactly the budget.
    assert 2 * 96 * 512 * (MSA_C + 2 * MSA_HID) // (P300C[0] * P300C[1]) == 514_755


def test_the_byte_budget_binds_when_hidden_exceeds_four_channels():
    """The element cap is the same statement only while hidden = 4c. Push hidden to 8c and the
    physical expression has to be the one that wins, or the rule would double the live bytes."""
    wide = _height(512, c=C, hid=8 * C)
    assert wide < T._BH_TRANSITION_CHUNK_ELEMS / (512 * C)
    assert 2 * wide * 512 * (C + 2 * 8 * C) <= 56_623_104


@pytest.mark.parametrize("c,hid", [(256, 1024), (384, 1536)])
def test_a_wider_channel_keeps_todays_height(c, hid):
    """The budget was fitted at c=128 and it over-predicts above it.

    Unbounded, the raise gives OpenDDE (c_z=384) 25 rows at W=320 and the fold dies on
    `program.cpp:1052` on every seed of `opendde-prot-prod`, `opendde-abag` and the gate's capacity
    leg, all three of which PASS with `TT_BIO_TRANSITION_L1_ROWS=0`. So the rule declines above the
    channel it can speak for, and those shapes keep the height they ship today.
    """
    assert T._BH_TRANSITION_L1_ROWS_MAX_C == 128
    for W in (320, 512, 768, 1024):
        b = (T.TRANSITION_H_CHUNK_SIZE_BIG
             if W <= T.TRANSITION_H_CHUNK_BIG_MAX_W and c <= 256 else T.TRANSITION_H_CHUNK_SIZE)
        assert _height(W, c=c, hid=hid) == max(1, int(b * min(1.0, (1024 * 128) / (W * c))))


def test_the_channels_that_do_ship_are_unaffected_by_the_bound():
    """c=128 pair track and c=64 MSA track, the two the heights were measured at."""
    assert _height(512) == 48 and _height(768) == 32 and _height(1024) == 24
    assert _height(512, c=MSA_C, hid=MSA_HID) == 96


@pytest.mark.parametrize("W,h_bf16,h_b8", [(736, 11, 28), (1024, 8, 20)])
def test_bfp8_hidden_is_priced_at_its_own_width(W, h_bf16, h_b8):
    """Fast mode (`transition_b8`) writes x_1/x_2 in bfp8, 1088 B per 1024 elements. Priced as
    bf16 and stopped at c=128, Protenix-v2's c=256 pair transition ran the ratio height (11 at 736
    tokens) in fast mode too. On a p150a the bfp8 module ran h=32 clean at 736 and was 1.2x faster
    than h=11 (perf/spd_bh/transition_h.py); bf16 hidden clashes at h=20, so normal keeps its bound."""
    assert _height(W, c=256, hid=1024) == h_bf16
    assert _height(W, c=256, hid=1024, b8=True) == h_b8


def test_bfp8_hidden_at_the_bf16_bound_is_the_byte_ratio():
    # c=128: the same bytes per row hold (2 + 8 * 2) / (2 + 8 * 1.0625) = 1.714x the rows.
    assert _height(736) == 33
    assert _height(736, b8=True) == 57


def test_bfp8_hidden_does_not_reach_opendde_channel():
    assert _height(512, c=384, hid=1536, b8=True) == _height(512, c=384, hid=1536)
