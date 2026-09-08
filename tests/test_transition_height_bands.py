"""The Transition row height is a step function of the width, and three rungs share a step.

That map is what refuted the first explanation of the 896 aa anomaly. `h=2` covers W=608..896, so
640 aa (folds in 490 s) and 768 aa (folds in 876 s) run the SAME height as 896 aa (four attempts,
never finished) -- the height cannot be what distinguishes them. What survives is the width at a
fixed height: 768 -> 896 is 1.36x the element-work for at least 4.2x the wall clock, and 1024 aa
does MORE element-work than 896 and finishes in 1553 s.

Pinned here because the whole reading rests on those band edges: if the derivation moves, the
argument in state/opendde-l1-clash-to-1024.md needs redoing, and this fails first. Host-only.
"""
from __future__ import annotations

from unittest import mock

import pytest

from tt_bio import tenstorrent as T

GIB = 2 ** 30
C = 384        # OpenDDE's pair-track channel
HID = 1536     # 4 * C, the Transition's fc1 width
GRID = (8, 9)

_WRITES = ("_IS_SMALL_GRID", "SEQ_LEN_MORE_CHUNKING", "TRANSITION_W_CHUNKING_THRESHOLD",
           "TRANSITION_BATCH_CHUNKING_THRESHOLD", "TRIANGLE_ATT_CHUNK_SIZE_FAST",
           "TRANSITION_W_CHUNK_SIZE", "TRIANGLE_MULT_L1_MAX_SEQ_FAST", "SMALL_GRID_SEQ_TILE",
           "SMALL_GRID_PAIR_TILE_AREA", "SMALL_GRID_MSA_TILE_AREA", "TRIANGLE_MULT_L1_MAX_SEQ",
           "TRANSITION_L1_CHUNK_BYTES_PER_CORE")


@pytest.fixture
def wormhole():
    saved = {n: getattr(T, n) for n in _WRITES}
    with mock.patch.object(T, "_dram_total_bytes", lambda d: 12 * GIB), \
         mock.patch.object(T.ttnn, "get_max_worker_l1_unreserved_size",
                           lambda: T._WH_FULL_L1_PER_CORE):
        T._apply_grid_thresholds(GRID)
    yield
    for n, v in saved.items():
        setattr(T, n, v)


def _height(W):
    """`Transition.__call__`'s height at this width, mirrored (see wh_transition_chunk.py)."""
    tile = lambda v: -(-int(v) // 32) * 32
    gx, gy = T.COMPUTE_GRID_MAIN
    base_h = T.TRANSITION_H_CHUNK_SIZE
    ref = 1024 * 128 * 128 // max(128, C) if T._IS_SMALL_GRID else 1024 * 128
    l1_rows = lambda w: (T.TRANSITION_L1_CHUNK_BYTES_PER_CORE * gx * gy
                         / (2 * tile(w) * (tile(C) + 2 * tile(HID))))
    rows_at = lambda w: min(base_h * min(1.0, ref / (w * C)), l1_rows(w))
    w_eff = min(W, T.TRANSITION_W_CHUNK_SIZE) if rows_at(W) < 1.0 else W
    return min(max(1, int(base_h * min(1.0, ref / (w_eff * C)))), max(1, int(l1_rows(w_eff))))


def test_the_band_edges(wormhole):
    assert T.TRANSITION_H_CHUNK_SIZE == 16                     # the base the bands scale from
    # Measured edges, not guessed: the h=3 band starts at 480, and h jumps BACK to 3 above 1792
    # because that is where w_chunked flips on and w_eff drops to TRANSITION_W_CHUNK_SIZE=512.
    bands = [(480, 576, 3), (608, 896, 2), (928, 1792, 1), (1824, 2080, 3)]
    for lo, hi, h in bands:
        assert _height(lo) == h and _height(hi) == h, (lo, hi, h)
        if lo > 480:
            assert _height(lo - 32) != h                        # the edge is where it is claimed
        if hi < 2080:
            assert _height(hi + 32) != h
    assert _height(1792) == 1 and _height(1824) == 3            # the w_chunked flip, explicitly


def test_three_rungs_share_h2_and_two_of_them_fold(wormhole):
    """The refutation, as an assertion: the height cannot separate 896 from 640 and 768."""
    assert _height(640) == _height(768) == _height(896) == 2
    assert _height(1024) == 1


def test_the_width_is_what_isolates_896(wormhole):
    """At the shared height, 896 does more work than 768 and less than 1024 -- and only 896's
    width is not 2-smooth-with-one-small-odd-factor."""
    work = lambda W, h: -(-W // h) * h * W * C
    assert work(768, 2) < work(896, 2) < work(1024, 1)          # 226.5 M < 308.3 M < 402.7 M
    tiles = {W: W // 32 for W in (640, 768, 896, 1024)}
    assert tiles == {640: 20, 768: 24, 896: 28, 1024: 32}
    assert 28 % 7 == 0 and all(t % 7 for t in (20, 24, 32))     # 896 is the only one with a 7
