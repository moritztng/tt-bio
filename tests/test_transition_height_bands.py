"""The Transition row height at OpenDDE's c_z=384 on a Wormhole Galaxy, as a step function of width.

Through 1088 tokens the height is the element raise, SMALL_GRID_TRANSITION_ELEMS // (W * c), capped
by per-core L1; above it the ratio budget takes over. The raise used to stop at the 608-token chunking
threshold, which left h=2 from 608 to 896 and h=1 from 928 up. The heights it picks now are the ones
measured torch.equal against h=1 (state/spd-opendde.md, th1). Host-only.
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
    h = max(1, int(base_h * min(1.0, ref / (w_eff * C))))
    if W <= T.SEQ_LEN_MORE_CHUNKING:   # the c=384 element raise, H = W here
        h = max(1, min(base_h, T.SMALL_GRID_TRANSITION_ELEMS // (w_eff * C)))
    return min(h, max(1, int(l1_rows(w_eff))))


def test_the_band_edges(wormhole):
    assert T.TRANSITION_H_CHUNK_SIZE == 16                     # the base the bands scale from
    assert T.SEQ_LEN_MORE_CHUNKING == 1088                     # the raise's upper bound on this part
    # Up to 1088 the element raise sets the height (1179648 // (W * 384)); above it the ratio does,
    # and h jumps back to 3 above 1792 where w_chunked flips on and w_eff drops to 512.
    bands = [(480, 512, 6), (544, 608, 5), (640, 768, 4), (800, 1024, 3), (1056, 1088, 2),
             (1120, 1792, 1), (1824, 2080, 3)]
    for lo, hi, h in bands:
        assert _height(lo) == h and _height(hi) == h, (lo, hi, h)
        if lo > 480:
            assert _height(lo - 32) != h                        # the edge is where it is claimed
        if hi < 2080:
            assert _height(hi + 32) != h
    assert _height(1792) == 1 and _height(1824) == 3            # the w_chunked flip, explicitly


def test_the_raise_stays_at_measured_exact_heights(wormhole):
    """th1 (WH .114): every height up to 4 at W=736 is torch.equal to h=1, h=5 and h=6 are not
    (max |diff| 2.0) although the L1 cap admits 5. 896 and 1024 land on 3."""
    assert _height(736) == 4
    assert _height(896) == _height(1024) == 3
