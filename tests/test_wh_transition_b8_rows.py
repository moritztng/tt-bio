"""Wormhole Transition row height with bfp8 hidden (`transition_b8`, fast mode) at c <= 128.

The small-grid path capped the height by an L1 budget that priced x_1/x_2 as bf16 and started from
the 16-row base, so the c=128 MSA transition ran h=16 at 736 tokens in fast mode. With bfp8 hidden
it starts from the fast base (32) and the bfp8-priced cap binds: 28 at 736. Row height does not
change a row-local swiglu's result (torch.equal at h=16..48 on WH, perf/spd_bh/transition_h.py).

Host-only, mirrored like test_transition_height_bands.py: the engine's copy lives in a closure
inside `Transition.__call__`. Valid up to SEQ_LEN_MORE_CHUNKING tokens (no W chunking below it).
"""
from __future__ import annotations

from unittest import mock

import pytest

from tt_bio import tenstorrent as T

GIB = 2 ** 30
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


def _height(W, c, b8):
    tile = lambda v: -(-int(v) // 32) * 32
    hid = 4 * c
    base = T.TRANSITION_H_CHUNK_SIZE_FAST if b8 and c <= 128 else T.TRANSITION_H_CHUNK_SIZE
    ref = 1024 * 128 * 128 // max(128, c)
    hb = 1088 / 1024 if b8 else 2
    l1_rows = T.TRANSITION_L1_CHUNK_BYTES_PER_CORE * GRID[0] * GRID[1] / (tile(W) * (2 * tile(c) + 2 * hb * tile(hid)))
    return min(max(1, int(base * min(1.0, ref / (W * c)))), max(1, int(l1_rows)))


@pytest.mark.parametrize("W,h_bf16,h_b8", [(736, 16, 28), (1024, 12, 20), (1536, 8, 13)])
def test_msa_transition_c128(wormhole, W, h_bf16, h_b8):
    assert _height(W, 128, False) == h_bf16
    assert _height(W, 128, True) == h_b8


def test_c64_takes_the_fast_base(wormhole):
    assert _height(736, 64, False) == 16 and _height(736, 64, True) == 32


@pytest.mark.parametrize("W", [512, 736, 1024])
def test_pair_c256_unchanged(wormhole, W):
    # the 16-row base and the ratio bind at c=256 whatever the hidden's width
    assert _height(W, 256, True) == _height(W, 256, False)
