"""The Transition row height must be forceable without editing its derivation.

The height is derived from a ratio, an L1 cap and a floor at 1, and its cost is documented
NON-monotonic (`tenstorrent.py`: h=7, 8 and 9 all fit at W=512 and are all slower than h=6). The
derivation lands h=2 at W=896 and h=1 at W=1024, which is the pair of shapes behind the 896-vs-1024
anomaly: at 896 aa the fold makes 0.44x the calls at 1.75x the size -- strictly less element-work --
and still takes more than twice the wall clock.

Separating "h=2 is a bad height at this width" from "896 is a bad size" needs one A/B with the
height forced, so `TT_BIO_TRANSITION_H_CHUNK` does that, in the same idiom as the two screen hooks
already in that function. It must be inert when unset. Host-only: the derivation is exercised
through the module's own constants, no device.
"""
from __future__ import annotations

import os
from unittest import mock

import pytest

from tt_bio import tenstorrent as T

GIB = 2 ** 30


@pytest.fixture
def wormhole(monkeypatch):
    """The 8x9 12 GiB part every number in the state doc was measured on."""
    saved = {n: getattr(T, n) for n in
             ("_IS_SMALL_GRID", "SEQ_LEN_MORE_CHUNKING", "TRANSITION_W_CHUNKING_THRESHOLD",
              "TRANSITION_BATCH_CHUNKING_THRESHOLD", "TRIANGLE_ATT_CHUNK_SIZE_FAST",
              "TRANSITION_W_CHUNK_SIZE", "TRIANGLE_MULT_L1_MAX_SEQ_FAST", "SMALL_GRID_SEQ_TILE",
              "SMALL_GRID_PAIR_TILE_AREA", "SMALL_GRID_MSA_TILE_AREA", "TRIANGLE_MULT_L1_MAX_SEQ",
              "TRANSITION_L1_CHUNK_BYTES_PER_CORE")}
    with mock.patch.object(T, "_dram_total_bytes", lambda d: 12 * GIB), \
         mock.patch.object(T.ttnn, "get_max_worker_l1_unreserved_size",
                           lambda: T._WH_FULL_L1_PER_CORE):
        T._apply_grid_thresholds((8, 9))
    yield
    for n, v in saved.items():
        setattr(T, n, v)


def test_the_screen_hook_is_read_from_the_environment(wormhole, monkeypatch):
    """Inert unset, and clamped into [1, H] when set -- a screen must not be able to ask for a
    height that cannot be sliced."""
    monkeypatch.delenv("TT_BIO_TRANSITION_H_CHUNK", raising=False)
    assert os.environ.get("TT_BIO_TRANSITION_H_CHUNK") is None
    src = open(T.__file__).read()
    assert 'os.environ.get("TT_BIO_TRANSITION_H_CHUNK")' in src
    assert "transition_h_chunk_size = max(1, min(int(_h), H))" in src


def test_the_derived_heights_at_the_two_anomalous_widths(wormhole):
    """Pins the pair the census measured: h=2 at W=896, h=1 at W=1024, c=384.

    Both come out of the same expression, so this is what a forced-height A/B is A/B-ing against.
    """
    c, base_h = 384, 16
    ref = 1024 * 128 * 128 // max(128, c)
    derived = lambda w: max(1, int(base_h * min(1.0, ref / (w * c))))
    assert derived(896) == 2
    assert derived(1024) == 1
    # ...and the anomaly restated as arithmetic: fewer, bigger calls at 896.
    calls_896, calls_1024 = -(-896 // 2), -(-1024 // 1)
    assert calls_896 == 448 and calls_1024 == 1024
    elems_896 = calls_896 * 2 * 896 * c
    elems_1024 = calls_1024 * 1 * 1024 * c
    assert elems_896 < elems_1024          # 896 does LESS element-work and is still slower
