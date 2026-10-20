"""`TraceScratch` keeps its hole fenced when a capture is refused.

A capture that throws must not hand any of the hole to eager work: replays of the traces already
in it would overwrite whatever lands there. Before 30d92d3bc the hole was refilled while the
refusal's traceback still held the failed run's buffers, and an eager tensor allocated next sat
at the hole's first byte (BC2 seed 102 on qb2 card 3, 2026-10-20).

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trace_scratch_hw.py
"""
import gc

import pytest
import ttnn

from tt_bio import tenstorrent as T

pytestmark = pytest.mark.device


def test_a_refused_capture_leaves_the_hole_fenced():
    dev = T.get_device(trace="bc2")
    sc = T.TraceScratch(dev)

    def piece():   # 4 MiB per bank
        return ttnn.empty([sc.banks * 4, 1 << 19], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                          device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def refused():
        held_by_the_frame = piece()   # noqa: F841, allocated in the hole during the capture
        raise RuntimeError("refused")

    with pytest.raises(RuntimeError, match="refused"):
        sc.capture(sc.CHUNK, sc.CHUNK, [refused], settle=gc.collect)
    gc.collect()
    try:
        lo, hi = sc.hole
        assert not lo <= piece().buffer_address() < hi
    finally:
        sc.release()
