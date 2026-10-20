"""Wormhole sends a trimul to the DRAM route once batch x S_pad^2 passes TRIMUL_WH_DRAM_AREA.

The L1 channel loop narrows its chunk as batch x S^2 grows and pays a fixed set of ops per chunk, so past an area the
DRAM route (gin_moved in-projection, bmm_res einsum) is faster. The memory config and the chunk width must read the
same predicate, or the loop would size an L1 chunk for a DRAM pass. Host-only, no device.
"""
from __future__ import annotations

import pytest
import ttnn

from tt_bio import tenstorrent as T

AREA = 256 * 256


@pytest.fixture
def wh(monkeypatch):
    monkeypatch.setattr(T, "is_wormhole", lambda: True)
    monkeypatch.setattr(T, "_FAST_MODE", False)
    monkeypatch.setattr(T, "TRIANGLE_MULT_L1_MAX_SEQ", 288)
    monkeypatch.setattr(T, "TRIMUL_WH_DRAM_AREA", AREA)


@pytest.mark.parametrize("seq,batch,l1", [(256, 1, True), (288, 1, False), (225, 1, True), (128, 4, True),
                                          (128, 5, False), (96, 5, True), (96, 8, False)])
def test_area_rule(wh, seq, batch, l1):
    assert T._trimul_in_l1(seq, batch) is l1
    dram = T._triangle_mul_memory_config(seq, batch).buffer_type == ttnn.BufferType.DRAM
    assert dram is not l1
    if not l1:
        assert T._trimul_chunk_size(seq, 128, batch) == T.TRIANGLE_MULT_CHUNK_SIZE


def test_seq_only_callers_and_other_arches_keep_the_threshold(wh, monkeypatch):
    assert T._trimul_in_l1(288, None) and not T._trimul_in_l1(320, None)
    monkeypatch.setattr(T, "is_wormhole", lambda: False)
    assert T._trimul_in_l1(288, 8)
    monkeypatch.setattr(T, "is_wormhole", lambda: True)
    monkeypatch.setattr(T, "TRIMUL_WH_DRAM_AREA", 0)
    assert T._trimul_in_l1(288, 8)
