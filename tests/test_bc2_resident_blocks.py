"""`bindcraft2.resident_blocks`: how many Evoformer blocks keep their tape instead of a checkpoint."""
import pytest

B = pytest.importorskip("tt_bio.bindcraft2")

WH = 12.885e9  # a Wormhole Galaxy chip's DRAM as the allocator reports it


def test_auto_matches_the_measured_wormhole_fit():
    # .114, 224 tokens: 11 resident blocks fit beside a fast round and 12 ran out of memory.
    assert 8 <= B.resident_blocks("fast", 224, WH, requested="auto") <= 11


def test_auto_shrinks_with_the_token_axis_and_ends_at_zero():
    ks = [B.resident_blocks("fast", n, WH, requested="auto") for n in (224, 256, 320, 512)]
    assert ks == sorted(ks, reverse=True) and ks[-1] == 0


def test_auto_is_zero_in_the_slower_modes_and_for_an_interleaved_trajectory():
    assert B.resident_blocks("lean", 224, WH, requested="auto") == 0
    assert B.resident_blocks("offload", 224, WH, requested="auto") == 0
    assert B.resident_blocks("fast", 224, WH, slot="t1", requested="auto") == 0


def test_a_pinned_count_is_clamped_to_the_stack():
    assert B.resident_blocks("lean", 224, WH, requested="8") == 8
    assert B.resident_blocks("fast", 224, WH, requested="99") == B.EVOFORMER_BLOCKS
    assert B.resident_blocks("fast", 224, WH, requested="0") == 0
