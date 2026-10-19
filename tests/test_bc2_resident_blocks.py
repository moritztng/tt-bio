"""`bindcraft2.resident_blocks`: how many Evoformer blocks keep their tape instead of a checkpoint."""
import pytest

B = pytest.importorskip("tt_bio.bindcraft2")

WH = 12.885e9  # a Wormhole Galaxy chip's DRAM as the allocator reports it
BH = B.P150A_DRAM_BYTES


@pytest.mark.parametrize("tokens, held, refused", [(224, 48, None), (256, 36, 40), (288, None, 32),
                                                    (320, 24, 28)])
def test_auto_stays_inside_the_measured_blackhole_brackets(tokens, held, refused):
    # Real BindCraft 2 rounds on a Blackhole chip, 2026-10-19 (sweep13/sweep14 of bc4-bet-tape).
    k = B.resident_blocks("fast", tokens, BH, requested="auto")
    if refused is not None:
        assert k < refused
    if held is not None:
        assert k >= held - 2


def test_auto_on_wormhole_stays_under_the_measured_refusal():
    # .114, 224 tokens, the 1.03 GB/block tape of 2026-10-13: 11 fit, 12 refused. Today's tape is
    # 50.5 pair tensors a block, so the same card holds more; the law with that day's tape stays
    # under the refusal.
    pair = 256 * 224 ** 2
    room = WH * B._MODE_USABLE - B._MODE_BASE_BYTES - B._RESIDENT_ROUND_PAIR_TENSORS * pair
    assert room // (1.03e9) < 12
    assert 0 < B.resident_blocks("fast", 224, WH, requested="auto") <= B.EVOFORMER_BLOCKS


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
