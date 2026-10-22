"""`bindcraft2.resident_blocks`: how many Evoformer blocks keep their tape instead of a checkpoint."""
import pytest

B = pytest.importorskip("tt_bio.bindcraft2")

WH = 12.885e9  # a Wormhole Galaxy chip's DRAM as the allocator reports it
BH = B.P150A_DRAM_BYTES


@pytest.mark.parametrize("tokens, held, refused", [(224, 48, None), (256, 36, 40), (288, None, 32),
                                                    (320, 24, 26), (352, 20, None)])
def test_auto_stays_inside_the_measured_blackhole_brackets(tokens, held, refused):
    # Real BindCraft 2 rounds on a Blackhole chip, 2026-10-19 (sweep13/14 of bc4-bet-tape).
    k = B.resident_blocks("fast", tokens, BH, requested="auto")
    if refused is not None:
        assert k < refused
    if held is not None:
        assert k >= held - 2


def test_the_law_stays_under_the_wormhole_refusal():
    # .114, 224 tokens, the 1.03 GB/block tape of 2026-10-13: 11 fit, 12 refused.
    pair = 256 * 224 ** 2
    room = WH * B._MODE_USABLE - B._MODE_BASE_BYTES - B._RESIDENT_ROUND_PAIR_TENSORS * pair
    assert room // (1.03e9) < 12


@pytest.mark.parametrize("card", [WH, 24e9])
def test_auto_keeps_no_tape_on_a_board_without_a_resident_ladder(card):
    assert B.resident_blocks("fast", 224, card, requested="auto") == 0
    assert B.resident_blocks("fast", 224, card, requested="8") == 8


def test_auto_shrinks_with_the_token_axis_and_ends_at_zero():
    ks = [B.resident_blocks("fast", n, BH, requested="auto") for n in (224, 256, 320, 512, 864)]
    assert ks == sorted(ks, reverse=True) and ks[0] == B.EVOFORMER_BLOCKS and ks[-1] == 0


def test_auto_is_zero_in_the_slower_modes_and_for_an_interleaved_trajectory():
    assert B.resident_blocks("lean", 224, BH, requested="auto") == 0
    assert B.resident_blocks("offload", 224, BH, requested="auto") == 0
    assert B.resident_blocks("fast", 224, BH, slot="t1", requested="auto") == 0


def test_a_pinned_count_is_clamped_to_the_stack():
    assert B.resident_blocks("lean", 224, WH, requested="8") == 8
    assert B.resident_blocks("fast", 224, WH, requested="99") == B.EVOFORMER_BLOCKS
    assert B.resident_blocks("fast", 224, WH, requested="0") == 0


def test_the_structure_module_on_card_comes_off_the_room():
    # 224 tokens on a Blackhole chip: 48 blocks fill the card to within 0.32 GB, and the on-card
    # structure module holds 0.75 GB at its peak while the tape is live. Measured on qb2 chip 3,
    # 2026-10-21: with 48 resident blocks the stack refused a 0.9 MB buffer in the 4th iteration.
    from tt_bio import af2_structure as S
    full = B.resident_blocks("fast", 224, BH, requested="auto")
    beside = B.resident_blocks("fast", 224, BH, requested="auto", beside=S.device_bytes(224))
    assert beside < full
    pair = 256 * 224 ** 2
    room = BH * B._MODE_USABLE - B._MODE_BASE_BYTES - B._RESIDENT_ROUND_PAIR_TENSORS * pair
    assert beside * B._RESIDENT_BLOCK_PAIR_TENSORS * pair + S.device_bytes(224) <= room


def test_the_default_at_224_on_a_blackhole_chip():
    # PD-L1 (195 residues) pads to 224, a size users run today. With the structure module on the
    # card beside the tape the law gives 47 blocks, and the main tree of 2026-10-21 refused its
    # first backward there: masked OPM's reshape left one dead [224, 224, 128] view per resident
    # block on the card (+0.50 GB at 40 blocks), on top of 0.27 GB of slack. The tape now
    # releases dead view groups, and 40 blocks peak at 25.44 GB over the first two gradient
    # calls on qb2 chip 2 (was 27.44 GB). This pins what `auto` offers there, so a change to
    # the law that moves it has to be measured again at 224.
    from tt_bio import af2_structure as S
    card = 34_225_520_128  # a p300c chip's DRAM as the allocator reports it
    assert B._measured_board(card)[2] == "p150a"
    k = B.resident_blocks("fast", 224, card, requested="auto", beside=S.device_bytes(224))
    assert k == 47
    pair = 256 * 224 ** 2
    peak = (B._MODE_BASE_BYTES + B._RESIDENT_ROUND_PAIR_TENSORS * pair + S.device_bytes(224)
            + k * B._RESIDENT_BLOCK_PAIR_TENSORS * pair)
    assert peak <= card * B._MODE_USABLE
