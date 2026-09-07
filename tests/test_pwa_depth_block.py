"""PairWeightedAveraging's MSA-depth block reads bytes, and the bytes are measured.

PWA projects each head's gated output to the full [depth, tokens, c_m] width and allocates a
fresh one per head beside an accumulator of the same shape. Above the largest such tensor this
part has been MEASURED to place it blocks the depth axis instead; below it, nothing changes.
These pin that budget on both parts, and the partition, without a device.

OuterProductMean's z answers to a refusal rather than to a byte budget -- Boltz-2 folds 1024
today with the byte-identical 2 GiB tensor -- and lives in test_opm_dram_narrow.py.
"""
import os

import tt_bio.tenstorrent as T


BF16 = 2
C_M = 64            # OF3 msa_module c_m; the shape PWA projects each head's output up to
WH_DRAM = 12 * 1073741792          # a Galaxy Wormhole chip, as ttnn reports it
BH_DRAM = 8 * 4278190080           # a p150a, 31.875 GiB


def pwa_bytes(depth, tokens):
    """The [depth, tokens, c_m] tensor PWA allocates per head, unblocked."""
    return depth * tokens * C_M * BF16


def pwa_blk(depth, tokens, dram=WH_DRAM):
    return T.pwa_depth_block(depth, tokens, C_M, T._pwa_single_shot_budget(dram))


def test_base_is_the_768_tensor_and_800_is_its_negative_control():
    # 768 x 14191 allocates and the fold runs on; 800 x 14191 is refused on this exact buffer,
    # 121 098 240 B/bank against a 60 549 120 B largest free block.
    assert T.PWA_SINGLE_SHOT_BYTES_BASE == pwa_bytes(14191, 768) == 1395032064
    assert pwa_bytes(14191, 800) == 1453158400
    assert T._pwa_single_shot_budget(WH_DRAM) == T.PWA_SINGLE_SHOT_BYTES_BASE
    assert T._pwa_single_shot_budget(BH_DRAM) > T._pwa_single_shot_budget(WH_DRAM)
    # A part that reports nothing falls back to the measured figure rather than to zero, which
    # would block every size.
    assert T._pwa_single_shot_budget(0) == T.PWA_SINGLE_SHOT_BYTES_BASE


def test_single_shot_wherever_openfold3_folds_today():
    # `depth` back means today's exact path: one layer_norm, one accumulator, no concat.
    for tokens in (128, 256, 384, 512, 576, 640, 704, 736, 768):
        assert pwa_blk(14191, tokens) == 14191, tokens


def test_blocks_every_rung_that_was_refused():
    for tokens in (800, 832, 896, 960, 1024, 1088):
        blk = pwa_blk(14191, tokens)
        assert blk < 14191, tokens
        assert blk % 32 == 0 and blk >= 32
        # the block is derived from the token width, so the per-block tensor is bounded whatever
        # the token count -- the fixed-constant mistake OPM's row block already made at 992
        assert pwa_bytes(blk, tokens) <= T.PWA_DEPTH_BUDGET_BYTES, tokens


def test_depth_blocks_are_a_partition_of_the_alignment():
    for tokens in (800, 1024):
        blk = pwa_blk(14191, tokens)
        bounds = [(s, min(s + blk, 14191)) for s in range(0, 14191, blk)]
        assert bounds[0][0] == 0 and bounds[-1][1] == 14191
        assert all(b[0] == a[1] for a, b in zip(bounds, bounds[1:])), tokens
        assert sum(e - s for s, e in bounds) == 14191


def test_a_shallow_alignment_never_blocks_however_wide_the_tokens():
    # The budget is on the whole tensor, so the models on this path that carry a small MSA --
    # every one of them but OpenFold3 and OpenBind -- keep the single-shot path at any size.
    for depth in (1, 8, 64, 512):
        for tokens in (512, 1024, 1088, 1536):
            assert pwa_blk(depth, tokens) == depth, (depth, tokens)


def test_neutral_on_blackhole_across_the_whole_advertised_range():
    # 2.99 GiB there, and production caps OF3's alignment at 16384 rows, so a p150a never blocks
    # and needs no arch flag of its own.
    for tokens in (512, 768, 896, 1024, 1088):
        assert pwa_blk(16384, tokens, BH_DRAM) == 16384, tokens
    # and the Wormhole part it was measured on does block there, so this is not a vacuous pass
    assert pwa_blk(16384, 1024) < 16384


def test_the_budget_scales_by_the_shared_dram_fraction():
    assert T._single_shot_budget(0, WH_DRAM) == WH_DRAM * 3 // 32
    assert (T._single_shot_budget(T.PWA_SINGLE_SHOT_BYTES_BASE, BH_DRAM)
            == BH_DRAM * 3 // 32 > T.PWA_SINGLE_SHOT_BYTES_BASE)


def test_env_override_exists_for_an_ab_screen():
    T._PWA_SINGLE_SHOT_BYTES = None
    try:
        os.environ["TT_BIO_PWA_SINGLE_SHOT_BYTES"] = "12345"
        assert T.pwa_single_shot_bytes() == 12345
    finally:
        os.environ.pop("TT_BIO_PWA_SINGLE_SHOT_BYTES", None)
        T._PWA_SINGLE_SHOT_BYTES = None
