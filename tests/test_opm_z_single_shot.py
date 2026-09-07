"""The MSA track's two single-shot gates read bytes, not a token count.

The gate used to be ``I > SEQ_LEN_MORE_CHUNKING``. That constant belongs to the pair-tensor
paths and was moved 608 -> 1088 on a 12 GiB Galaxy part on a pair-tensor measurement, which
turned OPM's row blocking off for every token count in 640-1088 -- the band where its single-shot
z is largest. These pin the byte gate that replaced it, on both parts, without a device.
"""
import tt_bio.tenstorrent as T


C = D = 32          # OPM projection width for every model on this path
BF16 = 2
WH_DRAM = 12 * 1073741792          # a Galaxy Wormhole chip, as ttnn reports it
BH_DRAM = 8 * 4278190080           # a p150a, 31.875 GiB


def z_bytes(tokens):
    """The single-shot z, (I*C, D*J) bf16, at a square token count."""
    return tokens * C * D * tokens * BF16


def blocked(tokens, dram):
    return z_bytes(tokens) > T._opm_z_single_shot_budget(dram)


def test_budget_is_the_measured_figure_on_the_part_it_was_measured_on():
    # 1 207 959 552 B is the 768-token single-shot z, the largest measured to allocate.
    assert T.OPM_Z_SINGLE_SHOT_BYTES_BASE == z_bytes(768) == 1207959552
    assert T._opm_z_single_shot_budget(WH_DRAM) == T.OPM_Z_SINGLE_SHOT_BYTES_BASE
    assert T._opm_z_single_shot_budget(BH_DRAM) > T._opm_z_single_shot_budget(WH_DRAM)
    # A part that reports nothing falls back to the measured figure rather than to zero, which
    # would block every size.
    assert T._opm_z_single_shot_budget(0) == T.OPM_Z_SINGLE_SHOT_BYTES_BASE


def test_the_refusals_that_set_the_budget_are_the_negative_control():
    # 800 and 832 were refused on this exact allocation, by 832 B and 256 B per bank, with ~300 MB
    # per bank nominally free. So the budget has no margin to spend above 768's figure.
    assert z_bytes(800) == 1310720000 and z_bytes(832) == 1417674752
    assert blocked(800, WH_DRAM) and blocked(832, WH_DRAM)


def test_wormhole_1024_is_the_refused_allocation_and_now_blocks():
    # Byte for byte the DRAM buffer an OpenFold3 1024-token fold is refused on.
    assert z_bytes(1024) == 2147483648
    assert blocked(1024, WH_DRAM)


def test_wormhole_keeps_every_size_that_folds_today_on_the_single_shot_path():
    # 128-768 fold on a Galaxy today, so they must be byte-identical: same single matmul, same
    # single allocation, no concat.
    for tokens in (128, 256, 384, 512, 576, 608, 640, 672, 704, 736, 768):
        assert not blocked(tokens, WH_DRAM), tokens


def test_blackhole_is_neutral_by_construction_across_the_advertised_range():
    # Blackhole has 2.65x the DRAM and therefore 2.65x the budget, so no advertised size changes
    # path there -- the arch needs no flag of its own.
    for tokens in (512, 768, 896, 1024, 1088, 1216):
        assert not blocked(tokens, BH_DRAM), tokens
    assert blocked(1280, BH_DRAM)


def test_the_gate_is_not_the_token_count_it_replaced():
    # The whole defect: 1024 sits BELOW the 1088 the old test compared against, so the old gate
    # declined to block exactly the size that cannot afford not to be blocked.
    assert 1024 < 1088
    assert blocked(1024, WH_DRAM)


def test_row_block_keeps_a_blocked_call_under_the_per_block_budget():
    # The block is derived from the token width, so the per-block matmul result is bounded whatever
    # the token count -- that is the part of the lever that already worked.
    for tokens in (896, 1024, 1536):
        per_row = C * D * tokens * BF16
        rows = max(32, min(T.OPM_CHUNK_SIZE,
                           (T.OPM_Z_BUDGET_BYTES // per_row) // 32 * 32))
        assert rows % 32 == 0 and rows >= 32
        assert rows * per_row <= T.OPM_Z_BUDGET_BYTES, tokens
        # and it is a real partition of the token axis, so every row lands in exactly one block
        assert sum(min(i + rows, tokens) - i for i in range(0, tokens, rows)) == tokens


def test_env_override_exists_for_an_ab_screen():
    T._OPM_Z_SINGLE_SHOT_BYTES = None
    try:
        import os
        os.environ["TT_BIO_OPM_Z_SINGLE_SHOT_BYTES"] = "12345"
        assert T.opm_z_single_shot_bytes() == 12345
    finally:
        os.environ.pop("TT_BIO_OPM_Z_SINGLE_SHOT_BYTES", None)
        T._OPM_Z_SINGLE_SHOT_BYTES = None


# --- PairWeightedAveraging's MSA-depth block ---------------------------------------------------
C_M = 64            # OF3 msa_module c_m; the shape PWA projects each head's output up to


def pwa_bytes(depth, tokens):
    """The [depth, tokens, c_m] tensor PWA allocates per head, unblocked."""
    return depth * tokens * C_M * BF16


def test_pwa_base_is_the_768_tensor_and_800_is_its_negative_control():
    # 768 x 14191 allocates and the fold runs on; 800 x 14191 is refused on this exact buffer.
    assert T.PWA_SINGLE_SHOT_BYTES_BASE == pwa_bytes(14191, 768) == 1395032064
    assert pwa_bytes(14191, 800) == 1453158400
    assert T._pwa_single_shot_budget(WH_DRAM) == T.PWA_SINGLE_SHOT_BYTES_BASE
    assert T._pwa_single_shot_budget(0) == T.PWA_SINGLE_SHOT_BYTES_BASE




def pwa_blk(depth, tokens, dram=WH_DRAM):
    return T.pwa_depth_block(depth, tokens, C_M, T._pwa_single_shot_budget(dram))


def test_pwa_is_single_shot_wherever_openfold3_folds_today():
    # `depth` back means today's exact path: one layer_norm, one accumulator, no concat.
    for tokens in (128, 256, 384, 512, 576, 640, 704, 736, 768):
        assert pwa_blk(14191, tokens) == 14191, tokens


def test_pwa_blocks_every_rung_that_was_refused():
    for tokens in (800, 832, 896, 960, 1024):
        blk = pwa_blk(14191, tokens)
        assert blk < 14191, tokens
        assert blk % 32 == 0 and blk >= 32
        # the block is derived from the token width, so the per-block tensor is bounded whatever
        # the token count -- the fixed-constant mistake OPM's row block already made at 992
        assert pwa_bytes(blk, tokens) <= T.PWA_DEPTH_BUDGET_BYTES, tokens


def test_pwa_depth_blocks_are_a_partition_of_the_alignment():
    for tokens in (800, 1024):
        blk = pwa_blk(14191, tokens)
        covered = sum(min(s + blk, 14191) - s for s in range(0, 14191, blk))
        assert covered == 14191, (tokens, blk)


def test_pwa_is_neutral_on_blackhole_across_the_whole_advertised_range():
    # 2.99 GiB there, and production caps OF3's alignment at 16384 rows, so a p150a never blocks.
    for tokens in (512, 768, 1024):
        assert pwa_blk(16384, tokens, BH_DRAM) == 16384, tokens
    # and the Wormhole part it was measured on does block there, so this is not a vacuous pass
    assert pwa_blk(16384, 1024) < 16384


def test_both_gates_scale_by_the_same_dram_fraction():
    # One rule, two measured bases: a part with more DRAM widens both together, so no single
    # lever can drift away from the other.
    assert T._single_shot_budget(0, WH_DRAM) == WH_DRAM * 3 // 32
    for base in (T.OPM_Z_SINGLE_SHOT_BYTES_BASE, T.PWA_SINGLE_SHOT_BYTES_BASE):
        assert T._single_shot_budget(base, BH_DRAM) == BH_DRAM * 3 // 32 > base
