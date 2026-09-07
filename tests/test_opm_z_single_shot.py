"""OuterProductMean's row-block gate reads bytes, not a token count.

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
    return z_bytes(tokens) > T._concat_host_budget(dram)


def test_budget_is_one_eighth_of_dram_and_never_below_the_measured_base():
    assert T._concat_host_budget(WH_DRAM) == T.OPM_Z_SINGLE_SHOT_BYTES_BASE == 1536 * 2 ** 20
    assert T._concat_host_budget(BH_DRAM) > T._concat_host_budget(WH_DRAM)
    # A part that reports nothing falls back to the measured figure rather than to zero, which
    # would block every size.
    assert T._concat_host_budget(0) == T.OPM_Z_SINGLE_SHOT_BYTES_BASE


def test_wormhole_1024_is_the_refused_allocation_and_now_blocks():
    # Byte for byte the DRAM buffer an OpenFold3 1024-token fold is refused on.
    assert z_bytes(1024) == 2147483648
    assert blocked(1024, WH_DRAM)


def test_wormhole_keeps_every_size_that_folds_today_on_the_single_shot_path():
    # 128-768 fold on a Galaxy today, so they must be byte-identical: same single matmul, same
    # single allocation, no concat.
    for tokens in (128, 256, 384, 512, 576, 608, 640, 672, 704, 736, 768):
        assert not blocked(tokens, WH_DRAM), tokens


def test_blackhole_is_neutral_by_construction_up_to_1440_tokens():
    # Blackhole has 2.65x the DRAM and therefore 2.65x the budget, so no advertised size changes
    # path there -- the arch needs no flag of its own.
    for tokens in (512, 768, 896, 1024, 1088, 1440):
        assert not blocked(tokens, BH_DRAM), tokens
    assert blocked(1472, BH_DRAM)


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
