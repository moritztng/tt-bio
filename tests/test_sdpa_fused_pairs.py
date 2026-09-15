"""Which (q_chunk, k_chunk) pairs the fused triangle-attention kernel is offered above 1024
padded tokens, in what order, and that nothing at or below 1024 moves.

`triatt_sdpa.fused_pairs` is the only place that knows the fused kernel's two constraints pull in
opposite directions -- one q chunk per core against a persistent mask CB that grows with q_chunk
and not with k_chunk. A regression there does not fail loudly: the kernel declines, the stock op
serves, and the fold is 2-3x slower on this op with the identical output shape. So the invariants
are asserted rather than left to a perf run to notice.

Opens no device.
"""

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import sdpa_generic as SG                                       # noqa: E402
from tt_bio import triatt_sdpa as TS                                        # noqa: E402

CORES, HEADS, HEAD_DIM = 110, 4, 32


@pytest.fixture
def cap():
    """Set `_Q_SPLIT_MAX_S` and clear the memo around a case, both ways."""
    def _set(value):
        TS.fused_pairs.cache_clear()
        TS._Q_SPLIT_MAX_S = value
    before = TS._Q_SPLIT_MAX_S
    yield _set
    _set(before)


def test_the_above_cap_route_is_strictly_above_the_cap():
    """What keeps every length that folds today byte-identical. `_tri_att_sdpa_at` may consult
    `fused_pairs` only where the stock ladder serves the fused kernel nothing, and that boundary
    is the cap itself -- at and below it the ladder already lands on a fused pair."""
    import inspect

    from tt_bio import tenstorrent as T
    src = inspect.getsource(T._tri_att_sdpa_at)
    assert "q_len > _triatt_sdpa._Q_SPLIT_MAX_S" in src
    # On since the shipped-config fold A/B priced it: 1.1856x on a 200-step 1536 aa fold, 15.3x
    # its own A/A floor. What keeps the flip safe is the line above, not this one -- at and below
    # the cap the route is unreachable, so every length that folds today is byte-identical.
    assert T._SDPA_FUSED_LARGE_S is True, "the above-cap route ships on"
    assert TS._Q_SPLIT_MAX_S == 1024, "the cap moved; the neutrality argument moved with it"


def test_fused_pairs_does_not_read_the_cap(cap):
    """`fused_pairs` enumerates uncapped on purpose -- the cap governs the STOCK ladder, and the
    above-cap route is the one caller allowed past it (`sdpa(..., q_split_cap=0)`). So its answer
    must not move with the cap, at any length."""
    for seq in (512, 768, 1024, 1536, 2208):
        cap(1024)
        shipped = TS.fused_pairs(seq, HEADS, HEAD_DIM, CORES, None)
        cap(4096)
        assert TS.fused_pairs(seq, HEADS, HEAD_DIM, CORES, None) == shipped, seq
        assert (seq <= 1024) or shipped, seq


def test_the_pick_is_the_cheapest_per_core_not_the_widest_k(cap):
    """Widest-k-first was right at padded 864, where every candidate lit up the same core count.
    Above 1024 it forces a q narrow enough to idle cores, and `per_core_cost` picks instead.

    The three numbers are measured interleaved against the incumbent ladder
    (`perf/ttx_a3/ab1536_qb2c2.json`, `perf/bgsdpa/ab1920.json`, `ab2208.json`): widest-k-first
    2.061x / 1.518x / 1.865x, these picks 2.737x / 2.678x / 1.865x, surface best
    2.862x / 2.678x / 1.865x."""
    cap(4096)
    assert TS.fused_pairs(1536, HEADS, HEAD_DIM, CORES, None)[0] == (256, 512)
    assert TS.fused_pairs(1920, HEADS, HEAD_DIM, CORES, None)[0] == (160, 960)
    assert TS.fused_pairs(2208, HEADS, HEAD_DIM, CORES, None)[0] == (96, 736)
    assert TS.fused_pairs(2592, HEADS, HEAD_DIM, CORES, None)[0] == (96, 864)
    # A 32-wide k rescales the accumulator seq/32 times, so it must never lead even though its
    # narrow CB lets a very wide q fit beside it.
    for seq in (1536, 1920, 2208, 2592):
        assert TS.fused_pairs(seq, HEADS, HEAD_DIM, CORES, None)[0][1] > 32, seq


def test_a_length_with_no_usable_divisor_offers_nothing(cap):
    """1856 = 2^6 * 29, so its 32-aligned divisors are 32, 64, 928 and 1856. 928 and 1856 blow
    the persistent mask CB; 64 and 32 leave 29 and 58 q chunks against the 27 a 110-core grid can
    give one each. Nothing fits, and an empty answer is a real answer."""
    cap(4096)
    assert SG.chunk_divisors(1856) == (1856, 928, 64, 32)
    assert TS.fused_pairs(1856, HEADS, HEAD_DIM, CORES, None) == ()


def test_every_offered_pair_divides_and_fits(cap):
    """The two things the kernel refuses on, checked against the model rather than the device."""
    cap(4096)
    for seq in (1152, 1536, 2208, 2304, 2592):
        for qc, kc in TS.fused_pairs(seq, HEADS, HEAD_DIM, CORES, None):
            assert seq % qc == 0 and seq % kc == 0, (seq, qc, kc)
            q_pf = TS.q_parallel_factor(seq, HEADS, qc, CORES, cap=0)
            p = SG.plan_for_shape(seq, HEADS, HEAD_DIM, qc, kc,
                                  split=(max(CORES // (HEADS * q_pf), 1), HEADS, q_pf))
            pers = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
            assert p["q_per_core"] == 1 and not p["use_padded_mask"], (seq, qc, kc)
            assert SG.cb_fits_l1(p, mask_cb_tiles=pers), (seq, qc, kc)


def test_chunk_divisors_are_tile_aligned_and_widest_first():
    assert SG.chunk_divisors(2208) == (2208, 736, 96, 32)
    assert SG.chunk_divisors(1024) == (1024, 512, 256, 128, 64, 32)
    for d in SG.chunk_divisors(2592):
        assert 2592 % d == 0 and d % 32 == 0


def test_env_int_refuses_a_value_it_cannot_parse(monkeypatch):
    from tt_bio.envflags import env_int
    monkeypatch.setenv("TT_BIO_X_INT", "2208")
    assert env_int("TT_BIO_X_INT", 1024) == 2208
    monkeypatch.setenv("TT_BIO_X_INT", "")
    assert env_int("TT_BIO_X_INT", 1024) == 1024
    monkeypatch.setenv("TT_BIO_X_INT", "1o24")
    with pytest.raises(ValueError):
        env_int("TT_BIO_X_INT", 1024)
