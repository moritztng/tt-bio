"""The guard for K6: which (q_chunk, k_chunk) pairs the fused triangle-attention kernel is
offered above 1024 padded tokens, and that today's default offers none.

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


def test_the_shipped_cap_offers_nothing_above_1024(cap):
    """The finding this whole lever rests on. The default must keep it that way: K6's branch is
    off, and even with the flag on it would have no pair to try."""
    cap(1024)
    for seq in (1056, 1280, 1536, 2048, 2208, 2592):
        assert TS.fused_pairs(seq, HEADS, HEAD_DIM, CORES, None) == (), seq
    assert TS._Q_SPLIT_MAX_S == 1024, "the shipped cap moved; K6 is no longer opt-in"


def test_raising_the_cap_offers_the_widest_k_first(cap):
    cap(4096)
    assert TS.fused_pairs(2208, HEADS, HEAD_DIM, CORES, None)[0] == (96, 736)
    assert TS.fused_pairs(2592, HEADS, HEAD_DIM, CORES, None)[0] == (96, 864)
    # One k chunk over the whole sequence needs no online-softmax rescale at all, so where it
    # fits it is both the fastest and the closest arm to a torch reference (K5 at padded 864).
    assert TS.fused_pairs(1536, HEADS, HEAD_DIM, CORES, None)[0] == (128, 1536)


def test_at_or_below_1024_the_cap_changes_nothing(cap):
    """Neutrality below the cap is by construction, and this is the assertion of it: a raised cap
    must not move a length that already serves fused today."""
    for seq in (512, 768, 896, 1024):
        cap(1024)
        shipped = TS.fused_pairs(seq, HEADS, HEAD_DIM, CORES, None)
        cap(4096)
        assert TS.fused_pairs(seq, HEADS, HEAD_DIM, CORES, None) == shipped, seq


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
            q_pf = TS.q_parallel_factor(seq, HEADS, qc, CORES)
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
