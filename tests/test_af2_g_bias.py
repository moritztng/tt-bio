"""`AF2PairBlock.tri_att_g_in_matmul`: which triangle-attention biases ride in their matmul.

The blocks are built before `bindcraft2.fast_round()` arms the switch, so it has to be read when
the block runs, and it must not leak: off, both attentions keep exactly the "o" form AF2-IG's tap
gate was scored on. Device-free: the ops are stubs.
"""
import pytest

A = pytest.importorskip("tt_bio.af2")


class _Att:
    bias_in_matmul = frozenset({"o"})

    def __call__(self, z, mask):
        return None


def _block():
    b = object.__new__(A.AF2PairBlock)
    b.evoformer_order, b.skip, b.substitute = True, (), ()
    b.tri_att_start, b.tri_att_end = _Att(), _Att()
    for name in ("tri_mul_out", "tri_mul_in", "pair_transition"):
        setattr(b, name, lambda *a: None)
    b._residual = lambda z, u: z
    return b


@pytest.mark.parametrize("armed", [False, True])
def test_the_g_bias_rides_in_matmul_only_when_armed(monkeypatch, armed):
    monkeypatch.setattr(A.AF2PairBlock, "tri_att_g_in_matmul", armed)
    b = _block()
    b("z")
    want = {"g", "o"} if armed else {"o"}
    assert b.tri_att_start.bias_in_matmul == want and b.tri_att_end.bias_in_matmul == want


def test_fast_round_arms_it_and_puts_it_back():
    """Blackhole arms it; Wormhole keeps it off (`bindcraft2._BLACKHOLE_ONLY`)."""
    B = pytest.importorskip("tt_bio.bindcraft2")
    from tt_bio import tenstorrent
    assert A.AF2PairBlock.tri_att_g_in_matmul is False
    with B.fast_round():
        assert A.AF2PairBlock.tri_att_g_in_matmul is not tenstorrent.is_wormhole()
    assert A.AF2PairBlock.tri_att_g_in_matmul is False
