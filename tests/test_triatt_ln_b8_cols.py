"""`triatt_ln.b8_cols`: every output tile is owned by exactly one column group, each group's bfp8 run starts its
row and ends on a DST block (or the group's end), and the bfp8 tiles are exactly q|k|v (|g)."""
import pytest

pytest.importorskip("ttnn")
from tt_bio import triatt_ln as TL  # noqa: E402


@pytest.mark.parametrize("H,g,gate", [(12, 1, False), (12, 1, True), (8, 1, False), (12, 2, False), (12, 2, True),
                                      (8, 2, False), (8, 2, True)])
def test_cols_cover_once(H, g, gate):
    nwf = 4 * H + 1
    nw = -(-nwf // g)
    cols = TL.b8_cols(g, nw, H, gate)
    assert cols is not None and len(cols) == g
    owner, b8 = [0] * nwf, set()
    for n0, nq, end in cols:
        assert end - n0 <= nw and (nq % 4 == 0 or nq == nw)
        for n in range(n0, end):
            owner[n] += 1
        b8 |= set(range(n0, min(n0 + nq, end)))
    assert owner == [1] * nwf
    assert b8 == set(range((4 if gate else 3) * H))


def test_declines():
    assert TL.b8_cols(1, 33, 9) is None            # 3H = 27 straddles a DST block
    assert TL.b8_cols(2, 19, 6) is None            # H = 6: group 1's v run ends mid-block
    assert TL.b8_cols(4, 13, 12) is None
