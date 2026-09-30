"""The fused-HiFi arm pads a 32 * p axis up to the next length that serves, and nothing else.

At a padded length of 32 * p for a prime p >= 17 the arm has no legal config, so it declined every
call at 544, 608 and 736 and the fallback materialised the [N, 4, N, N] fp32 scores. The route pads
one tile up and slices the rows back. Device-free: the ladder, the pad and the slice are stubs, so
what is tested is the route's decisions -- when it pads, to what, and what it frees.
"""
import types

import pytest

T = pytest.importorskip("tt_bio.tenstorrent")


class _T:
    def __init__(self, n, tag="op"):
        self.shape = (n, 4, n, 32)
        self.dtype, self.tag = "bf16", tag

    def __getitem__(self, idx):
        return ("slice", self.shape[2], idx[2].stop)


@pytest.fixture
def route(monkeypatch):
    st = {"serves": set(), "ladder": [], "freed": []}

    def ladder(q, k, v, bias, scale, one_k_chunk):
        n = int(q.shape[2])
        st["ladder"].append(n)
        return _T(n, "out") if n in st["serves"] else None

    monkeypatch.setattr(T, "_tri_att_hifi_ladder", ladder)
    monkeypatch.setattr(T, "_sdpa_pad_ragged",
                        lambda q, k, v, b, to=0: (_T(to), _T(to), _T(to), _T(to), to - q.shape[2]))
    monkeypatch.setattr(T, "ttnn", types.SimpleNamespace(deallocate=st["freed"].append))
    monkeypatch.setattr(T, "TRIATT_FUSED_HIFI_PADDED", {})
    return st


def _call(n, bias=True):
    x = _T(n)
    return T._tri_att_sdpa_hifi_inner(x, x, x, _T(n, "bias") if bias else None, 1.0)


def test_a_declining_axis_serves_one_tile_up_and_is_sliced_back(route):
    route["serves"] = {640}
    assert _call(608) == ("slice", 640, 608)
    assert route["ladder"] == [608, 640]
    assert T.TRIATT_FUSED_HIFI_PADDED == {608: 640}
    # the four padded operands and the padded output are freed, the caller's operands are not
    assert len(route["freed"]) == 5 and all(t.tag != "bias" for t in route["freed"])


def test_an_axis_that_serves_natively_is_never_padded(route):
    route["serves"] = {576, 608}
    assert _call(576).shape[2] == 576
    assert route["ladder"] == [576] and T.TRIATT_FUSED_HIFI_PADDED == {}


def test_the_pad_is_bounded(route, monkeypatch):
    monkeypatch.setattr(T, "_TRIATT_HIFI_PAD_UP_TILES", 2)
    route["serves"] = {704}
    assert _call(608) is None
    assert route["ladder"] == [608, 640, 672]


def test_zero_tiles_is_the_old_route(route, monkeypatch):
    monkeypatch.setattr(T, "_TRIATT_HIFI_PAD_UP_TILES", 0)
    route["serves"] = {640}
    assert _call(608) is None and route["ladder"] == [608]


def test_no_bias_no_pad(route):
    """Padding without a bias to mask the new keys would hand them a real share of the softmax."""
    route["serves"] = {640}
    assert _call(608, bias=False) is None and route["ladder"] == [608]
