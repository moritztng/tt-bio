"""Transition T3 host side: CB plan fits L1 on both parts, the core split covers every row tile once, and the
lever is named but in no mode until graded. The device A/B is perf/spd_trimul/t3.py."""
import pytest

ttnn = pytest.importorskip("ttnn")
from tt_bio import tenstorrent as T, transition_t3 as T3


def test_cb_plan_fits_l1():
    for kt, ht in T3.KEYS:
        assert T3.cb_bytes(kt, ht, True) <= 0.85 * 1_461_760
        assert T3.cb_bytes(kt, ht, True) <= 0.85 * 1_499_136 * 0.98


@pytest.mark.parametrize("nu,cores", [(8192, 110), (8192, 130), (2048, 110), (7, 110)])
def test_split_covers_each_row_tile_once(nu, cores):
    parts = T3.split(nu, cores)
    seen = [u for u0, n in parts for u in range(u0, u0 + n)]
    assert seen == list(range(nu))
    assert max(n for _, n in parts) - min(n for _, n in parts) <= 1


def test_lever_named_and_ungraded():
    assert "transition_t3" in T.LEVERS
    assert "transition_t3" in T.UNGRADED_LEVERS
    assert "transition_t3" not in T.NORMAL_LEVERS | T.FAST_LEVERS


def test_off_declines_without_counting():
    T3.STATS[:] = [0, 0]
    prev, T._LEVERS = T._LEVERS, frozenset()
    try:
        assert T3.eligible(None, None, None, None, None) == "off"
    finally:
        T._LEVERS = prev
    assert T3.STATS == [0, 0]
