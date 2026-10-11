"""The forward move's L1 floor follows work groups per core, and reproduces every measured edge.

Measured (perf/bigswing/reblock_window_band.py): on 110 and 130 cores the move wins from N=288
(Nt=9) and loses at N=256 (Nt=8, 0.952x on 110); on a Wormhole chip's 64 cores it wins from N=195
(Nt=7) and loses at N=192 (Nt=6). Host only, no card: a fake tensor carries the grid.
"""

import pytest

from tt_bio import reblock_permute as R


class _Grid:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Dev:
    _ids = 0

    def __init__(self, gx, gy):
        _Dev._ids += 1
        self._id, self._g = 1000 + _Dev._ids, _Grid(gx, gy)

    def id(self):
        return self._id

    def compute_with_storage_grid_size(self):
        return self._g


class _X:
    def __init__(self, dev):
        self._dev = dev

    def device(self):
        return self._dev


@pytest.mark.parametrize("grid, opens, stays_shut", [
    ((8, 8), [193, 195, 224, 256], [160, 192]),       # Wormhole: Nt=7 wins, Nt=6 loses
    ((11, 10), [257, 288], [224, 256]),               # p300c: Nt=9 wins, Nt=8 loses
    ((13, 10), [257, 288], [224, 256]),               # p150a: same tile edge
])
def test_l1_floor_by_grid(grid, opens, stays_shut):
    x = _X(_Dev(*grid))
    assert all(R._l1_fills_grid(x, n) for n in opens)
    assert not any(R._l1_fills_grid(x, n) for n in stays_shut)


def test_zero_restores_the_plain_edge(monkeypatch):
    monkeypatch.setattr(R, "L1_GROUPS_PER_CORE", 0.0)
    assert not R._l1_fills_grid(_X(_Dev(8, 8)), 352)
