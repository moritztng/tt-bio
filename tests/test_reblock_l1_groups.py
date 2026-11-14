"""The forward move's L1 floor follows work groups per core, and reproduces every measured edge.

Measured (perf/bigswing/reblock_window_band.py): on 110 cores the move wins from Nt=9 and loses at
Nt=8 (N=256, 0.952x); on a 130-core p150a Nt=9 is break-even and Nt=10 wins; on a Wormhole chip's 64
cores (72 on a Galaxy) it wins from Nt=7 (N=195) and loses at Nt=6 (N=192). Host only, no card: a fake tensor
carries the grid. The p150a's N=288 stays open through the plain L1_N_MIN edge, not this rule.
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
    ((8, 9), [193, 195, 224, 256], [160, 192]),       # Galaxy Wormhole, 72 cores: Nt=7 still wins
    ((11, 10), [257, 288], [224, 256]),               # p300c: Nt=9 wins, Nt=8 loses
    ((13, 10), [289, 320], [256, 257, 288]),          # p150a: Nt=9 break-even, Nt=10 wins
])
def test_l1_floor_by_grid(grid, opens, stays_shut):
    x = _X(_Dev(*grid))
    assert all(R._l1_fills_grid(x, n) for n in opens)
    assert not any(R._l1_fills_grid(x, n) for n in stays_shut)


def test_zero_restores_the_plain_edge(monkeypatch):
    monkeypatch.setattr(R, "L1_GROUPS_PER_CORE", 0.0)
    assert not R._l1_fills_grid(_X(_Dev(8, 8)), 352)


@pytest.mark.parametrize("grid, opens, stays_shut", [
    ((8, 9), [195, 224, 256], [160, 192]),            # Galaxy Wormhole: BindCraft 2's DRAM pair at 224
    ((13, 10), [224, 256, 320], [192, 225, 255]),     # p150a: tile-aligned 224 measured, ragged stays shut
    ((11, 10), [224, 256], [192, 200]),               # p150a qb1 card 0, the measured grid
])
def test_dram_window_by_grid(grid, opens, stays_shut):
    x = _X(_Dev(*grid))
    assert all(R._dram_open(x, n) for n in opens)
    assert not any(R._dram_open(x, n) for n in stays_shut)


def test_dram_fill_off_restores_256(monkeypatch):
    monkeypatch.setattr(R, "DRAM_FILL", False)
    x = _X(_Dev(8, 9))
    assert not R._dram_open(x, 224) and R._dram_open(x, 256)


class _XW(_X):
    """A fused [1, N, N, 4C] bf16 projection, interleaved in DRAM, on a fake grid."""

    def __init__(self, dev, N, C=128):
        super().__init__(dev)
        self.shape, self.dtype, self.layout = [1, N, N, 4 * C], R._DTYPE, R.ttnn.TILE_LAYOUT

    def memory_config(self):
        return R.ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("grid, opens, stays_shut", [
    ((8, 9), [195, 224, 256], [192]),                 # Galaxy Wormhole: open by grid fill already
    ((11, 10), [224, 256], [192, 200]),               # p150a: BindCraft 2's gated trimul move at 224
])
def test_gated_dram_window_follows_the_plain_one(grid, opens, stays_shut):
    dev, D = _Dev(*grid), R.ttnn.DRAM_MEMORY_CONFIG
    assert all(R.eligible_gated(_XW(dev, n), 128, D) for n in opens)
    assert not any(R.eligible_gated(_XW(dev, n), 128, D) for n in stays_shut)


def test_gated_fill_off_restores_256(monkeypatch):
    monkeypatch.setattr(R, "DRAM_FILL_GATED", False)
    dev, D = _Dev(11, 10), R.ttnn.DRAM_MEMORY_CONFIG
    assert not R.eligible_gated(_XW(dev, 224), 128, D) and R.eligible_gated(_XW(dev, 256), 128, D)


@pytest.mark.parametrize("n, c, opens", [
    (128, 256, True), (128, 128, True), (96, 256, True), (192, 64, True), (256, 32, True),  # measured wins
    (128, 64, False), (160, 64, False), (224, 32, False), (96, 128, False),                  # measured losses
])
def test_gated_l1_opens_by_slice_size_on_110_cores(n, c, opens):
    assert R.eligible_gated(_XW(_Dev(11, 10), n, c), c, R.ttnn.L1_MEMORY_CONFIG) is opens


def test_gated_l1_size_rule_only_on_the_measured_grid():
    for grid in ((8, 9), (13, 10)):
        assert not R.eligible_gated(_XW(_Dev(*grid), 128, 256), 256, R.ttnn.L1_MEMORY_CONFIG)
