"""`trimul_tail.gin_ragged`'s channel split per cube: where it is taken, and that it changes no output.

qs 2 is chosen only where twice the cubes still fit the grid (BindCraft 2's 224 tokens on Blackhole);
a split reorders no accumulation, so its outputs must equal qs 1 exactly.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_gin_ragged_qs.py
"""
import pytest
import torch
import ttnn

from tt_bio import trimul_tail as TTL


@pytest.mark.parametrize("S,grid,qs", [
    (224, (11, 10), 2),     # Blackhole p300c, 49 cubes on 110 cores
    (224, (13, 10), 2),     # Blackhole p150a
    (195, (8, 9), 1),       # Wormhole AF2: 98 > 72
    (288, (11, 10), 1),
    (288, (13, 10), 1),
    (864, (11, 10), 1),
])
def test_gin_ragged_qs_predicate(S, grid, qs, monkeypatch):
    monkeypatch.setattr(TTL, "GIN_RAGGED_QS", 0)
    St = -(-S // 32)
    for ct2 in (4, 8):
        assert TTL._gin_ragged_qs(ct2, St * St, grid[0] * grid[1]) == qs


@pytest.mark.parametrize("C", [128, 64])
def test_gin_ragged_auto_qs_equals_one_split(C, monkeypatch):
    from tt_bio import mm_generic as MG
    from tt_bio import tenstorrent as T
    from tt_bio.af2 import compute_kernel_config

    dev = T.get_device()
    grid = tuple(T.COMPUTE_GRID_MAIN)
    n, K = 224, 128
    up = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    g = torch.Generator().manual_seed(C)
    x = torch.randn(1, n, n, K, generator=g).to(torch.bfloat16)
    wp, wg = ((torch.randn(K, 2 * C, generator=g) / 11).to(torch.bfloat16) for _ in range(2))
    bp, bg = ((torch.randn(2 * C, generator=g) / 4).to(torch.bfloat16) for _ in range(2))
    m = (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16)
    args = (up(x), up(wp.t()), up(wg.t()), TTL.gin_ragged_bias(bp, bg, n, dev),
            MG.ckc_args(compute_kernel_config()), grid, n)
    res = {}
    for qs in (1, 0):
        monkeypatch.setattr(TTL, "GIN_RAGGED_QS", qs)
        res[qs] = [ttnn.to_torch(t) for t in TTL.gin_ragged(*args, mask=up(m))]
    assert all(torch.equal(a, b) for a, b in zip(res[1], res[0]))
