"""`permute_back_pk` (the trimul's back channel move on Wormhole) against torch's permute.

A copy through a 16-bit dest, so every element must match exactly: whole output, a channel slice at
c_off into a wider output, and through `reblock_permute_back`, which routes here on Wormhole.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_permute_back_pk.py
"""
import pytest
import torch
import ttnn

from tt_bio import permute_back_pk as PK
from tt_bio import reblock_permute as RB
from tt_bio import tenstorrent as T


def _dev(t, dev):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _out(n, h, dev, b=1):
    return ttnn.allocate_tensor_on_device(ttnn.Shape([b, n, n, h]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                          ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("n,c", [(64, 64), (160, 128), (256, 256)])
def test_permute_back_pk_exact(n, c):
    dev = T.get_device()
    x = torch.randn(1, c, n, n, generator=torch.Generator().manual_seed(n + c)).to(torch.bfloat16)
    ref = x.permute(0, 2, 3, 1)
    dx = _dev(x, dev)
    grid = tuple(T.COMPUTE_GRID_MAIN)

    out = _out(n, c, dev)
    assert PK.ok(dx, out)
    assert torch.equal(ttnn.to_torch(PK.permute_back_pk(dx, out, grid)), ref)

    wide = _out(n, 2 * c, dev)
    for c_off in (0, c):
        PK.permute_back_pk(dx, wide, grid, c_off=c_off)
    got = ttnn.to_torch(wide)
    assert torch.equal(got[..., :c], ref) and torch.equal(got[..., c:], ref)


def test_reblock_permute_back_routes_to_pk(monkeypatch):
    dev = T.get_device()
    monkeypatch.setattr(RB, "_PBPK", True)
    n, c = 96, 64
    x = torch.randn(1, c, n, n, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    before = RB.STATS_PBPK[0]
    got = ttnn.to_torch(RB.reblock_permute_back(_dev(x, dev), ttnn.DRAM_MEMORY_CONFIG, out=_out(n, c, dev)))
    assert RB.STATS_PBPK[0] == before + 1
    assert torch.equal(got, x.permute(0, 2, 3, 1))


@pytest.mark.parametrize("b,n,c", [(3, 64, 64), (16, 128, 64)])
def test_permute_back_pk_batch_exact(b, n, c):
    """A sample batch (RFD4 `pair_batch`): each design's move lands in its own slab, whole and at a c_off."""
    dev = T.get_device()
    x = torch.randn(b, c, n, n, generator=torch.Generator().manual_seed(b + n)).to(torch.bfloat16)
    ref = x.permute(0, 2, 3, 1)
    dx = _dev(x, dev)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    assert torch.equal(ttnn.to_torch(PK.permute_back_pk(dx, _out(n, c, dev, b), grid)), ref)
    wide = _out(n, 2 * c, dev, b)
    for c_off in (0, c):
        PK.permute_back_pk(dx, wide, grid, c_off=c_off)
    got = ttnn.to_torch(wide)
    assert torch.equal(got[..., :c], ref) and torch.equal(got[..., c:], ref)
