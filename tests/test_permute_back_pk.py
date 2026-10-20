"""`permute_back_pk` and `permute_fwd_pk` (the trimul channel moves, packer re-tiling) against torch's permute.

A copy through a 16-bit dest, so every element must match exactly: whole output, a channel slice at
c_off, and through the `reblock_permute` / `reblock_permute_back` entry points that route to them.

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


def _out(n, h, dev):
    return ttnn.allocate_tensor_on_device(ttnn.Shape([1, n, n, h]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
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


@pytest.mark.parametrize("n,c", [(64, 64), (160, 128), (224, 256)])
def test_permute_fwd_pk_exact(n, c):
    dev = T.get_device()
    x = torch.randn(1, n, n, 4 * c, generator=torch.Generator().manual_seed(n + c)).to(torch.bfloat16)
    dx = _dev(x, dev)
    grid = tuple(T.COMPUTE_GRID_MAIN)
    assert torch.equal(ttnn.to_torch(PK.permute_fwd_pk(dx, grid)), x.permute(0, 3, 1, 2))
    for c_off in (0, c, 3 * c):
        got = ttnn.to_torch(PK.permute_fwd_pk(dx, grid, C=c, c_off=c_off))
        assert torch.equal(got, x[..., c_off:c_off + c].permute(0, 3, 1, 2))


def test_reblock_permute_routes_to_fwd_pk(monkeypatch):
    dev = T.get_device()
    n, c = 96, 64
    x = torch.randn(1, n, n, c, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    calls = []
    real = PK.permute_fwd_pk
    monkeypatch.setattr(RB, "_PFPK", True)
    monkeypatch.setattr(PK, "permute_fwd_pk", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = ttnn.to_torch(RB.reblock_permute(_dev(x, dev), ttnn.DRAM_MEMORY_CONFIG))
    assert calls and torch.equal(got, x.permute(0, 3, 1, 2))
