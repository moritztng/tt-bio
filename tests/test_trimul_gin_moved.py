"""`trimul_tail.gin_moved` (the trimul_gin lever's kernel) against the route it replaces and float64.

The resident split (`fused_tail(split=2)`) plus two channel moves gives a and b as [B, C, S, S]; gin_moved
writes the same tensors from one kernel. Both are one bf16 rounding of the same fp32 sums, so they must
sit equally close to float64 and differ from each other by at most a bf16 ULP or two, with and without
the pair mask folded into a.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_gin_moved.py
"""
import pytest
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL
from tt_bio.af2 import compute_kernel_config

CZ, C = 256, 128    # protenix-v2's c_z; one chunk of a and of b


def _dev(t, dev):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("n,masked", [(64, True), (160, True), (160, False)])
def test_gin_moved_matches_split_route_and_float64(n, masked, monkeypatch):
    dev = T.get_device()
    monkeypatch.setattr(TTL, "GIN_MOVE", True)
    monkeypatch.setattr(TTL, "EPI", 1)       # the resident split route needs the lean epilogue
    g = torch.Generator().manual_seed(n)
    x = torch.randn(1, n, n, CZ, generator=g).to(torch.bfloat16)
    wp = (torch.randn(CZ, 2 * C, generator=g) / 16).to(torch.bfloat16)
    wg = (torch.randn(CZ, 2 * C, generator=g) / 16).to(torch.bfloat16)
    m = (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16) if masked else None

    d = torch.float64
    x64 = x.to(d).reshape(n * n, CZ)
    ab = ((x64 @ wp.to(d)) * torch.sigmoid(x64 @ wg.to(d))).reshape(1, n, n, 2 * C)
    ref = [ab[..., :C], ab[..., C:]]
    if m is not None:
        ref[0] = ref[0] * m.to(d)[..., None]
    ref = [r.permute(0, 3, 1, 2) for r in ref]

    dx, dwp, dwg, dwpT, dwgT = (_dev(t, dev) for t in (x, wp, wg, wp.t(), wg.t()))
    dm = None if m is None else _dev(m, dev)
    ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
    grid = tuple(T.COMPUTE_GRID_MAIN)

    assert TTL.gin_moved_ok(dx, dwpT, dm)
    moved = TTL.gin_moved(dx, dwpT, dwgT, ckc, grid, mask=dm)
    assert moved is not None
    split = TTL.fused_tail(dx, dx, dwp, dwg, ckc, grid, split=2, mask=dm)
    assert split is not None, TTL.REJECTS
    split_moved = [T._channel_move(o, ttnn.DRAM_MEMORY_CONFIG) for o in split]

    for mv, sp, r in zip(moved, split_moved, ref):
        mv, sp = ttnn.to_torch(mv).to(d), ttnn.to_torch(sp).to(d)
        assert mv.shape == r.shape
        e_mv, e_sp = _rel_rms(mv, r), _rel_rms(sp, r)
        assert e_sp < 1e-2, e_sp
        assert e_mv <= 1.05 * e_sp + 1e-4, (e_mv, e_sp)
        # no isolated gross element hiding under a good rms (the WH HiFi4 erratum looked like this)
        assert (mv - r).abs().max().item() < 0.1
        assert _rel_rms(mv, sp) < 4e-3
