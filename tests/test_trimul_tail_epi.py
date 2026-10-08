"""The trimul tail's lean epilogues (`TT_BIO_TRIMUL_TAIL_EPI`) against the production one and float64.

EPI 0 is production's bit-exact order. EPI 1 packs each GEMM pass straight from DST, applies the
sigmoid there and gates with the FPU multiply, so it may differ from 0 by bf16 ULPs and must stay
as close to float64 as 0 is. EPI 2 is 1 with the residual folded in: it writes `z + p * sigmoid(g)`
into `z` itself and returns `z`.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_tail_epi.py
"""
import pytest
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL
from tt_bio.af2 import compute_kernel_config

CZ = 256    # protenix-v2's c_z: the (8, 8) key, the only one F1 serves


def _dev(t, dev):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("n", [64, 160])
def test_epi_matches_production_and_float64(n, monkeypatch):
    dev = T.get_device()
    g = torch.Generator().manual_seed(n)
    bf = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).float()
    xa, xb, z = bf(1, n, n, CZ), bf(1, n, n, CZ), bf(1, n, n, CZ)
    wa, wb = bf(CZ, CZ, sc=CZ ** -0.5), bf(CZ, CZ, sc=CZ ** -0.5)
    d = torch.float64
    ref = (xa.to(d) @ wa.to(d)) * torch.sigmoid(xb.to(d) @ wb.to(d))

    xa_d, xb_d, wa_d, wb_d = _dev(xa, dev), _dev(xb, dev), _dev(wa, dev), _dev(wb, dev)
    ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
    grid = tuple(T.COMPUTE_GRID_MAIN)
    outs = {}
    for epi in (0, 1):
        monkeypatch.setattr(TTL, "EPI", epi)
        y = TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid)
        assert y is not None, TTL.REJECTS
        outs[epi] = ttnn.to_torch(y).float()
        ttnn.deallocate(y)

    e0, e1 = _rel_rms(outs[0].to(d), ref), _rel_rms(outs[1].to(d), ref)
    assert e0 < 1e-2, e0
    assert e1 <= 1.05 * e0 + 1e-4, (e1, e0)
    assert _rel_rms(outs[1], outs[0]) < 4e-3      # under one bf16 ULP on average

    monkeypatch.setattr(TTL, "EPI", 2)
    z_d = _dev(z, dev)
    taken = TTL.RESID_STATS[0]
    y = TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid, resid=z_d)
    assert y is z_d and TTL.RESID_STATS[0] == taken + 1
    y2 = ttnn.to_torch(y).float()
    want = z + outs[1]
    # one bf16 rounding of the sum, plus the product's own rounding that EPI 2 skips
    tol = 2.0 ** -7 * (z.abs() + outs[1].abs()) + 1e-6
    assert ((y2 - want).abs() <= tol).all(), (y2 - want).abs().max().item()
    assert _rel_rms((y2 - z).to(d), ref) <= 1.05 * e0 + 2e-3

    # without a residual EPI 2 runs as 1
    y = TTL.fused_tail(xa_d, xb_d, wa_d, wb_d, ckc, grid)
    assert torch.equal(ttnn.to_torch(y).float(), outs[1])
    for t in (y, z_d, xa_d, xb_d, wa_d, wb_d):
        ttnn.deallocate(t)
