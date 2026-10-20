"""`trimul_tail.gin_moved` under trimul_gin_o8 (a and b written as bfp8_b) against its own bf16 output and float64.

The bfp8 writer moves whole 16-value face rows plus their shared exponents, so a wrong gather shows as O(1) errors
while a right one differs from the bf16 output by one bfp8 rounding: as far as the host's bfp8 rounding of that bf16
output is, give or take the packer's rounding mode. Masked and unmasked, one launch and ESMFold2's two channel slices.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_trimul_gin_o8.py
"""
import pytest
import torch
import ttnn

from tt_bio import mm_generic as MG
from tt_bio import tenstorrent as T
from tt_bio import trimul_tail as TTL
from tt_bio.af2 import compute_kernel_config


def _dev(t, dev, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("n,cz,c,masked", [(160, 256, 128, True), (160, 256, 128, False), (128, 256, 256, False)])
def test_gin_o8_is_one_bfp8_rounding_of_the_bf16_output(n, cz, c, masked, monkeypatch):
    dev = T.get_device()
    monkeypatch.setattr(T, "_TRUNK_MATH_FIDELITY", "hifi3")
    monkeypatch.setattr(TTL, "GIN_MOVE", True)
    g = torch.Generator().manual_seed(n + c)
    x = torch.randn(1, n, n, cz, generator=g).to(torch.bfloat16)
    wp = (torch.randn(cz, 2 * c, generator=g) / 16).to(torch.bfloat16)
    wg = (torch.randn(cz, 2 * c, generator=g) / 16).to(torch.bfloat16)
    m = (torch.rand(1, n, n, generator=g) > 0.1).to(torch.bfloat16) if masked else None

    d = torch.float64
    x64 = x.to(d).reshape(n * n, cz)
    ab = ((x64 @ wp.to(d)) * torch.sigmoid(x64 @ wg.to(d))).reshape(1, n, n, 2 * c)
    ref = [ab[..., :c], ab[..., c:]]
    if m is not None:
        ref[0] = ref[0] * m.to(d)[..., None]
    ref = [r.permute(0, 3, 1, 2) for r in ref]

    dx = _dev(x, dev)
    dm = None if m is None else _dev(m, dev)
    ckc = MG.ckc_args(T.trunk_compute_kernel_config(compute_kernel_config()))
    grid = tuple(T.COMPUTE_GRID_MAIN)
    ns = TTL.gin_move_split(dx, c, dm)
    assert ns is not None
    # The weights as gin_moved's channel slices take them: slice s holds rows s of each half's n parts.
    wpT, wgT = wp.t(), wg.t()
    part = lambda w, s: torch.cat([w[h * c + s * c // ns: h * c + (s + 1) * c // ns] for h in (0, 1)])  # noqa: E731
    dwpT = [_dev(part(wpT, s), dev) for s in range(ns)]
    dwgT = [_dev(part(wgT, s), dev) for s in range(ns)]

    outs = {}
    for o8 in (False, True):
        monkeypatch.setattr(TTL, "_gin_o8", lambda o8=o8: o8)
        r = TTL.gin_moved(dx, dwpT, dwgT, ckc, grid, mask=dm)
        assert r is not None
        assert all(t.dtype == (ttnn.bfloat8_b if o8 else ttnn.bfloat16) for t in r)
        outs[o8] = [ttnn.to_torch(t).to(d) for t in r]

    for b16, b8, rf in zip(outs[False], outs[True], ref):
        host8 = ttnn.to_torch(_dev(b16.to(torch.bfloat16), dev, ttnn.bfloat8_b)).to(d)
        e_host = _rel_rms(host8, b16)
        assert _rel_rms(b8, b16) <= 1.5 * e_host + 1e-4, (_rel_rms(b8, b16), e_host)
        assert (b8 - b16).abs().max().item() <= 4 * (host8 - b16).abs().max().item() + 1e-3
        assert _rel_rms(b8, rf) <= 1.2 * _rel_rms(host8, rf) + 1e-4
