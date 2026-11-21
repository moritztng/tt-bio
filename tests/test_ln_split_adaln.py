"""`ln_split.adaln` (the token DiT's AdaLN as one kernel, scale and shift read as column windows) against float64
and against the route it replaces, `ttnn.layer_norm` (or `ttnn.rms_norm`, rms=True) then `addcmul_at` on the same
windows.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_ln_split_adaln.py
"""
import glob

import pytest
import torch
import ttnn

from tt_bio import ln_split as LS
from tt_bio import tenstorrent as T
from tt_bio.addcmul_at import addcmul_at

EPS = 1e-5
pytestmark = pytest.mark.skipif(not glob.glob("/dev/tenstorrent/*"), reason="needs a Tenstorrent device")


def _dev(t, dev):
    return ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rel_rms(a, b):
    return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()


# (samples, tokens, channels, stacked steps or None, window offsets in tiles of the scale and shift tensors, rms)
# The rms cases are RFD3's DiT: 704 tokens, 768 channels, windows of a 27648-wide hoisted tensor (eps 1e-6).
@pytest.mark.parametrize("b,n,c,steps,offs,rms", [(1, 544, 768, None, (24, 0), False), (1, 544, 768, 3, (48, 24), False),
                                                  (2, 256, 768, 2, (0, 72), False), (1, 128, 384, None, (12, 0), False),
                                                  (1, 704, 768, None, (72, 72), True), (1, 128, 384, None, (12, 0), True)])
def test_adaln_matches_float64_and_the_two_op_route(b, n, c, steps, offs, rms):
    eps = 1e-6 if rms else EPS
    dev = T.get_device()
    g = torch.Generator().manual_seed(n + c + b)
    a = (torch.randn(b, n, c, generator=g) * 3 + torch.randn(b, 1, c, generator=g)).to(torch.bfloat16)
    lead = steps or 1
    row = None if steps is None else steps - 1
    width = 4 * c
    sc = torch.sigmoid(torch.randn(lead, n, width, generator=g)).to(torch.bfloat16)
    sh = (torch.randn(lead, n, width, generator=g) / 2).to(torch.bfloat16)
    o_sc, o_sh = offs[0] * 32, offs[1] * 32
    r = row or 0

    d = torch.float64
    ad = a.to(d)
    norm = (ad * torch.rsqrt(ad.pow(2).mean(-1, keepdim=True) + eps) if rms
            else torch.nn.functional.layer_norm(ad, (c,), eps=eps))
    ref = sh[r, :, o_sh:o_sh + c].to(d) + norm * sc[r, :, o_sc:o_sc + c].to(d)

    da, dsc, dsh = _dev(a, dev), _dev(sc, dev), _dev(sh, dev)
    out = LS.adaln(da, (dsc, o_sc, c, row), (dsh, o_sh, c, row), eps, rms=rms)
    assert out is not None
    got = ttnn.to_torch(out).to(d)

    ln = ttnn.rms_norm(da, epsilon=eps) if rms else ttnn.layer_norm(da, epsilon=eps)
    two = ttnn.to_torch(addcmul_at(dsh, ln, dsc, c, x_off=o_sh, z_off=o_sc, x_row=row, z_row=row)).to(d)

    e_got, e_two = _rel_rms(got, ref), _rel_rms(two, ref)
    assert e_got < 1e-2, (e_got, e_two)
    # The norm stays fp32 into the modulation, one rounding fewer than the route it replaces.
    assert e_got <= e_two * 1.05, (e_got, e_two)
