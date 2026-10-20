"""A split `bmm_res` call (BRES_SPLIT, b[c] resident one column block at a time) finishes and is as close to float64
as ttnn.matmul, with transpose_a at K tile counts that are not a multiple of the 4-tile transpose group.

At 736 tokens (KT 23) a row's last raw-a group was 3 tiles, so the next row's 4-tile reserve straddled the end of the
8-tile ring and the call hung the device (OpenDDE's refiner einsum, every fold whose structural tokens bucket past 704,
.107 10-19). 768 (KT 24) never hit it.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_bmm_res_split_hw.py
"""
import pytest
import torch
import ttnn

from tt_bio import bmm_res
from tt_bio import tenstorrent as T


def _rel64(x, ref):
    return ((x.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


@pytest.mark.parametrize("seq,ta,tb", [(736, True, False), (800, True, False), (736, False, True), (768, True, False)])
def test_split_call_finishes_and_matches_float64(seq, ta, tb, monkeypatch):
    monkeypatch.setattr(bmm_res, "BRES_SPLIT", True)
    dev = T.get_device()
    g = torch.Generator().manual_seed(seq)
    a = torch.randn(1, 8, seq, seq, generator=g).to(torch.bfloat16)
    b = (torch.randn(1, 8, seq, seq, generator=g) * seq ** -0.5).to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    A, B = up(a), up(b)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=False)
    kt = seq // 32
    t = bmm_res.MG.tile_bytes(ttnn.bfloat16)
    assert bmm_res._plan(kt, kt, t, t, transpose_a=ta)[0] > 1   # the split path, not the resident-whole-b one
    out = bmm_res.matmul(A, B, compute_kernel_config=ckc, transpose_a=ta, transpose_b=tb)
    assert out is not None
    got = ttnn.to_torch(out)
    ref = ttnn.to_torch(ttnn.matmul(A, B, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    dtype=ttnn.bfloat16, transpose_a=ta, transpose_b=tb))
    a64, b64 = a.double(), b.double()
    r64 = (a64.transpose(-1, -2) if ta else a64) @ (b64.transpose(-1, -2) if tb else b64)
    assert _rel64(got, r64) <= 1.05 * _rel64(ref, r64) + 1e-4
