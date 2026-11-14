"""A wide `_lin` pair projection (`_WIDE_PROJ`) is `torch.equal` to the `core_grid=` call it replaces.

The plan keeps in0_block_w 1, so the contraction folds in the shipped order and only the drain schedule moves.
Shapes are the three OpenDDE sites at a 640-token fold, plus the fp32-dest-off config of the fast sets.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_wide_proj_hw.py
"""
import pytest
import torch
import ttnn

from tt_bio import ops
from tt_bio import tenstorrent as T


@pytest.mark.parametrize("fp32_dest", [True, False])
@pytest.mark.parametrize("shape,n", [((1, 640, 640, 384), 384), ((1, 640, 640, 64), 384), ((1, 1280, 640, 34), 128)])
def test_wide_projection_is_bit_exact(shape, n, fp32_dest, monkeypatch):
    dev = T.get_device()
    g = torch.Generator().manual_seed(n + shape[-1])
    x = torch.randn(*shape, generator=g).to(torch.bfloat16)
    w = (torch.randn(shape[-1], n, generator=g) * shape[-1] ** -0.5).to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    X, W = up(x), up(w)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                           fp32_dest_acc_en=fp32_dest, packer_l1_acc=True)
    assert T._pair_proj_config(X, W, bw_cap=1) is not None, "the shape must reach the new plan"

    def run(wide):
        monkeypatch.setattr(T, "_WIDE_PROJ", wide)
        return ttnn.to_torch(ops.linear(X, W, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                        core_grid=T.CORE_GRID_MAIN, narrow_proj=True))

    assert torch.equal(run(True), run(False))
