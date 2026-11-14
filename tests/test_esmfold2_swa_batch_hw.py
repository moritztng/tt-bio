"""ESMFold2's windowed atom attention with diffusion samples batched (`draw(multiplicity > 1)`).

The batch is folded into the heads, so every sample must come out bit-identical to the same
sample run alone, and both must match the dense band in float64. Before the fold, a batched
draw died in `ttnn.concat` (the zero frame had batch 1), which is what a small multi-chain
complex hit on Wormhole once the windowed path became the default there.

Run: TT_VISIBLE_DEVICES=<card> python3 -m pytest tests/test_esmfold2_swa_batch_hw.py
"""
import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T
from tt_bio.esmfold2 import SWAWindow, _window_bias, band_mask
from tt_bio.main import ensure_p300_mesh_descriptor

pytestmark = pytest.mark.device


@pytest.mark.parametrize("N,tail", [(1000, 5), (2048, 0)])
def test_batched_window_is_per_sample_and_the_dense_band(N, tail):
    ensure_p300_mesh_descriptor()
    dev = T.get_device()
    B, H, dh = 3, 4, 32
    mask = torch.ones(N)
    if tail:
        mask[-tail:] = 0
    dense = band_mask(mask, 64)[0, 0]
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = SWAWindow(_window_bias(mask, 64), N, H, dh, up, dev)
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(B, H, N, dh, generator=g) for _ in range(3))
    try:
        got = ttnn.to_torch(w.attend(up(q), up(k), up(v), dh ** -0.5)).float()
        alone = torch.cat([ttnn.to_torch(w.attend(up(q[b:b + 1]), up(k[b:b + 1]), up(v[b:b + 1]),
                                                  dh ** -0.5)).float() for b in range(B)])
    finally:
        w.deallocate()
    assert got.shape == (B, H, N, dh) and torch.isfinite(got).all()
    assert torch.equal(got, alone)
    qd, kd, vd = (ttnn.to_torch(up(t)).double() for t in (q, k, v))  # the bf16 inputs the device saw
    ref = torch.softmax(qd @ kd.transpose(-1, -2) * dh ** -0.5 + dense.double(), -1) @ vd
    valid = mask > 0.5
    err = (got[:, :, valid] - ref[:, :, valid]).abs().max().item()
    assert err < 0.08, err  # bf16 output; the B = 1 kernel alone reads 0.0507 at N = 1000 on Wormhole
