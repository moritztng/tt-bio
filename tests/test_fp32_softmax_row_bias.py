"""A blocked fp32-softmax attention hands each row block its own rows of a per-row bias.

AF2's MSA row attention adds the pair bias ``[1, heads, n, n]`` and the per-row MSA mask bias
``[rows, 1, 1, n]`` into one ``[rows, heads, n, n]`` bias. Once the token axis is long enough that
the call blocks its leading dim, each block of q rows has to see the same rows of the bias. The
loop used to pass the whole bias to every block, so a 1-row block met a 2-row bias, the in-place
add broadcast to 2 rows and refused its 1-row output: BindCraft 2 crashed at 864 tokens.

Host-only. The block runner, the bias cast and the two ttnn calls the loop makes are stubbed
with float64 torch, so this checks the partition against the unblocked maths, not the kernels.
"""
from __future__ import annotations

import pytest
import torch

from tt_bio import tenstorrent as tt


class Fake:
    """Just enough of a ttnn tensor for the blocking loop: shape, padded_shape, row slicing."""

    def __init__(self, t):
        self.t = t
        self.shape = self.padded_shape = tuple(t.shape)

    def __getitem__(self, idx):
        return Fake(self.t[idx])


def reference(q, k, v, bias, scale):
    return torch.softmax(q @ k.transpose(-1, -2) * scale + bias, dim=-1) @ v


@pytest.fixture
def stubbed(monkeypatch):
    def block(q, k, v, bias, scale_inv, ckc, out_dtype, bias_scale_inv, shard=None, l1_key=None,
              bias_f=None, **_):
        b = bias_f.t if bias_f is not None else bias.t * bias_scale_inv
        # ttnn.add_ writes into the scores, so the bias may broadcast up to them and never past.
        assert b.shape[0] in (1, q.t.shape[0]), (tuple(b.shape), tuple(q.t.shape))
        return Fake(reference(q.t, k.t, v.t, b, scale_inv))

    monkeypatch.setattr(tt, "_fp32_softmax_attention_block", block)
    monkeypatch.setattr(tt, "_fp32_softmax_bias",
                        lambda bias, s, bs: Fake(bias.t * (s if bs is None else bs)))
    monkeypatch.setattr(tt, "_fp32_softmax_l1_plan", lambda *a, **kw: (0, 0))
    monkeypatch.setattr(tt.ttnn, "deallocate", lambda t: None)
    monkeypatch.setattr(tt.ttnn, "concat", lambda parts, dim: Fake(torch.cat([p.t for p in parts], dim)))
    tt._FP32_SOFTMAX_DRAM_ROW_CAP.clear()
    before = dict(tt.FP32_SOFTMAX_STATS)
    yield
    tt._FP32_SOFTMAX_DRAM_ROW_CAP.clear()
    tt.FP32_SOFTMAX_STATS.update(before)


@pytest.mark.parametrize("hoist", [True, False])
@pytest.mark.parametrize("bias_rows", [1, 40])
def test_blocked_call_matches_the_unblocked_maths(stubbed, monkeypatch, hoist, bias_rows):
    rows, heads, n, d = 40, 2, 32, 8
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(rows, heads, n, d, generator=g, dtype=torch.float64) for _ in range(3))
    bias = torch.randn(bias_rows, heads, n, n, generator=g, dtype=torch.float64)
    monkeypatch.setattr(tt, "FP32_SOFTMAX_BIAS_HOIST", hoist)
    # One row's fp32 scores is the whole budget, so the call blocks at the 32-row floor: a full
    # block and a ragged 8-row tail, the shape of the crash with a tail to get wrong as well.
    monkeypatch.setattr(tt, "_FP32_SOFTMAX_BLOCK_BYTES", heads * n * n * 4)
    blocked = tt.FP32_SOFTMAX_STATS["blocks"]
    out = tt._fp32_softmax_attention(Fake(q), Fake(k), Fake(v), Fake(bias), scale_inv=d ** -0.5,
                                     compute_kernel_config=None, bias_scale_inv=1.0,
                                     l1_padded_plan=False)
    assert tt.FP32_SOFTMAX_STATS["blocks"] - blocked == 2
    torch.testing.assert_close(out.t, reference(q, k, v, bias, d ** -0.5), rtol=0, atol=1e-12)
