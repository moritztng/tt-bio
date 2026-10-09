"""PairWeightedAveraging's unpadded head path moves its output back to [rows, T, H*hd] with two tile transposes
when rows is a multiple of 32, and with permute (1, 2, 0) otherwise. Both are pure data movement, so the transposed
path must equal the permute bit for bit, and both must match float64.
"""
from types import SimpleNamespace

import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

H, HD, C_M, TOK = 8, 8, 128, 64


def _setup(rows):
    dev = T.get_device()
    torch.manual_seed(rows)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    bf = lambda t: t.to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(bf(t), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mn = bf(torch.randn(rows, TOK, C_M))
    wv, wg = bf(torch.randn(C_M, H * HD) / C_M ** 0.5), bf(torch.randn(C_M, H * HD) / C_M ** 0.5)
    wo = bf(torch.randn(H * HD, C_M) / (H * HD) ** 0.5)
    w = bf(torch.softmax(torch.randn(H, TOK, TOK), -1))
    mod = SimpleNamespace(n_heads=H, head_dim=HD, compute_kernel_config=ckc,
                          m_weight=up(wv), g_weight=up(wg), o_weight=up(wo))
    ws = [up(w[h:h + 1]) for h in range(H)]
    v = mn.double() @ wv.double()
    g = torch.sigmoid(mn.double() @ wg.double())
    o = torch.einsum("hij,rjhc->rihc", w.double(), v.reshape(rows, TOK, H, HD)) * g.reshape(rows, TOK, H, HD)
    ref = o.reshape(rows, TOK, H * HD) @ wo.double()
    return mod, up(mn), ws, ref


def _permute_path(mod, mc, ws):
    """The pre-transpose code of `_heads_unpadded`, kept here as the bit-exact reference."""
    rows, T_ = int(mc.shape[0]), int(mc.shape[1])
    lin = dict(compute_kernel_config=mod.compute_kernel_config, core_grid=T.CORE_GRID_MAIN)
    vh = ttnn.reshape(ttnn.permute(ttnn.linear(mc, mod.m_weight, **lin), (2, 0, 1)), (H, HD * rows, T_))
    o = ttnn.matmul(vh, ttnn.concat(list(ws), dim=0), transpose_b=True, **lin)
    ot = ttnn.permute(ttnn.reshape(o, (H * HD, rows, T_)), (1, 2, 0))
    ot = ttnn.multiply_(ot, ttnn.linear(mc, mod.g_weight, **lin),
                        input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    return ttnn.linear(ot, mod.o_weight, **lin)


@pytest.mark.device
@pytest.mark.parametrize("rows", [64, 48])
def test_head_transpose_matches_permute_and_float64(rows):
    mod, mc, ws, ref = _setup(rows)
    got = ttnn.to_torch(T.PairWeightedAveraging._heads_unpadded(mod, mc, ws)).reshape(rows, TOK, C_M)
    base = ttnn.to_torch(_permute_path(mod, mc, ws)).reshape(rows, TOK, C_M)
    assert torch.equal(got, base)
    rel = (got.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert torch.isfinite(got).all() and rel < 1e-2, float(rel)
