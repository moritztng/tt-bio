"""PairWeightedAveraging's fused head path (0.14.0) sums the heads inside one output projection's fp32
accumulator; the per-head loop it replaced summed one bf16 projection per head with bf16 adds. The fused
form is a different rounding of the same algebra, so it is not bit-exact with the loop, and it must be at
least as close to float64 as the loop was. Heads of width 8 are Protenix-v2's (the unpadded path), width
32 Boltz-2's (the fused path with no padding); rows=1 is a single-sequence fold.
"""
from types import SimpleNamespace

import pytest
import torch
import ttnn

from tt_bio import tenstorrent as T

H, TOK = 8, 64


def _setup(rows, hd, c_m):
    dev = T.get_device()
    torch.manual_seed(rows * hd)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    bf = lambda t: t.to(torch.bfloat16)
    up = lambda t: ttnn.from_torch(bf(t), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mn = bf(torch.randn(rows, TOK, c_m))
    wv, wg = bf(torch.randn(c_m, H * hd) / c_m ** 0.5), bf(torch.randn(c_m, H * hd) / c_m ** 0.5)
    wo = bf(torch.randn(H * hd, c_m) / (H * hd) ** 0.5)
    w = bf(torch.softmax(torch.randn(H, TOK, TOK), -1))
    mod = SimpleNamespace(n_heads=H, head_dim=hd, compute_kernel_config=ckc,
                          m_weight=up(wv), g_weight=up(wg), o_weight=up(wo))
    ws = [up(w[h:h + 1]) for h in range(H)]
    v = mn.double() @ wv.double()
    g = torch.sigmoid(mn.double() @ wg.double())
    o = torch.einsum("hij,rjhc->rihc", w.double(), v.reshape(rows, TOK, H, hd)) * g.reshape(rows, TOK, H, hd)
    ref = o.reshape(rows, TOK, H * hd) @ wo.double()
    return mod, up(mn), ws, ref, (wv, wg, wo), up


def _loop_path(mod, mc, ws, wv, wg, wo, up):
    """The pre-0.14.0 per-head loop: projection, average, gate and output projection per head, the head
    outputs summed in bf16."""
    rows, T_, hd = int(mc.shape[0]), int(mc.shape[1]), mod.head_dim
    lin = dict(compute_kernel_config=mod.compute_kernel_config, core_grid=T.CORE_GRID_MAIN)
    acc = None
    for h in range(H):
        s = slice(h * hd, (h + 1) * hd)
        v = ttnn.linear(mc, up(wv[:, s].contiguous()), **lin)                  # [rows, T, hd]
        v = ttnn.reshape(ttnn.permute(v, (1, 2, 0)), (T_, hd * rows))
        o = ttnn.matmul(ttnn.reshape(ws[h], (T_, T_)), v, **lin)               # [T, hd*rows]
        o = ttnn.permute(ttnn.reshape(o, (T_, hd, rows)), (2, 0, 1))           # [rows, T, hd]
        o = ttnn.multiply_(o, ttnn.linear(mc, up(wg[:, s].contiguous()), **lin),
                           input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        o = ttnn.linear(o, up(wo[s].contiguous()), **lin)
        acc = o if acc is None else ttnn.add(acc, o)
    return acc


@pytest.mark.device
@pytest.mark.parametrize("hd,c_m", [(8, 128), (32, 64)])
@pytest.mark.parametrize("rows", [1, 64])
def test_fused_heads_at_least_as_close_to_float64_as_the_loop(rows, hd, c_m):
    mod, mc, ws, ref, (wv, wg, wo), up = _setup(rows, hd, c_m)
    if hd == 32:    # head width is a whole tile: the packed weights are the checkpoint's own
        got = T.PairWeightedAveraging._heads_fused(mod, mc, ws, (up(wv), up(wg), up(wo)))
    else:
        got = T.PairWeightedAveraging._heads_unpadded(mod, mc, ws)
    fused = ttnn.to_torch(got).reshape(rows, TOK, c_m)
    loop = ttnn.to_torch(_loop_path(mod, mc, ws, wv, wg, wo, up)).reshape(rows, TOK, c_m)
    assert torch.isfinite(fused).all() and torch.isfinite(loop).all()
    err = lambda x: float((x.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
    print(f"rows={rows} hd={hd} rel-rms vs float64: fused {err(fused):.3e} loop {err(loop):.3e}")
    assert err(fused) <= err(loop) * 1.05, (err(fused), err(loop))
