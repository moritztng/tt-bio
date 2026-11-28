"""Device grade of `col_mix` against float64, beside the composed chain it replaces.

    TT_VISIBLE_DEVICES=<chip> python3 perf/bc4_msafuse/col_probe.py [--n 224]

Prints rel L2 against the float64 function of the device's own bf16 inputs for both paths. The kernel has to be
at least as close as the chain. `vjp` is `col_mix_bw` against float64 autograd of the same function, which a
bfloat16 output can meet to about 2e-3.
"""
import argparse
import json

import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
from tt_bio import col_mix as C  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402

H, D = 8, 32


def ref(qkv, bias):
    c = H * D
    q, k, v = qkv[..., :c], qkv[..., c:2 * c], qkv[..., 2 * c:]
    n = qkv.shape[1]
    kd, vd = k[0] - k[1], v[0] - v[1]
    gap = D ** -0.5 * (q * kd).reshape(2, n, H, D).sum(-1, keepdim=True)
    if bias is not None:
        gap = gap + bias.reshape(1, n, 1, 1)
    return v[1] + torch.sigmoid(gap).expand(2, n, H, D).reshape(2, n, c) * vd


def chain(qkv, bias, head_sum, cfg):
    n, w = int(qkv.shape[1]), H * D
    part = lambda r0, r1, j: ttnn.slice(qkv, (r0, 0, j * w), (r1, n, (j + 1) * w))
    q, k0, k1, v0, v1 = part(0, 2, 0), part(0, 1, 1), part(1, 2, 1), part(0, 1, 2), part(1, 2, 2)
    kd, vd = ttnn.subtract(k0, k1), ttnn.subtract(v0, v1)
    gap = ttnn.linear(ttnn.multiply(q, kd), head_sum, compute_kernel_config=cfg)
    if bias is not None:
        gap = ttnn.add(gap, bias)
    return ttnn.add(ttnn.multiply(gap, vd, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]), v1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[224, 200, 32])
    a = ap.parse_args()
    dev = T.get_device()
    C.COL_MIX = True
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    head_sum = up(torch.block_diag(*[torch.full((D, D), D ** -0.5)] * H))
    cfg = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True)
    rows = []
    for n in a.n:
        for with_bias in (False, True):
            torch.manual_seed(n)
            qkv = up(torch.randn(2, n, 3 * H * D))
            b = torch.zeros(1, n, 1)
            b[0, ::7] = 1e9
            b[0, 3::11] = -1e9
            b[0, 1::5] = torch.randn(len(b[0, 1::5]), 1)
            bias = ttnn.fill_implicit_tile_padding(up(b), 0.0) if with_bias else None
            want = ref(ttnn.to_torch(qkv).double(), ttnn.to_torch(bias).double() if with_bias else None)
            assert C.eligible(qkv, D, bias)
            rel = lambda y: float((ttnn.to_torch(y).double() - want).norm() / want.norm())
            row = {"n": n, "bias": with_bias, "kernel": rel(C.col_mix(qkv, D ** -0.5, bias)),
                   "chain": rel(chain(qkv, bias, head_sum, cfg))}
            x = ttnn.to_torch(qkv).double().requires_grad_()
            d = up(torch.randn(2, n, H * D))
            (gref,) = torch.autograd.grad(ref(x, ttnn.to_torch(bias).double() if with_bias else None), x,
                                          ttnn.to_torch(d).double())
            got = ttnn.to_torch(C.col_mix_bw(qkv, d, D ** -0.5, bias)).double()
            row["vjp"] = float((got - gref).norm() / gref.norm())
            row["vjp_parts"] = [float((got[..., j * 256:(j + 1) * 256] - gref[..., j * 256:(j + 1) * 256]).norm()
                                      / gref[..., j * 256:(j + 1) * 256].norm()) for j in range(3)]
            row["ok"] = row["kernel"] <= row["chain"] * 1.05 + 1e-6 and row["vjp"] < 4e-3
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("PASS" if all(r["ok"] for r in rows) else "FAIL")


if __name__ == "__main__":
    main()
