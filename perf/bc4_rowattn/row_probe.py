"""Device grade of the MSA row attention's fused route against float64.

    TT_VISIBLE_DEVICES=<chip> python3 perf/bc4_rowattn/row_probe.py [--n 224]

`AF2Attention._row_sdpa` folds [rows, heads, n, d] into [1, rows * heads, n, d] and calls the fused
triangle-attention SDPA; under a tape `triatt_bw` is its backward. Both are graded here on the device's own
bf16 operands at that folded shape, with a per-row key mask in the bias, beside the composed fp32 chain the
route replaces. Reference: float64 softmax(q k^T * scale + bias) v and its autograd VJP.
"""
import argparse
import json

import torch
import ttnn

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import triatt_bw as BW  # noqa: E402

ROWS, H, D = 2, 8, 32
S = D ** -0.5


def ref(q, k, v, bias):
    return torch.softmax(q @ k.transpose(-1, -2) * S + bias, -1) @ v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[224, 256, 192])
    a = ap.parse_args()
    dev = T.get_device()
    BW.FUSED = BW.EXP_21F = True
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    dn = lambda t: ttnn.to_torch(t).double()
    cfg = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True)
    rel = lambda got, want: float((got - want).norm() / want.norm())
    rows = []
    for n in a.n:
        for masked in (False, True):
            torch.manual_seed(n)
            q, k, v, g = (up(torch.randn(ROWS, H, n, D)) for _ in range(4))
            pair = torch.randn(1, H, n, n)
            mask = torch.zeros(ROWS, 1, 1, n)
            if masked:   # padded keys, a different count per MSA row
                mask[0, ..., n - 19:] = -1e9
                mask[1, ..., n - 40:] = -1e9
            bias = ttnn.add(up(pair), up(mask))   # [rows, H, n, n], as AF2Attention sums them
            qd, kd, vd, bd = (dn(t).requires_grad_() for t in (q, k, v, bias))
            want = ref(qd, kd, vd, bd)
            grads = torch.autograd.grad(want, (qd, kd, vd, bd), dn(g))
            fold = lambda t: ttnn.reshape(t, (1, ROWS * H, n, int(t.shape[-1])))
            b = ttnn.multiply(fold(bias), 1.0 / S)
            out = T._tri_att_sdpa_hifi(fold(q), fold(k), fold(v), b, S)
            row = {"n": n, "masked": masked, "served": out is not None}
            if out is not None:
                row["kernel"] = rel(dn(out).reshape(ROWS, H, n, D), want.detach())
            row["chain"] = rel(dn(T._fp32_softmax_attention(
                q, k, v, bias, scale_inv=S, compute_kernel_config=cfg, out_dtype=ttnn.bfloat16,
                bias_scale_inv=1.0)), want.detach())
            ok, why = BW.eligible(fold(q), fold(k), fold(v), fold(bias))
            row["bw_eligible"] = why
            if ok:
                got = BW.run(dev, fold(q), fold(k), fold(v), fold(bias), fold(g), S,
                             (ttnn.MathFidelity.HiFi4,))
                keep = (mask.expand(ROWS, H, n, n) == 0) if masked else None
                for name, t, r in zip(("dq", "dk", "dv", "dbias"), got, grads):
                    t = dn(t).reshape(r.shape)
                    if name == "dbias" and masked:   # a masked key's bias gradient is 0 in both
                        t, r = t[keep], r[keep]
                    row[name] = rel(t, r)
            row["ok"] = (row["served"] and ok and row["kernel"] <= max(row["chain"] * 1.5, 6e-3)
                         and all(row[x] < 1.5e-2 for x in ("dq", "dk", "dv", "dbias")))
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("PASS" if all(r["ok"] for r in rows) else "FAIL")


if __name__ == "__main__":
    main()
