"""The column attention's softmax over R keys, on the two routes `AF2Attention._attend` can take.

q, k, v are [n, 8, R, 32] as the column attention hands them over. The bias is the all-ones
mask's `1e9 * (mask - 1)`, exactly zero, laid out [n, 1, 1, R] with its padding zeroed as
`_mask_biases` builds it. The reference is float64 torch on the same bf16 inputs.
"""
import json
import sys

import torch
import ttnn

from tt_bio import af2, tenstorrent as tn

dev = ttnn.open_device(device_id=0)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             math_approx_mode=False, fp32_dest_acc_en=True,
                                             packer_l1_acc=True)
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
out = {}
for R in [int(r) for r in sys.argv[1].split(",")]:
    torch.manual_seed(0)
    q, k, v = (torch.randn(256, 8, R, 32).bfloat16() for _ in range(3))
    s = (q.double() @ k.double().transpose(-1, -2)) * 32 ** -0.5
    want = torch.softmax(s, -1) @ v.double()
    flat = ttnn.multiply(ttnn.subtract(up(torch.ones(R, 256)), 1.0), af2.MASK_LOGIT_BIAS)
    raw = ttnn.reshape(ttnn.permute(flat, (1, 0)), (256, 1, 1, R))
    fp32 = tn._fp32_softmax_attention(up(q), up(k), up(v), ttnn.fill_implicit_tile_padding(raw, 0.0),
                                      scale_inv=32 ** -0.5,
                                      compute_kernel_config=ckc, out_dtype=ttnn.bfloat16,
                                      bias_scale_inv=1.0, l1_padded_plan=True)
    # `_mask_biases` before bcx-nan's e5cf74790: the reshape's padding left as the buffer held it.
    nofill = tn._fp32_softmax_attention(up(q), up(k), up(v), raw, scale_inv=32 ** -0.5,
                                        compute_kernel_config=ckc, out_dtype=ttnn.bfloat16,
                                        bias_scale_inv=1.0, l1_padded_plan=True)
    sc = ttnn.multiply_(tn.batched_matmul(up(q), ttnn.permute(up(k), (0, 1, 3, 2)),
                                          compute_kernel_config=ckc), 32 ** -0.5)
    col = tn.batched_matmul(ttnn.softmax(sc, dim=-1, compute_kernel_config=ckc), up(v),
                            compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    rec = {}
    for name, t in (("fp32_softmax_attention", fp32), ("fp32_softmax_attention@nofill", nofill),
                    ("column_ckc_softmax", col)):
        got = ttnn.to_torch(t).double()
        rec[name] = {"rel_l2": float((got - want).norm() / want.norm()),
                     "max_abs": float((got - want).abs().max())}
    out[R] = rec
    print(R, json.dumps(rec), flush=True)
ttnn.close_device(dev)
if len(sys.argv) > 2:
    open(sys.argv[2], "w").write(json.dumps(out, indent=1))
