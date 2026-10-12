"""Fused HiFi4 triangle attention against float64, at OpenFold3's two triangle head dims.

The fused one-k-chunk route (`_tri_att_sdpa_hifi`) was measured at head_dim 32 only. OpenFold3's
template stack uses head_dim 16, padded with zeros to a 32-wide tile, and is the one site where the
route moves ipTM far outside seed variation (spd-of3 ab11t). This asks whether that is imprecision
or a wrong result: both routes against a float64 reference on the same bf16 operands.

usage (one chip, tt-bio's device path): TT_VISIBLE_DEVICES=N python perf/spd/triatt_headdim.py [S] [ROWS]
"""
import math
import sys

import torch
import ttnn

import tt_bio.tenstorrent as T

S = int(sys.argv[1]) if len(sys.argv) > 1 else 736
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
N_REAL = S - 6          # c730 buckets 730 tokens to 736
H = 4
dev = T.get_device()
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def err(o, ref, d):
    o = o[..., :d].double()
    rel = ((o - ref).pow(2).mean() / ref.pow(2).mean()).sqrt().item()
    return rel, (o - ref).abs().max().item()


torch.manual_seed(0)
for d in (16, 32):
    for bias_sd in (1.0, 4.0):
        q = torch.zeros(ROWS, H, S, 32)
        k, v = torch.zeros_like(q), torch.zeros_like(q)
        q[..., :d], k[..., :d], v[..., :d] = (torch.randn(ROWS, H, S, d) for _ in range(3))
        bias = torch.randn(1, H, S, S) * bias_sd
        bias[..., N_REAL:] = -1e9                       # padded keys, as the token pad mask does
        q, k, v, bias = (x.bfloat16().float() for x in (q, k, v, bias))
        scale = math.sqrt(d)
        s64 = q[..., :d].double() @ k[..., :d].double().transpose(-1, -2) / scale + bias.double()
        ref = torch.softmax(s64, -1) @ v[..., :d].double()
        # Unscaled pair bias (OF3's scale_pair_bias=False): the fused kernel adds the mask before
        # the scale, so it takes the bias times sqrt(d), exactly as `_attend_heads` passes it.
        fused = T._tri_att_sdpa_hifi(tt(q), tt(k), tt(v), tt(bias * scale), 1.0 / scale, one_k_chunk=True)
        mat = T._fp32_softmax_attention(tt(q), tt(k), tt(v), tt(bias), scale_inv=1.0 / scale,
                                        compute_kernel_config=ckc, bias_scale_inv=1.0)
        row = [f"d={d} bias_sd={bias_sd} S={S} rows={ROWS}"]
        for name, o in (("fused", fused), ("materialised", mat)):
            if o is None:
                row.append(f"{name}: declined")
                continue
            rel, mx = err(ttnn.to_torch(o).float(), ref, d)
            row.append(f"{name}: rel_rms {rel:.3e} max {mx:.3e}")
        print(" | ".join(row), flush=True)
