#!/usr/bin/env python3
"""Does ttnn SDPA flatten the atom-block attention distribution?

`tenstorrent.py:1789` rejects SDPA for the protenix/opendde DiT because it "systematically
flattens near-degenerate attention distributions" -- 7XI5 repeat-protein logits lost ~16% of
their output std at PCC 0.98128. RFD3's atom block is a candidate for the same regime: the
-1e4 mask leaves 128 live keys out of 4032 and the bias spread across them is small, so the
softmax over the live keys is close to uniform.

This measures the thing that claim is about -- the std of the attention OUTPUT, per head,
against a torch fp32 reference -- for the shipped chain and for SDPA, at the production
shape and across bias spreads from near-degenerate (0.05) to peaked (2.0). A ratio near 1.0
means no flattening; 0.84 would reproduce the 7XI5 observation.
"""
import argparse, json

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=4032)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=32)
ap.add_argument("--n-keys", type=int, default=128)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = get_device()
dg = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False, packer_l1_acc=False)
H, D, M = a.heads, a.head_dim, a.m
N = M
SC = float(D) ** -0.5
sdpa = ttnn.transformer.scaled_dot_product_attention
pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dg, q_chunk_size=32,
                            k_chunk_size=256, exp_approx_mode=False)
torch.manual_seed(0)
tq = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
tk = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
tv = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
T = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                              device=dev, memory_config=DRAM)
q, k, v = T(tq), T(tk), T(tv)
idx = torch.stack([torch.randperm(N)[: a.n_keys].sort().values for _ in range(M)])
idx = idx.view(1, 1, M, a.n_keys).expand(1, H, M, a.n_keys)

res = {}
print(f"M={M} H={H} D={D} live keys/row={a.n_keys}", flush=True)
print(f"{'spread':>8} {'entropy':>9} | {'chain std/ref':>13} {'chain pcc':>11} | "
      f"{'sdpa std/ref':>13} {'sdpa pcc':>11}", flush=True)
for spread in (0.05, 0.25, 0.5, 1.0, 2.0):
    tb = torch.full((1, H, M, N), -1e4)
    tb.scatter_(3, idx, torch.randn(1, H, M, a.n_keys) * spread)
    tb = tb.bfloat16().float()
    logits = (tq @ tk.transpose(-1, -2)) * SC + tb
    w = torch.softmax(logits, dim=-1)
    ref = w @ tv
    # effective number of attended keys: exp(entropy) of the softmax row
    ent = float(torch.exp(-(w.clamp_min(1e-30) * w.clamp_min(1e-30).log()).sum(-1)).mean())

    b16 = T(tb)
    b32 = ttnn.typecast(b16, ttnn.float32, memory_config=DRAM)
    s = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=ckc)
    s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
    s = ttnn.multiply(s, SC)
    s = ttnn.add(s, b32)
    s = ttnn.softmax(s, dim=-1)
    s = ttnn.typecast(s, ttnn.bfloat16, memory_config=s.memory_config())
    chain = ttnn.to_torch(ttnn.matmul(s, v, compute_kernel_config=ckc, dtype=ttnn.bfloat16)).float()
    # SDPA sees the bias pre-scaled by sqrt(head_dim), exactly as the shipped model does it
    bs = T((tb / SC).bfloat16().float())
    out = ttnn.to_torch(sdpa(q, k, v, attn_mask=bs, is_causal=False, scale=SC,
                             compute_kernel_config=ckc, program_config=pc)).float()

    def sc_(x):
        pcc = torch.corrcoef(torch.stack([x.flatten().double(), ref.flatten().double()]))[0, 1]
        return (float(x.std() / ref.std()), float(pcc), float((x - ref).abs().max()))

    c, sd = sc_(chain), sc_(out)
    res[str(spread)] = {"entropy_eff_keys": round(ent, 1),
                        "chain": {"std_ratio": round(c[0], 5), "pcc": round(c[1], 7),
                                  "max_abs": round(c[2], 6)},
                        "sdpa": {"std_ratio": round(sd[0], 5), "pcc": round(sd[1], 7),
                                 "max_abs": round(sd[2], 6)}}
    print(f"{spread:8.2f} {ent:9.1f} | {c[0]:13.5f} {c[1]:11.7f} | {sd[0]:13.5f} {sd[1]:11.7f}",
          flush=True)
    for t in (b16, b32, bs):
        ttnn.deallocate(t)

if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote " + a.out, flush=True)
