#!/usr/bin/env python3
"""Which part of SDPA's contract differs from the shipped chain?

Both bias kinds gave the same PCC 0.925 against torch, so the defect is not the mask's
sparsity and it is not the flash softmax (that lands at 1e-3, as the shipped chain does).
The remaining candidates are what `scale` multiplies and whether the mask is inside it.
Each variant below is the same arithmetic under a different reading of the contract; the
one that reproduces torch names the contract.
"""
import json

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False, packer_l1_acc=False)
H, D, M = 4, 32, 1024
N = M
SC = float(D) ** -0.5
torch.manual_seed(0)
tq = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
tk = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
tv = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
tb = (torch.randn(1, H, M, N) * 0.5).bfloat16().float()

T = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(  # noqa: E731
    x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
q, k, v, b = T(tq), T(tk), T(tv), T(tb)
qs = T((tq * SC).bfloat16().float())
binv = T((tb / SC).bfloat16().float())
sdpa = ttnn.transformer.scaled_dot_product_attention


def ref(scale, bias, qq=None):
    s = ((qq if qq is not None else tq) @ tk.transpose(-1, -2)) * scale + bias
    return torch.softmax(s, dim=-1) @ tv


def cmp(label, out, r):
    x = ttnn.to_torch(out).float()
    d = (x - r).abs()
    pcc = torch.corrcoef(torch.stack([x.flatten().double(), r.flatten().double()]))[0, 1].item()
    print(f"  {label:38s} maxabs {d.max().item():.6f}  pcc {pcc:.9f}", flush=True)
    return {"max_abs": round(d.max().item(), 6), "pcc": round(pcc, 9)}


res = {}
r_true = ref(SC, tb)
r_noscale_bias = ref(SC, tb * SC)          # if `scale` also multiplies the mask
r_nomask = ref(SC, torch.zeros(1))

print("=== target: softmax(SC*qk + bias) @ v ===", flush=True)
res["A_scale_SC"] = cmp("sdpa(q,k,v,mask,scale=SC)",
                        sdpa(q, k, v, attn_mask=b, is_causal=False, scale=SC,
                             compute_kernel_config=ckc), r_true)
res["B_mask_predivided"] = cmp("sdpa(q,k,v,mask/SC,scale=SC)",
                               sdpa(q, k, v, attn_mask=binv, is_causal=False, scale=SC,
                                    compute_kernel_config=ckc), r_true)
res["C_prescaled_q"] = cmp("sdpa(q*SC,k,v,mask,scale=1.0)",
                           sdpa(qs, k, v, attn_mask=b, is_causal=False, scale=1.0,
                                compute_kernel_config=ckc), r_true)
res["D_prescaled_q_noscale"] = cmp("sdpa(q*SC,k,v,mask,scale=None)",
                                   sdpa(qs, k, v, attn_mask=b, is_causal=False,
                                        compute_kernel_config=ckc), r_true)
print("=== control: is `scale` reaching the mask? (compare A against softmax(SC*(qk+bias))) ===",
      flush=True)
res["A_vs_scaled_bias"] = cmp("sdpa(q,k,v,mask,scale=SC) vs SC*(qk+b)",
                              sdpa(q, k, v, attn_mask=b, is_causal=False, scale=SC,
                                   compute_kernel_config=ckc), r_noscale_bias)
print("=== control: no mask at all ===", flush=True)
res["E_nomask"] = cmp("sdpa(q,k,v,no mask,scale=SC)",
                      sdpa(q, k, v, is_causal=False, scale=SC,
                           compute_kernel_config=ckc), r_nomask)
json.dump(res, open("perf/atomblock_qkt/scale_qb1c2.json", "w"), indent=2)
