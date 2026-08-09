#!/usr/bin/env python3
"""Three-way: torch reference vs the shipped device chain vs SDPA.

The sweep measured SDPA at PCC 0.917 against the shipped chain, which is far too large to be
rounding. This isolates the cause: torch fp32 is the ground truth, and each device path is
scored against it, under a dense bias and under the real -1e4 sparse bias separately. If SDPA
is off only under the sparse bias, the cause is mask handling; if it is off under both, it is
the flash softmax; if the SHIPPED chain is the one that moves, the reference was wrong.
"""
import argparse, json

import torch
import ttnn

from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
ap = argparse.ArgumentParser()
ap.add_argument("--m", type=int, default=1024)
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
torch.manual_seed(0)

tq = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
tk = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()
tv = (torch.randn(1, H, M, D) * 0.3).bfloat16().float()


def make_bias(kind):
    if kind == "dense":
        b = torch.randn(1, H, M, N) * 0.5
    else:
        b = torch.full((1, H, M, N), -1e4)
        idx = torch.stack([torch.randperm(N)[: a.n_keys].sort().values for _ in range(M)])
        loc = torch.randn(1, H, M, a.n_keys) * 0.5
        b.scatter_(3, idx.view(1, 1, M, a.n_keys).expand(1, H, M, a.n_keys), loc)
    return b.bfloat16().float()   # the shipped bias is bf16 before its fp32 upcast


def torch_ref(tb):
    s = (tq @ tk.transpose(-1, -2)) * SC + tb
    return (torch.softmax(s, dim=-1) @ tv)


def score(x, ref, label):
    d = (x - ref).abs()
    xf, rf = x.flatten().double(), ref.flatten().double()
    pcc = torch.corrcoef(torch.stack([xf, rf]))[0, 1].item()
    r = {"max_abs": round(d.max().item(), 6), "mean_abs": round(d.mean().item(), 8),
         "pcc": round(pcc, 9), "ref_absmax": round(ref.abs().max().item(), 5)}
    print(f"    {label:26s} maxabs {r['max_abs']:.6f}  meanabs {r['mean_abs']:.8f}  "
          f"pcc {r['pcc']:.9f}", flush=True)
    return r


res = {}
sdpa = ttnn.transformer.scaled_dot_product_attention
q = ttnn.from_torch(tq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
k = ttnn.from_torch(tk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
v = ttnn.from_torch(tv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)

for kind in ("dense", "sparse"):
    print(f"=== bias: {kind} ===", flush=True)
    tb = make_bias(kind)
    ref = torch_ref(tb)
    b16 = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                          memory_config=DRAM)
    b32 = ttnn.typecast(b16, ttnn.float32, memory_config=DRAM)
    kT = ttnn.permute(k, (0, 1, 3, 2))
    s = ttnn.matmul(q, kT, compute_kernel_config=ckc)
    s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
    s = ttnn.multiply(s, SC)
    s = ttnn.add(s, b32)
    sm = ttnn.softmax(s, dim=-1)
    smb = ttnn.typecast(sm, ttnn.bfloat16, memory_config=sm.memory_config())
    shipped = ttnn.to_torch(ttnn.matmul(smb, v, compute_kernel_config=ckc, dtype=ttnn.bfloat16)).float()
    row = {"shipped_vs_torch": score(shipped, ref, "shipped vs torch")}

    # the softmax itself, before attn@v, to localise the error
    row["softmax_vs_torch"] = score(
        ttnn.to_torch(sm).float(),
        torch.softmax((tq @ tk.transpose(-1, -2)) * SC + tb, dim=-1),
        "  shipped softmax vs torch")

    try:
        sm_stable = ttnn.softmax(s, dim=-1, numeric_stable=True)
        row["softmax_stable_vs_torch"] = score(
            ttnn.to_torch(sm_stable).float(),
            torch.softmax((tq @ tk.transpose(-1, -2)) * SC + tb, dim=-1),
            "  stable softmax vs torch")
        ttnn.deallocate(sm_stable)
    except Exception as e:
        print(f"    numeric_stable ERR {str(e)[:80]}", flush=True)

    for lbl, kw in (("sdpa default", {}),
                    ("sdpa exp_approx=False", {"program_config": ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=dg, q_chunk_size=32, k_chunk_size=256,
                        exp_approx_mode=False)}),
                    ("sdpa exp_approx=True", {"program_config": ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=dg, q_chunk_size=32, k_chunk_size=256,
                        exp_approx_mode=True)})):
        try:
            o = sdpa(q, k, v, attn_mask=b16, is_causal=False, scale=SC,
                     compute_kernel_config=ckc, **kw)
            row[lbl] = score(ttnn.to_torch(o).float(), ref, lbl)
            ttnn.deallocate(o)
        except Exception as e:
            print(f"    {lbl} ERR {str(e)[:90]}", flush=True)
    # fp32 dest accumulate inside SDPA
    try:
        ckc32 = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=False)
        o = sdpa(q, k, v, attn_mask=b16, is_causal=False, scale=SC, compute_kernel_config=ckc32)
        row["sdpa fp32_dest_acc"] = score(ttnn.to_torch(o).float(), ref, "sdpa fp32_dest_acc")
        ttnn.deallocate(o)
    except Exception as e:
        print(f"    sdpa fp32_dest_acc ERR {str(e)[:90]}", flush=True)
    res[kind] = row

if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote " + a.out, flush=True)
