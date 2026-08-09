#!/usr/bin/env python3
"""Isolate row-blocked L1 residency from the chunked path's extra copies.

The in-model A/B lost at N=320. The chunked path it reuses was built for memory safety at
S > 1536 and pays for it: a second full layer_norm pass, a row slice per block, and a final
concat. This strips all of that and times only the projection -> head-split -> SDPA chain,
whole-tensor vs row-blocked, on the same input, so the mechanism can be judged on its own.

Stages timed separately so the win (or loss) can be attributed:
  proj      qkv projection
  heads     nlp_create_qkv_heads
  sdpa      scaled_dot_product_attention + nlp_concat_heads
  slice     the row slice the blocked arm needs (charged to it)
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=320)
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--blocks", type=int, nargs="*", default=[0, 2, 4, 5, 8, 10, 16, 20])
ap.add_argument("--warm", type=int, default=3)
ap.add_argument("--iters", type=int, default=7)
ap.add_argument("--out", default=None)
args = ap.parse_args()

med = lambda x: sorted(x)[len(x) // 2]
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

N, C = args.n, args.c
HEADS, HD = C // 32, 32
torch.manual_seed(0)
xn = ttnn.from_torch(torch.randn(N, N, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
wq = ttnn.from_torch(torch.randn(C, 3 * C) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
bias = ttnn.from_torch(torch.randn(1, HEADS, N, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
sdpa_pc = T._tri_att_sdpa_program_config(N, N)

print(f"N={N} c={C} heads={HEADS} grid={T.COMPUTE_GRID_MAIN} l1={ttnn.get_max_worker_l1_unreserved_size()}",
      flush=True)


def chain(x, rows):
    """rows=0 -> whole tensor; else one row block at a time, L1-resident where admitted."""
    outs = []
    for s in range(0, N, rows or N):
        xb = xn if not rows else ttnn.slice(xn, [s, 0, 0], [s + rows, N, C])
        cfg = T._l1_resident_linear_config(xb, wq, ttnn.bfloat16)
        if cfg is not None:
            qkv = ttnn.linear(xb, wq, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                              memory_config=ttnn.L1_MEMORY_CONFIG, program_config=cfg)
        else:
            qkv = ttnn.experimental.minimal_matmul(input_tensor=xb, weight_tensor=wq,
                                                   compute_kernel_config=ckc, dtype=ttnn.bfloat16)
        if rows:
            ttnn.deallocate(xb)
        qkv = ttnn.unsqueeze(qkv, 1)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=HEADS, num_kv_heads=HEADS, transpose_k_heads=False,
            memory_config=qkv.memory_config())
        ttnn.deallocate(qkv)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=HD ** -0.5, program_config=sdpa_pc)
        for t in (q, k, v):
            ttnn.deallocate(t)
        oc = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o)
        outs.append(ttnn.squeeze(oc, 1))
    if not rows:
        return outs[0]
    r = ttnn.concat(outs, dim=0)
    for t in outs:
        ttnn.deallocate(t)
    return r


def timed(fn):
    for _ in range(args.warm):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(args.iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(r)
    return med(o)


# reference for bit-exactness
ref = ttnn.to_torch(chain(xn, 0))
res = {}
for nb in args.blocks:
    rows = 0 if nb == 0 else N // nb
    if rows and N % rows:
        continue
    cfg = None
    if rows:
        xb = ttnn.slice(xn, [0, 0, 0], [rows, N, C])
        cfg = T._l1_resident_linear_config(xb, wq, ttnn.bfloat16)
        ttnn.deallocate(xb)
    ms = timed(lambda r=rows: chain(xn, r))
    eq = bool(torch.equal(ref, ttnn.to_torch(chain(xn, rows))))
    tag = "whole" if not rows else f"{nb}x{rows}r"
    res[tag] = {"rows": rows, "ms": round(ms, 4), "l1": cfg is not None, "bit_exact": eq}
    print(f"  {tag:10s} rows={rows:4d} l1={'yes' if cfg is not None else 'NO ':3s} "
          f"{ms:8.4f} ms  bit_exact={eq}", flush=True)

base = res["whole"]["ms"]
print("=== vs whole ===", flush=True)
for tag, r in res.items():
    r["speedup"] = round(base / r["ms"], 4)
    print(f"  {tag:10s} {r['speedup']:6.3f}x", flush=True)
if args.out:
    json.dump({"n": N, "c": C, "grid": list(T.COMPUTE_GRID_MAIN), "res": res},
              open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
