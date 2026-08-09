#!/usr/bin/env python3
"""The remaining gap is the qkv result going to DRAM. Whether that is claimable depends on the
consumer: TriangleAttention feeds the projection straight into nlp_create_qkv_heads, which today
writes q, k and v back to DRAM. If create_heads forces a DRAM copy of an L1 input, the win
evaporates and the op-level number is a mirage.

So time the PAIR, not the linear: projection -> unsqueeze -> nlp_create_qkv_heads, at every
combination of result placement, and then the same with SDPA appended so the whole consumer chain
is covered. Shapes are the model's: x is (S, S, C_Z) with S as the batch, which is what
TriangleAttention hands to the projection.
"""
import argparse, json, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, _tri_att_sdpa_program_config
import tt_bio.tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--out", default=None)
args = ap.parse_args()

N, C_Z, H, D = args.n, 256, 8, 32
GF = 2 * (N * N) * C_Z * (3 * H * D) / 1e9
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def med(x):
    return sorted(x)[len(x) // 2]


def timed(dev, fn, warm=6, pipe=10, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(o)


dev = get_device()
gx, gy = T.COMPUTE_GRID_MAIN
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
x = ttnn.from_torch(torch.randn(N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
w = ttnn.from_torch(torch.randn(C_Z, 3 * H * D), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
bias = ttnn.from_torch(torch.randn(1, H, N, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
print(f"N={N} grid={gx}x{gy}  qkv result = {N*N*3*H*D*2/1e6:.2f} MB", flush=True)

# The explicit tall-narrow config, same one that was bit-exact with minimal_matmul last pass.
m_t, k_t, n_t = (N * N) // 32, C_Z // 32, (3 * H * D) // 32
per_core_M = next(p for p in range(-(-m_t // (gx * gy)), m_t + 1) if m_t % p == 0)
CFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(gx, gy), in0_block_w=k_t,
    out_subblock_h=1, out_subblock_w=4, out_block_h=per_core_M, out_block_w=n_t,
    per_core_M=per_core_M, per_core_N=n_t, fuse_batch=True, fused_activation=None, mcast_in0=False)
print(f"tall-narrow cfg: per_core_M={per_core_M} per_core_N={n_t} in0_block_w={k_t}", flush=True)


def mm_minimal(mm):
    return ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w,
                                            compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                            memory_config=mm)


def mm_linear(mm):
    return ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                       memory_config=mm, program_config=CFG)


def heads(qkv, mm):
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        ttnn.unsqueeze(qkv, 1), num_heads=H, num_kv_heads=H, transpose_k_heads=False,
        memory_config=mm)
    return q, k, v


CASES = [("minimal proj DRAM -> heads DRAM (production)", mm_minimal, DRAM, DRAM),
         ("minimal proj L1   -> heads DRAM", mm_minimal, L1, DRAM),
         ("minimal proj L1   -> heads L1", mm_minimal, L1, L1),
         ("linear+cfg proj DRAM -> heads DRAM", mm_linear, DRAM, DRAM),
         ("linear+cfg proj L1   -> heads L1", mm_linear, L1, L1)]

res, outs = {}, {}
for tag, mmfn, m_mm, m_hd in CASES:
    row = {}
    try:
        row["proj_ms"] = round(timed(dev, lambda: ttnn.deallocate(mmfn(m_mm))), 4)
        row["proj_tflops"] = round(GF / (row["proj_ms"] / 1e3) / 1e3, 2)

        def pair():
            for t in heads(mmfn(m_mm), m_hd):
                ttnn.deallocate(t)
        row["pair_ms"] = round(timed(dev, pair), 4)

        def chain():
            q, k, v = heads(mmfn(m_mm), m_hd)
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, scale=D ** -0.5,
                program_config=_tri_att_sdpa_program_config(q.shape[2], k.shape[2]))
            for t in (q, k, v, o):
                ttnn.deallocate(t)
        row["chain_ms"] = round(timed(dev, chain, warm=4, pipe=6), 4)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:100]}"
    try:
        qkv = heads(mmfn(m_mm), m_hd)
        outs[tag] = [ttnn.to_torch(t) for t in qkv]
        for t in qkv:
            ttnn.deallocate(t)
    except Exception:
        pass
    res[tag] = row
    print(f"  {tag:44s} proj {row.get('proj_ms','-'):>8}  pair {row.get('pair_ms','-'):>8}  "
          f"chain {row.get('chain_ms','-'):>8}  {row.get('error','')}", flush=True)

base = CASES[0][0]
print("\n=== vs production ===", flush=True)
for tag, row in res.items():
    for key in ("proj", "pair", "chain"):
        if f"{key}_ms" in row and f"{key}_ms" in res[base]:
            row[f"{key}_speedup"] = round(res[base][f"{key}_ms"] / row[f"{key}_ms"], 4)
    if tag in outs and base in outs:
        row["qkv_bit_exact"] = all(bool(torch.equal(a, b)) for a, b in zip(outs[tag], outs[base]))
    print(f"  {tag:44s} proj {row.get('proj_speedup','-'):>7}x  pair {row.get('pair_speedup','-'):>7}x  "
          f"chain {row.get('chain_speedup','-'):>7}x  bit_exact={row.get('qkv_bit_exact')}", flush=True)

if args.out:
    json.dump({"n": N, "grid": [gx, gy], "rows": res}, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
