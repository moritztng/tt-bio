#!/usr/bin/env python3
"""How large a token count can hold the qkv result in L1?

The result is S^2 * 3*H*D elements: 25.17 MB at S=128, 157 MB at S=320 against ~182 MB of
aggregate unreserved L1 for the whole program. So the L1 path needs a budget, established the way
the trimul chunk budget was: run it until it throws, and take the last size that both fits and wins.
"""
import argparse, json, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, _tri_att_sdpa_program_config
import tt_bio.tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 160, 192, 224, 256, 288, 320, 384])
ap.add_argument("--out", default=None)
args = ap.parse_args()

C_Z, H, D = 256, 8, 32
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def med(x):
    return sorted(x)[len(x) // 2]


def timed(dev, fn, warm=4, pipe=6, reps=5):
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
try:
    l1_per_core = int(ttnn.get_max_worker_l1_unreserved_size())
except Exception:
    l1_per_core = 0
print(f"grid={gx}x{gy}  unreserved L1/core={l1_per_core/1e6:.3f} MB  aggregate={gx*gy*l1_per_core/1e6:.1f} MB",
      flush=True)
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rows = {}

for N in args.sizes:
    m_t, k_t, n_t = (N * N) // 32, C_Z // 32, (3 * H * D) // 32
    per_core_M = next((p for p in range(max(1, -(-m_t // (gx * gy))), m_t + 1) if m_t % p == 0), 0)
    row = {"result_MB": round(N * N * 3 * H * D * 2 / 1e6, 2), "per_core_M": per_core_M}
    if not per_core_M or -(-m_t // per_core_M) > gx * gy:
        row["error"] = "no legal per_core_M"
        rows[N] = row
        print(f"  N={N:<4} {row}", flush=True)
        continue
    # Widest K block whose circular buffers fit. The fixed part (output + fp32 accumulation)
    # grows with per_core_M, so a larger S has less room for K blocking, not more.
    fixed = per_core_M * n_t * (2048 + 4096) + 128 * 1024
    per_block = (per_core_M + n_t) * 2048
    in0_block_w = next((d for d in range(k_t, 0, -1)
                        if k_t % d == 0 and fixed + d * per_block <= l1_per_core), 0)
    row["in0_block_w"] = in0_block_w
    if not in0_block_w:
        row["error"] = "no in0_block_w fits L1"
        rows[N] = row
        print(f"  N={N:<4} {row}", flush=True)
        continue
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=in0_block_w,
        out_subblock_h=1, out_subblock_w=4, out_block_h=per_core_M, out_block_w=n_t,
        per_core_M=per_core_M, per_core_N=n_t, fuse_batch=True, fused_activation=None, mcast_in0=False)
    try:
        x = ttnn.from_torch(torch.randn(N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn(C_Z, 3 * H * D), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        bias = ttnn.from_torch(torch.randn(1, H, N, N), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    except Exception as e:
        row["error"] = f"operand alloc: {str(e)[:80]}"
        rows[N] = row
        print(f"  N={N:<4} {row}", flush=True)
        continue

    def chain(mode):
        m_qkv = DRAM if mode == "prod" else L1
        m_hd = L1 if mode == "both_l1" else DRAM
        qkv = (ttnn.experimental.minimal_matmul(input_tensor=x, weight_tensor=w, dtype=ttnn.bfloat16,
                                                compute_kernel_config=ckc, memory_config=m_qkv)
               if mode == "prod" else
               ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                           memory_config=m_qkv, program_config=cfg))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(qkv, 1), num_heads=H, num_kv_heads=H, transpose_k_heads=False,
            memory_config=m_hd)
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=D ** -0.5,
            program_config=_tri_att_sdpa_program_config(q.shape[2], k.shape[2]))
        for t in (q, k, v, o):
            ttnn.deallocate(t)

    for lbl, mode in (("prod_ms", "prod"), ("l1_ms", "both_l1"), ("l1_qkv_only_ms", "qkv_l1")):
        try:
            row[lbl] = round(timed(dev, lambda: chain(mode)), 4)
        except Exception as e:
            row[lbl] = None
            row[lbl + "_error"] = f"{type(e).__name__}: {str(e)[:90]}"
    if row.get("prod_ms"):
        for lbl in ("l1_ms", "l1_qkv_only_ms"):
            if row.get(lbl):
                row[lbl.replace("_ms", "") + "_speedup"] = round(row["prod_ms"] / row[lbl], 4)
        row["speedup"] = row.get("l1_speedup") or row.get("l1_qkv_only_speedup")
        # bit-exactness of the qkv projection itself at this size
        try:
            a = ttnn.to_torch(ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc, memory_config=DRAM))
            b = ttnn.to_torch(ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                          memory_config=L1 if row.get("l1_ms") else DRAM,
                                          program_config=cfg))
            row["bit_exact"] = bool(torch.equal(a, b))
        except Exception as e:
            row["bit_exact_error"] = str(e)[:80]
    for t in (x, w, bias):
        ttnn.deallocate(t)
    rows[N] = row
    print(f"  N={N:<4} {row['result_MB']:7.2f} MB bw={row.get('in0_block_w')} pcM={per_core_M}  "
          f"prod {str(row.get('prod_ms')):>8}  bothL1 {str(row.get('l1_ms')):>8} "
          f"({row.get('l1_speedup','-')}x)  qkvL1 {str(row.get('l1_qkv_only_ms')):>8} "
          f"({row.get('l1_qkv_only_speedup','-')}x)  exact={row.get('bit_exact')}", flush=True)

fits = [n for n, r in rows.items() if (r.get("speedup") or 0) > 1.0]
print(f"\nL1 path wins at S = {fits}; largest = {max(fits) if fits else None}", flush=True)
if args.out:
    json.dump({"grid": [gx, gy], "l1_per_core": l1_per_core, "rows": rows,
               "largest_winning_S": max(fits) if fits else None}, open(args.out, "w"), indent=2)
    print("wrote", args.out, flush=True)
