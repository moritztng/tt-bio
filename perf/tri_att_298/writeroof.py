#!/usr/bin/env python3
"""Close T2's Q4 caveat on the card it applies to, and two loose ends in T1's own rows.

T2 settled Q4 on qb2 at ttnn 0.68.0: the DRAM write roof is not per-NOC (every writer in the build
compiles to brisc / NOC 0), it is per WRITER STRUCTURE -- 257.2 GB/s from a unary writer against
168.5 GB/s from a matmul writer. T2 flagged the one thing it could not settle: W3's original
283 / 171 split was measured on **qb1 at 0.67.4**, and if 0.67.4 placed some writer on NCRISC the
split could have been real there. T1 is on qb1 at 0.67.4, so this reproduces T2's table on the card
and version the caveat is about.

Also here:
  - `clone` DRAM -> L1, the non-transposing control the C2FIX residual needs (T1 had the
    DRAM -> DRAM control only, so "what binds the remaining 471.7 us" was argued, not measured).
  - the SDPA chunk sweep WITHOUT the bias mask, which is the only way to explain why
    `_tri_att_sdpa_program_config`'s comment claims q=k=64 is 2.45x faster when with the mask it is
    1.65x slower.
"""
import json, statistics as st, sys, time
from pathlib import Path
import torch, ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN, COMPUTE_GRID_MAIN  # noqa: E402

DEV = get_device()
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
ckc = ttnn.init_device_compute_kernel_config(
    DEV.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
R = {}


def timed(fn, warm=3, pipe=4, reps=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(DEV)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(DEV)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def T(shape, mc=DRAM):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=DEV,
                           dtype=ttnn.bfloat16, memory_config=mc)


# ---- A. write roof by writer structure, T2's shapes, on qb1 at 0.67.4 -------------------------
print("=== A. write roof by writer structure (qb1 card 2, 0.67.4) ===", flush=True)
M, K, N = 102400, 32, 256
OUT_B = M * N * 2
a0, w0 = T((M, K)), T((K, N))
paths = {}
paths["minimal_matmul"] = timed(lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
    input_tensor=a0, weight_tensor=w0, compute_kernel_config=ckc, dtype=ttnn.bfloat16)))
paths["linear_core_grid"] = timed(lambda: ttnn.deallocate(ttnn.linear(
    a0, w0, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN,
    memory_config=DRAM)))
gx, gy = COMPUTE_GRID_MAIN
pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(gx, gy), in0_block_w=1, out_subblock_h=1, out_subblock_w=4,
    out_block_h=32, out_block_w=8, per_core_M=32, per_core_N=8, fuse_batch=True,
    fused_activation=None, mcast_in0=False)
try:
    paths["linear_1d_mcast"] = timed(lambda: ttnn.deallocate(ttnn.linear(
        a0, w0, compute_kernel_config=ckc, dtype=ttnn.bfloat16, program_config=pc,
        memory_config=DRAM)))
except Exception as e:                                                    # noqa: BLE001
    print("  linear_1d_mcast ERR", str(e)[:120], flush=True)
R["matmul_writer"] = {k: {"ms": round(v * 1e3, 4), "write_GBs": round(OUT_B / v / 1e9, 1),
                         "rw_GBs": round((OUT_B + M * K * 2 + K * N * 2) / v / 1e9, 1)}
                      for k, v in paths.items()}
ttnn.deallocate(a0); ttnn.deallocate(w0)

uw = []
for mb in (8, 16, 32, 64, 128):
    nrow = int(mb * 1e6 / 2) // 4096
    nb = nrow * 4096 * 2
    xl = T((nrow, 4096), L1)
    t = timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)))
    uw.append({"MB": round(nb / 1e6, 2), "write_GBs": round(nb / t / 1e9, 1)})
    ttnn.deallocate(xl)
R["unary_writer"] = uw
for k, v in R["matmul_writer"].items():
    print(f"  {k:20s} {json.dumps(v)}", flush=True)
print("  unary: " + json.dumps(uw), flush=True)

# ---- B. the non-transposing DRAM -> L1 control for the C2FIX residual --------------------------
print("\n=== B. permute vs clone, both destinations, fold's true shape ===", flush=True)
Mt, Nt, CZ = 298, 320, 256
x = T((Mt, Nt, CZ))
by = Mt * Nt * CZ * 2
ctl = {}
ctl["permute_to_DRAM"] = timed(lambda: ttnn.deallocate(ttnn.permute(x, (1, 0, 2))))
ctl["permute_to_L1"] = timed(lambda: ttnn.deallocate(ttnn.permute(x, (1, 0, 2), memory_config=L1)))
ctl["clone_to_DRAM"] = timed(lambda: ttnn.deallocate(ttnn.clone(x, memory_config=DRAM)))
ctl["clone_to_L1"] = timed(lambda: ttnn.deallocate(ttnn.clone(x, memory_config=L1)))
R["permute_control"] = {k: {"us": round(v * 1e6, 1), "rw_GBs": round(2 * by / v / 1e9, 1)}
                        for k, v in ctl.items()}
for k, v in R["permute_control"].items():
    print(f"  {k:18s} {json.dumps(v)}", flush=True)
ttnn.deallocate(x)

# ---- C. SDPA chunk sweep with and without the bias, to explain the in-code comment ------------
print("\n=== C. SDPA chunk sweep, mask on vs mask off ===", flush=True)
NH, HD = 8, 32
q, k, v = (T((Mt, NH, Nt, HD)) for _ in range(3))
mb_ = T((1, NH, Nt, Nt))
sw = []
for c in (64, 128, 256, 320):
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=COMPUTE_GRID_MAIN,
                                 exp_approx_mode=False, q_chunk_size=c, k_chunk_size=c)
    row = {"chunk": c}
    for lbl, m in (("mask_us", mb_), ("nomask_us", None)):
        try:
            t = timed(lambda: ttnn.deallocate(ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=m, is_causal=False, scale=HD ** -0.5, program_config=cfg)),
                warm=2, pipe=2, reps=5)
            row[lbl] = round(t * 1e6, 1)
        except Exception as e:                                            # noqa: BLE001
            row[lbl] = str(e)[:60]
    sw.append(row)
    print("  " + json.dumps(row), flush=True)
R["chunk_sweep_mask_ab"] = sw

json.dump(R, open(REPO / "perf/tri_att_298/writeroof_c2.json", "w"), indent=1)
print("\nwrote perf/tri_att_298/writeroof_c2.json", flush=True)
