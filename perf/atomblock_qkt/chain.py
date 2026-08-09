#!/usr/bin/env python3
"""The RFD3 atom-block attention score chain at its real 298-token shape, op by op.

E2 (`perfwar-rfd3-esmfold2-sites`) recorded `rfd3/model.py:1346` -- the atom-block q@k^T --
at Mt=Nt=126, 471.9 us/call, "no legal reuse arm: every per_core_M overflows the CB budget".
This script tests the competing explanation: the op is not config-limited at all, it is
DRAM-write-bound and already at its roof, because a [1,H,M,32] @ [1,H,32,N] matmul turns
2 MB of operands into an M*N score matrix.

Roofline, stated before the measurement (H=4, M=N=4032, head_dim=32, bf16):
  FLOPs  = 2*H*M*N*32                      = 4.16 GFLOP
  bytes  = in 2*H*M*32*2 + out H*M*N*2     = 2.06 MB in, 130.1 MB out
  AI     = 31.5 FLOP/byte  <<  ~350 balance  -> memory bound, and the traffic is 98% write
  floor  = 130.1 MB / write_roof
Everything downstream of it (the fp32 typecast, scale, bias add, softmax, cast back) moves
the same matrix 2-4 more times, so the chain is measured as a whole, not just the matmul.

Every timed region synchronises on both sides.
"""
import argparse, json, statistics as st, sys, time

import torch
import ttnn

from tt_bio.tenstorrent import get_device

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG

ap = argparse.ArgumentParser()
ap.add_argument("--mt", type=int, default=126, help="M and N in tiles (126 = 4032 atoms)")
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--head-dim", type=int, default=32)
ap.add_argument("--out", default=None)
ap.add_argument("--skip-roofs", action="store_true")
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

dev = get_device()
dg = dev.compute_with_storage_grid_size()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False, packer_l1_acc=False)

H, D = a.heads, a.head_dim
M = N = a.mt * 32
res = {"shape": {"heads": H, "M": M, "N": N, "head_dim": D,
                 "grid": f"{dg.x}x{dg.y}"}}
print(f"grid={dg.x}x{dg.y}  H={H} M={M} N={N} D={D}", flush=True)


def timed(fn, warm=3, pipe=3, reps=None):
    reps = reps or a.reps
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
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


# ---------------------------------------------------------------- roofs, this card
if not a.skip_roofs:
    print("=== directional DRAM roofs (same method as perf/ledger_298/roofs_card.py) ===", flush=True)
    rows = []
    for mb in (16, 32, 64):
        nrow = int(mb * 1e6 / 2) // 4096
        nbytes = nrow * 4096 * 2
        r = {"MB": round(nbytes / 1e6, 2)}
        xd = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)), pipe=4)
        r["read_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xd)
        xl = ttnn.from_torch(torch.randn(nrow, 4096), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev, memory_config=L1)
        t = timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)), pipe=4)
        r["write_GBs"] = round(nbytes / t / 1e9, 1)
        ttnn.deallocate(xl)
        rows.append(r)
        print("  " + json.dumps(r), flush=True)
    res["dram_roofs"] = {"runs": rows,
                         "read_peak_GBs": max(r["read_GBs"] for r in rows),
                         "write_peak_GBs": max(r["write_GBs"] for r in rows)}
    print(f"READ {res['dram_roofs']['read_peak_GBs']} GB/s  "
          f"WRITE {res['dram_roofs']['write_peak_GBs']} GB/s", flush=True)

W_ROOF = res.get("dram_roofs", {}).get("write_peak_GBs", 263.6) * 1e9
R_ROOF = res.get("dram_roofs", {}).get("read_peak_GBs", 386.8) * 1e9

# ---------------------------------------------------------------- operands
tq = torch.randn(1, H, M, D) * 0.1
tk = torch.randn(1, H, M, D) * 0.1
tv = torch.randn(1, H, M, D) * 0.1
tb = torch.randn(1, H, M, N) * 0.1
q = ttnn.from_torch(tq, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
k = ttnn.from_torch(tk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
v = ttnn.from_torch(tv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
bias16 = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
bias32 = ttnn.typecast(bias16, ttnn.float32, memory_config=DRAM)

SC = float(D) ** -0.5
score_bytes16 = H * M * N * 2
score_bytes32 = H * M * N * 4
print(f"score matrix: {score_bytes16/1e6:.1f} MB bf16 / {score_bytes32/1e6:.1f} MB fp32", flush=True)

steps = []


def record(name, fn, read_B, write_B, flops=0.0, keep=False):
    """Time one op, place it on the roof its traffic implies."""
    try:
        t = timed(fn if keep else (lambda: ttnn.deallocate(fn())))
    except Exception as e:
        print(f"  {name:22s} ERR {str(e)[:90]}", flush=True)
        steps.append({"op": name, "err": str(e)[:200]})
        return None
    row = {"op": name, "us": round(t * 1e6, 1),
           "read_MB": round(read_B / 1e6, 1), "write_MB": round(write_B / 1e6, 1),
           "GFLOP": round(flops / 1e9, 3),
           "pct_write_roof": round(100 * (write_B / t) / W_ROOF, 1) if write_B else None,
           "pct_read_roof": round(100 * (read_B / t) / R_ROOF, 1) if read_B else None,
           "eff_GBs": round((read_B + write_B) / t / 1e9, 1)}
    steps.append(row)
    print(f"  {name:22s} {t*1e6:9.1f} us   r{row['read_MB']:7.1f} w{row['write_MB']:7.1f} MB   "
          f"wroof {row['pct_write_roof']}%  rroof {row['pct_read_roof']}%", flush=True)
    return t


print("=== the shipped chain, op by op ===", flush=True)
kT = ttnn.permute(k, (0, 1, 3, 2))
record("permute k", lambda: ttnn.permute(k, (0, 1, 3, 2)),
       H * M * D * 2, H * M * D * 2)
record("1346 q@kT", lambda: ttnn.matmul(q, kT, compute_kernel_config=ckc),
       2 * H * M * D * 2, score_bytes16, 2 * H * M * N * D)
scores16 = ttnn.matmul(q, kT, compute_kernel_config=ckc)
record("typecast->f32", lambda: ttnn.typecast(scores16, ttnn.float32, memory_config=DRAM),
       score_bytes16, score_bytes32)
scores32 = ttnn.typecast(scores16, ttnn.float32, memory_config=DRAM)
record("multiply scale", lambda: ttnn.multiply(scores32, SC),
       score_bytes32, score_bytes32)
record("add bias", lambda: ttnn.add(scores32, bias32),
       2 * score_bytes32, score_bytes32)
biased = ttnn.add(scores32, bias32)
record("softmax f32", lambda: ttnn.softmax(biased, dim=-1),
       score_bytes32, score_bytes32)
attn32 = ttnn.softmax(biased, dim=-1)
record("typecast->bf16", lambda: ttnn.typecast(attn32, ttnn.bfloat16, memory_config=DRAM),
       score_bytes32, score_bytes16)
attn16 = ttnn.typecast(attn32, ttnn.bfloat16, memory_config=DRAM)
record("1358 attn@v", lambda: ttnn.matmul(attn16, v, compute_kernel_config=ckc, dtype=ttnn.bfloat16),
       score_bytes16 + H * M * D * 2, H * M * D * 2, 2 * H * M * N * D)

chain_us = sum(s["us"] for s in steps if "us" in s and s["op"] != "permute k")
print(f"CHAIN TOTAL (ex-permute) {chain_us:.1f} us", flush=True)
res["chain"] = steps
res["chain_total_us"] = round(chain_us, 1)

# ---------------------------------------------------------------- fused alternatives
print("=== fused alternatives ===", flush=True)
alts = {}


def alt(name, fn, note=""):
    try:
        t = timed(lambda: ttnn.deallocate(fn()))
    except Exception as e:
        print(f"  {name:28s} ERR {str(e)[:110]}", flush=True)
        alts[name] = {"err": str(e)[:250]}
        return None
    alts[name] = {"us": round(t * 1e6, 1), "note": note,
                  "speedup_vs_chain": round(chain_us / (t * 1e6), 2)}
    print(f"  {name:28s} {t*1e6:9.1f} us   {chain_us/(t*1e6):6.2f}x vs chain", flush=True)
    return t


# scale_mask_softmax fuses scale + additive mask + softmax into one pass over the matrix.
alt("scale_mask_softmax f32",
    lambda: ttnn.scale_mask_softmax(scores32, SC, bias32),
    "replaces multiply+add+softmax")
alt("scale_mask_softmax bf16",
    lambda: ttnn.scale_mask_softmax(scores16, SC, bias16),
    "and skips both typecasts")

sdpa = getattr(ttnn.transformer, "scaled_dot_product_attention", None)
if sdpa is not None:
    for qc, kc_ in ((128, 512), (256, 512), (512, 1024), (32, 4032)):
        try:
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=dg, q_chunk_size=qc, k_chunk_size=kc_,
                exp_approx_mode=False)
        except Exception as e:
            print(f"  SDPAProgramConfig({qc},{kc_}) ERR {str(e)[:80]}", flush=True)
            continue
        alt(f"sdpa mask=bf16 q{qc} k{kc_}",
            lambda pc=pc: sdpa(q, k, v, attn_mask=bias16, is_causal=False, scale=SC,
                               program_config=pc, compute_kernel_config=ckc),
            "flash attention, scores never reach DRAM")
    alt("sdpa mask=bf16 default",
        lambda: sdpa(q, k, v, attn_mask=bias16, is_causal=False, scale=SC,
                     compute_kernel_config=ckc),
        "flash attention, default chunking")
    alt("sdpa mask=f32 default",
        lambda: sdpa(q, k, v, attn_mask=bias32, is_causal=False, scale=SC,
                     compute_kernel_config=ckc),
        "fp32 mask")
else:
    print("  ttnn.transformer.scaled_dot_product_attention MISSING", flush=True)

res["alternatives"] = alts

# ---------------------------------------------------------------- roofline placement
w_floor_us = score_bytes16 / W_ROOF * 1e6
res["model"] = {
    "qkT_flops": 2 * H * M * N * D,
    "qkT_bytes_in": 2 * H * M * D * 2,
    "qkT_bytes_out": score_bytes16,
    "qkT_AI_flop_per_byte": round(2 * H * M * N * D / (2 * H * M * D * 2 + score_bytes16), 1),
    "qkT_write_floor_us": round(w_floor_us, 1),
    "write_roof_GBs": W_ROOF / 1e9,
    "read_roof_GBs": R_ROOF / 1e9,
}
print(json.dumps(res["model"], indent=2), flush=True)

if a.out:
    json.dump(res, open(a.out, "w"), indent=2)
    print("wrote " + a.out, flush=True)
