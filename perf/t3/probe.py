#!/usr/bin/env python3
"""T3 (transition + norms + PairformerLayer body) Phase-1 probe, qb2 card 1.

Everything here is measured in ONE process so the roofs and the ops share a device context and a
clock. Shapes are the shapes the fold actually runs, taken from perf/t3/ops_pv2_320_qb2c1.json
(pf_block_ops.py on this card): protenix-v2, N=320, c_z=256, Transition chunked at h=32 -> 10 chunks
of (1, 32, 320, 256), i.e. mt=320 kt=8 nt=32, NOT the flattened mt=3200 a microbenchmark would use.

Five products:
  A  DRAM read/write roof swept to 128 MB (the ledger's sweep stopped at 64 MB and had not saturated)
  B  the K-CORRECTED compute rate for every K these ops actually contract over
  C  standalone time for every T3 op at its in-fold shape + buffer types, including the 30 ops
     pf_block_ops.py scored 0.0 s because it could not re-run them under in-fold L1 pressure
  D  a same-bytes bandwidth control for each norm / residual / elementwise row
  E  a core_grid occupancy A/B for the four Transition linears (no device profiler on this host)
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
dg = dev.compute_with_storage_grid_size()
CKC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
out = {"grid": f"{dg.x}x{dg.y}", "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}"}
print(f"grid {dg.x}x{dg.y}  core_grid_main {CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)


def timed(fn, warm=4, pipe=5, reps=5):
    """Median of `reps` runs of `pipe` back-to-back calls. Synchronised on both sides."""
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


def T(shape, mc=DRAM, dt=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


# ---------------------------------------------------------------- A: DRAM roofs to 128 MB
print("\n=== A  DRAM roofs, swept to 128 MB ===", flush=True)
rows = []
for mb in (8, 16, 32, 64, 96, 128):
    nrow = int(mb * 1e6 / 2) // 4096
    nbytes = nrow * 4096 * 2
    r = {"MB": round(nbytes / 1e6, 2)}
    xd = T((nrow, 4096), DRAM)
    r["read_GBs"] = round(nbytes / timed(lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=L1)),
                                        warm=3, pipe=4) / 1e9, 1)
    r["d2d_rw_GBs"] = round(2 * nbytes / timed(
        lambda: ttnn.deallocate(ttnn.clone(xd, memory_config=DRAM)), warm=3, pipe=4) / 1e9, 1)
    ttnn.deallocate(xd)
    xl = T((nrow, 4096), L1)
    r["write_GBs"] = round(nbytes / timed(lambda: ttnn.deallocate(ttnn.clone(xl, memory_config=DRAM)),
                                          warm=3, pipe=4) / 1e9, 1)
    ttnn.deallocate(xl)
    rows.append(r)
    print("  " + json.dumps(r), flush=True)
READ = max(r["read_GBs"] for r in rows)
WRITE = max(r["write_GBs"] for r in rows)
D2D = max(r["d2d_rw_GBs"] for r in rows)
out["dram_sweep"] = {"rows": rows, "read_peak_GBs": READ, "write_peak_GBs": WRITE,
                     "dram2dram_rw_peak_GBs": D2D}
print(f"  READ {READ} GB/s  WRITE {WRITE} GB/s  DRAM->DRAM(r+w) {D2D} GB/s", flush=True)


# ---------------------------------------------------------------- B: K-corrected compute rate
print("\n=== B  compute rate vs K (HiFi4, fp32_dest_acc + packer_l1_acc, the model's ckc) ===",
      flush=True)
kroof = {}
for K in (256, 384, 1024, 1536, 4096):
    best = {"tflops": 0.0}
    for M, N in ((10240, 256), (10240, 1024), (10240, 4096), (4096, 4096)):
        a = T((1, 1, M, K), DRAM)
        b = T((1, 1, K, N), DRAM)
        gf = 2 * M * K * N / 1e9
        for lbl, kw in (("default", {}), ("core_grid", {"core_grid": CORE_GRID_MAIN})):
            try:
                s = timed(lambda: ttnn.deallocate(
                    ttnn.matmul(a, b, compute_kernel_config=CKC, memory_config=DRAM, **kw)),
                    warm=3, pipe=4, reps=5)
            except Exception as e:                                     # noqa: BLE001
                print(f"  K={K:<5} M={M} N={N:<5} {lbl:9s} ERR {str(e)[:60]}", flush=True)
                continue
            tf = gf / s / 1e3
            print(f"  K={K:<5} M={M} N={N:<5} {lbl:9s} {s * 1e6:9.1f} us {tf:8.2f} TFLOP/s",
                  flush=True)
            if tf > best["tflops"]:
                best = {"tflops": round(tf, 2), "M": M, "N": N, "cfg": lbl, "us": round(s * 1e6, 1)}
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    kroof[K] = best
    print(f"  K={K} CORRECTED COMPUTE ROOF {best['tflops']} TFLOP/s ({best['cfg']}, "
          f"M={best.get('M')} N={best.get('N')})", flush=True)
out["k_corrected_compute_TFLOPs"] = kroof
DENSE = kroof[4096]["tflops"]
out["machine_balance_FLOP_per_byte"] = round(DENSE * 1e12 / (READ * 1e9), 1)
print(f"  dense(K=4096) {DENSE} TFLOP/s   machine balance {out['machine_balance_FLOP_per_byte']} "
      f"FLOP/byte against the measured read roof", flush=True)


# ---------------------------------------------------------------- C/D/E: the ops
print("\n=== C/D/E  T3 ops at their in-fold shapes ===", flush=True)
ops = {}


def rec(key, s, flops, dram_bytes, all_bytes, note=""):
    e = {"us": round(s * 1e6, 2), "GFLOP": round(flops / 1e9, 4),
         "dram_MB": round(dram_bytes / 1e6, 3), "all_MB": round(all_bytes / 1e6, 3),
         "TFLOPs": round(flops / s / 1e12, 2) if flops else 0.0,
         "dram_GBs": round(dram_bytes / s / 1e9, 1) if dram_bytes else 0.0,
         "all_GBs": round(all_bytes / s / 1e9, 1),
         "AI_dram": round(flops / dram_bytes, 1) if dram_bytes else None,
         "AI_all": round(flops / all_bytes, 2) if all_bytes else None, "note": note}
    ops[key] = e
    print(f"  {key:<34} {e['us']:9.2f} us  {e['TFLOPs']:7.2f} TF/s  dram {e['dram_GBs']:7.1f} GB/s  "
          f"all {e['all_GBs']:7.1f} GB/s  AI_dram {e['AI_dram']}  {note}", flush=True)


B2 = 2  # bf16
# --- the Transition z chunk: (1, 32, 320, 256), 10 of them per Transition call ---
CH = (1, 32, 320, 256)
M_CH, K1, N1 = 32 * 320, 256, 1024
x_ch = T(CH, DRAM)
wn = T((32, 256), DRAM)
bn = T((32, 256), DRAM)
w1 = T((256, 1024), DRAM)
w3 = T((1024, 256), DRAM)

# layer_norm @2038, DRAM -> L1
rec("transition_z.layer_norm@2038",
    timed(lambda: ttnn.deallocate(ttnn.layer_norm(x_ch, weight=wn, bias=bn, epsilon=1e-5,
                                                 compute_kernel_config=CKC, memory_config=L1))),
    0, M_CH * K1 * B2, 2 * M_CH * K1 * B2, "in DRAM, out L1; ledger scored this 0.0 s")
# same-bytes control: a plain DRAM->L1 clone moves exactly the read half
rec("CONTROL clone DRAM->L1 same bytes",
    timed(lambda: ttnn.deallocate(ttnn.clone(x_ch, memory_config=L1))),
    0, M_CH * K1 * B2, 2 * M_CH * K1 * B2, "the norm's copy-roof control")

x_norm = ttnn.clone(x_ch, memory_config=L1)
# fc1 (silu) and fc2 @2046/@2055: L1 in, DRAM weight, L1 out
for lbl, act in (("transition_z.fc1_silu@2046", "silu"), ("transition_z.fc2@2055", None)):
    kw = {"activation": act} if act else {}
    rec(lbl, timed(lambda: ttnn.deallocate(ttnn.linear(
            x_norm, w1, compute_kernel_config=CKC, memory_config=L1, core_grid=CORE_GRID_MAIN,
            **kw))),
        2 * M_CH * K1 * N1, K1 * N1 * B2, (M_CH * K1 + K1 * N1 + M_CH * N1) * B2,
        "in/out L1, weight DRAM; ledger scored this 0.0 s")
# the silu premium, priced against the same matmul
rec("transition_z.fc1 NO activation (control)",
    timed(lambda: ttnn.deallocate(ttnn.linear(x_norm, w1, compute_kernel_config=CKC,
                                              memory_config=L1, core_grid=CORE_GRID_MAIN))),
    2 * M_CH * K1 * N1, K1 * N1 * B2, (M_CH * K1 + K1 * N1 + M_CH * N1) * B2,
    "prices activation='silu' against the identical matmul")

h1 = T((1, 32, 320, 1024), L1)
h2 = T((1, 32, 320, 1024), L1)
rec("transition_z.multiply_@2064",
    timed(lambda: ttnn.multiply_(h1, h2)), M_CH * N1,
    0, 3 * M_CH * N1 * B2, "all three operands L1")
rec("CONTROL clone L1->L1 same bytes", timed(lambda: ttnn.deallocate(ttnn.clone(h1, memory_config=L1))),
    0, 0, 2 * M_CH * N1 * B2, "L1 copy roof for the multiply_")
rec("transition_z.fc3@2066",
    timed(lambda: ttnn.deallocate(ttnn.linear(h1, w3, compute_kernel_config=CKC,
                                              memory_config=DRAM, core_grid=CORE_GRID_MAIN))),
    2 * M_CH * N1 * K1, (N1 * K1 + M_CH * K1) * B2,
    (M_CH * N1 + N1 * K1 + M_CH * K1) * B2, "in L1, out DRAM; W1's AI-930.9 row")

# E: occupancy A/B on fc1 and fc3 at the REAL chunked shape
print("\n  --- E  core_grid occupancy A/B, real chunk shape mt=320 kt=8 nt=32 ---", flush=True)
occ = {}
for lbl, fn in (("fc1 mt320 kt8 nt32", lambda g: ttnn.linear(
                    x_norm, w1, compute_kernel_config=CKC, memory_config=L1, core_grid=g)),
                ("fc3 mt320 kt32 nt8", lambda g: ttnn.linear(
                    h1, w3, compute_kernel_config=CKC, memory_config=DRAM, core_grid=g))):
    series = []
    for gx, gy in ((2, 2), (4, 4), (6, 6), (8, 8), (10, 10), (11, 10)):
        g = ttnn.CoreGrid(x=gx, y=gy)
        try:
            s = timed(lambda: ttnn.deallocate(fn(g)), warm=3, pipe=4, reps=5)
        except Exception as e:                                         # noqa: BLE001
            print(f"    {lbl:20s} {gx}x{gy:<3} ERR {str(e)[:50]}", flush=True)
            continue
        series.append({"cores": gx * gy, "grid": f"{gx}x{gy}", "us": round(s * 1e6, 2)})
        print(f"    {lbl:20s} {gx}x{gy:<4} {gx * gy:4d} cores {s * 1e6:9.2f} us", flush=True)
    occ[lbl] = series
    if len(series) >= 2:
        base, top = series[0], series[-1]
        ideal = base["us"] * base["cores"] / top["cores"]
        print(f"    {lbl:20s} scaling {base['cores']}->{top['cores']} cores: "
              f"{base['us'] / top['us']:.2f}x actual vs {base['cores'] / top['cores']:.2f}x ideal "
              f"=> {100 * ideal / top['us']:.0f}% of linear scaling", flush=True)
out["occupancy_ab"] = occ

ttnn.deallocate(x_norm)
ttnn.deallocate(h1)
ttnn.deallocate(h2)
ttnn.deallocate(x_ch)

# --- ttnn.chunk / ttnn.concat, the chunking's own overhead ---
z = T((1, 320, 320, 256), DRAM)
ZB = 320 * 320 * 256 * B2
rec("Transition.chunk@2145", timed(lambda: [ttnn.deallocate(c) for c in ttnn.chunk(z, 10, dim=1)]
                                   and None, warm=2, pipe=3, reps=3),
    0, 2 * ZB, 2 * ZB, "materialises a full second copy of z; 1 call per Transition")
parts = ttnn.chunk(z, 10, dim=1)
rec("Transition.concat@2148", timed(lambda: ttnn.deallocate(ttnn.concat(list(parts), dim=1)),
                                    warm=2, pipe=3, reps=3),
    0, 2 * ZB, 2 * ZB, "reassembles the 10 chunks; 1 call per Transition")
for c in parts:
    ttnn.deallocate(c)

# --- PairformerLayer body: the 5 residual adds and the two norms ---
z2 = T((1, 320, 320, 256), DRAM)
rec("PairformerLayer.add_ x5 @2223..2239", timed(lambda: ttnn.add_(z, z2)),
    320 * 320 * 256, 3 * ZB, 3 * ZB, "in-place: 2 reads + 1 write, all DRAM")
rec("CONTROL clone DRAM->DRAM same bytes",
    timed(lambda: ttnn.deallocate(ttnn.clone(z, memory_config=DRAM))),
    0, 2 * ZB, 2 * ZB, "W6's copy-roof control for the adds, re-measured on this card")
wz = T((32, 256), DRAM)
bz = T((32, 256), DRAM)
rec("AttentionPairBias.z_norm@1893",
    timed(lambda: ttnn.deallocate(ttnn.layer_norm(z, weight=wz, bias=bz, epsilon=1e-5,
                                                  compute_kernel_config=CKC))),
    0, 2 * ZB, 2 * ZB, "(1,320,320,256) DRAM->DRAM layernorm, the biggest norm in the block")
wzp = T((256, 32), DRAM)
rec("AttentionPairBias.z_proj@1900",
    timed(lambda: ttnn.deallocate(ttnn.linear(z, wzp, compute_kernel_config=CKC,
                                              core_grid=CORE_GRID_MAIN))),
    2 * 320 * 320 * 256 * 32, (ZB + 256 * 32 * B2 + 320 * 320 * 32 * B2),
    (ZB + 256 * 32 * B2 + 320 * 320 * 32 * B2), "c_z=256 -> 16 heads of bias, nt=1")
ttnn.deallocate(z2)
ttnn.deallocate(z)

# --- the s track: pre_norm_s, transition_s, the APB projections ---
s = T((1, 320, 384), DRAM)
SB = 320 * 384 * B2
ws = T((32, 384), DRAM)
bs = T((32, 384), DRAM)
rec("PairformerLayer.pre_norm_s@2242",
    timed(lambda: ttnn.deallocate(ttnn.layer_norm(s, weight=ws, bias=bs, epsilon=1e-5,
                                                  compute_kernel_config=CKC))),
    0, 2 * SB, 2 * SB, "(1,320,384) -- 0.246 MB, far under any bandwidth roof")
rec("CONTROL clone (1,320,384) DRAM->DRAM",
    timed(lambda: ttnn.deallocate(ttnn.clone(s, memory_config=DRAM))),
    0, 2 * SB, 2 * SB, "same bytes as pre_norm_s: isolates the per-op floor")
wq = T((384, 1536), DRAM)
bq = T((32, 1536), DRAM)
rec("AttentionPairBias.qkv@1876",
    timed(lambda: ttnn.deallocate(ttnn.linear(s, wq, bias=bq, compute_kernel_config=CKC,
                                              core_grid=CORE_GRID_MAIN))),
    2 * 320 * 384 * 1536, (SB + 384 * 1536 * B2 + 320 * 1536 * B2),
    (SB + 384 * 1536 * B2 + 320 * 1536 * B2), "mt=10 kt=12 nt=48")
wo = T((384, 384), DRAM)
rec("AttentionPairBias.o_proj@2001",
    timed(lambda: ttnn.deallocate(ttnn.linear(s, wo, compute_kernel_config=CKC,
                                              core_grid=CORE_GRID_MAIN))),
    2 * 320 * 384 * 384, (SB + 384 * 384 * B2 + SB), (SB + 384 * 384 * B2 + SB), "mt=10 kt=12 nt=12")
# transition_s: the 3D path, one swiglu, no chunking
ws1 = T((384, 1536), DRAM)
ws3 = T((1536, 384), DRAM)
rec("transition_s.layer_norm@2038",
    timed(lambda: ttnn.deallocate(ttnn.layer_norm(s, weight=ws, bias=bs, epsilon=1e-5,
                                                  compute_kernel_config=CKC, memory_config=L1))),
    0, SB, 2 * SB, "3D path, no chunking")
s_norm = ttnn.clone(s, memory_config=L1)
rec("transition_s.fc1_silu@2046",
    timed(lambda: ttnn.deallocate(ttnn.linear(s_norm, ws1, activation="silu",
                                              compute_kernel_config=CKC, memory_config=L1,
                                              core_grid=CORE_GRID_MAIN))),
    2 * 320 * 384 * 1536, 384 * 1536 * B2, (SB + 384 * 1536 * B2 + 320 * 1536 * B2), "mt=10")
hs = T((1, 320, 1536), L1)
rec("transition_s.multiply_@2064", timed(lambda: ttnn.multiply_(hs, hs)),
    320 * 1536, 0, 3 * 320 * 1536 * B2, "")
rec("transition_s.fc3@2066",
    timed(lambda: ttnn.deallocate(ttnn.linear(hs, ws3, compute_kernel_config=CKC,
                                              memory_config=DRAM, core_grid=CORE_GRID_MAIN))),
    2 * 320 * 1536 * 384, (1536 * 384 * B2 + SB),
    (320 * 1536 * B2 + 1536 * 384 * B2 + SB), "mt=10 kt=48 nt=12")

out["ops"] = ops
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("\nwrote " + sys.argv[1], flush=True)
