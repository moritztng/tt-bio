#!/usr/bin/env python3
"""Two things the qkv root-cause pass still owed:

(a) MEASURE the DRAM bandwidth roof on this card instead of inheriting 435.2 GB/s (which was
    asserted in a planning doc, never benchmarked) or the 512 GB/s datasheet figure.
(b) Check the NUMERICS of the placement/config ladder in the state doc, which timed four
    configurations but never looked at their outputs.

Every timed region syncs the device immediately before the clock starts and immediately before it
stops, so queued work is charged to the region that issued it.
"""
import json, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

N, C_Z, H, D = 128, 256, 8, 32
GF = 2 * (N * N) * C_Z * (3 * H * D) / 1e9
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=8, pipe=12, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(out)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


dev = get_device()
dg = dev.compute_with_storage_grid_size()
res = {"device": {"compute_grid": f"{dg.x}x{dg.y}",
                  "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}",
                  "ttnn": ttnn.__version__ if hasattr(ttnn, "__version__") else "?"}}
print(f"compute_grid={dg.x}x{dg.y}  CORE_GRID_MAIN={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}", flush=True)

# ---------------------------------------------------------------- (a) DRAM bandwidth roof
# Streaming, DRAM-interleaved, whole chip, far beyond the 190 MB of aggregate L1 at the top sizes.
# read+write  = clone;  2 reads + 1 write = add;  write-dominated = fill via ttnn.full_like.
print("\n=== DRAM streaming bandwidth ===", flush=True)
bw = {}
for r, c in ((4096, 4096), (8192, 4096), (8192, 8192), (16384, 8192)):
    nb = r * c * 2
    try:
        a = ttnn.ones((1, 1, r, c), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
        b = ttnn.ones((1, 1, r, c), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=DRAM)
    except Exception as e:
        print(f"  {r}x{c} alloc failed: {str(e)[:80]}", flush=True)
        break
    for name, fn, traffic in (
        ("clone   (1r+1w)", lambda: ttnn.deallocate(ttnn.clone(a, memory_config=DRAM)), 2 * nb),
        ("add     (2r+1w)", lambda: ttnn.deallocate(ttnn.add(a, b, memory_config=DRAM)), 3 * nb),
        ("mul     (2r+1w)", lambda: ttnn.deallocate(ttnn.multiply(a, b, memory_config=DRAM)), 3 * nb),
    ):
        try:
            ms = timed(dev, fn, warm=4, pipe=6, reps=5)
        except Exception as e:
            print(f"  {name} {r}x{c} ERR {str(e)[:70]}", flush=True)
            continue
        gbs = traffic / (ms / 1e3) / 1e9
        bw[f"{name.split()[0]}_{r}x{c}"] = {"MB": round(traffic / 1e6, 1), "ms": round(ms, 4), "GBs": round(gbs, 1)}
        print(f"  {name}  {r}x{c:<6} {traffic/1e6:8.1f} MB {ms:9.4f} ms {gbs:7.1f} GB/s", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
peak = max((v["GBs"] for v in bw.values()), default=0.0)
res["dram_bandwidth"] = {"runs": bw, "peak_GBs": peak, "datasheet_GBs": 512.0,
                         "pct_of_datasheet": round(100 * peak / 512.0, 1)}
print(f"MEASURED_DRAM_BW_PEAK {peak:.1f} GB/s = {100*peak/512.0:.1f}% of the 512 GB/s datasheet figure", flush=True)

# ---------------------------------------------------------------- (b) numerics of the ladder
print("\n=== ladder: time + numerics ===", flush=True)
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
at = torch.randn(1, N, N, C_Z)
wt = torch.randn(C_Z, 3 * H * D)
ref = (at.to(torch.bfloat16).float().reshape(-1, C_Z) @ wt.to(torch.bfloat16).float()).reshape(1, N, N, 3 * H * D)

BEST = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(dg.x, dg.y), in0_block_w=8,
    out_subblock_h=1, out_subblock_w=4, out_block_h=4, out_block_w=24,
    per_core_M=4, per_core_N=24, fuse_batch=True, fused_activation=None, mcast_in0=False)
BW1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
    compute_with_storage_grid_size=(dg.x, dg.y), in0_block_w=1,
    out_subblock_h=1, out_subblock_w=4, out_block_h=4, out_block_w=24,
    per_core_M=4, per_core_N=24, fuse_batch=True, fused_activation=None, mcast_in0=False)

ladder = [("prod_auto_xDRAM_oDRAM", None, DRAM, DRAM),
          ("bw1_g13x10_xDRAM_oDRAM", BW1, DRAM, DRAM),
          ("bw8_g13x10_xDRAM_oDRAM", BEST, DRAM, DRAM),
          ("bw8_g13x10_xDRAM_oL1", BEST, DRAM, L1),
          ("bw8_g13x10_xL1_oL1", BEST, L1, L1),
          ("bw8_g13x10_xL1_oDRAM", BEST, L1, DRAM)]

rows, base_out = {}, None
for name, cfg, xmem, omem in ladder:
    try:
        x = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=xmem)
        w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        kw = {"program_config": cfg} if cfg is not None else {"core_grid": CORE_GRID_MAIN}
        call = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=omem, **kw)
        o = call()
        t = ttnn.to_torch(o).float()
        ttnn.deallocate(o)
        ms = timed(dev, lambda: ttnn.deallocate(call()))
        ttnn.deallocate(x)
        ttnn.deallocate(w)
    except Exception as e:
        rows[name] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"  {name:26s} ERR {str(e)[:90]}", flush=True)
        continue
    exact = None if base_out is None else bool(torch.equal(t, base_out))
    if base_out is None:
        base_out = t
    tf = GF / (ms / 1e3) / 1e3
    rows[name] = {"ms": round(ms, 4), "tflops": round(tf, 2),
                  "pcc_vs_fp32_torch": round(pcc(t, ref), 8),
                  "max_abs_err_vs_fp32_torch": round(float((t - ref).abs().max()), 5),
                  "rel_fro_err": round(float((t - ref).norm() / ref.norm()), 8),
                  "bit_exact_vs_production": exact}
    print(f"  {name:26s} {ms:8.4f} ms {tf:7.2f} TF/s  pcc={rows[name]['pcc_vs_fp32_torch']:.8f} "
          f"maxabs={rows[name]['max_abs_err_vs_fp32_torch']:.5f} exact_vs_prod={exact}", flush=True)
res["ladder"] = rows

# roofline restated against the MEASURED roof
byt = ((N * N * C_Z) + (C_Z * 3 * H * D) + (N * N * 3 * H * D)) * 2
ai = GF * 1e9 / byt
res["roofline"] = {"GFLOP": round(GF, 4), "dram_bytes_MB": round(byt / 1e6, 3),
                   "arith_intensity_FLOP_per_byte": round(ai, 1),
                   "compute_roof_TFLOPs": 100.6,
                   "measured_bw_roof_GBs": peak,
                   "machine_balance_FLOP_per_byte": round(100.6 / (peak / 1000), 1) if peak else None,
                   "roofline_at_AI_TFLOPs": round(min(100.6, ai * peak / 1000), 2) if peak else None}
print("\nroofline vs MEASURED roof:", json.dumps(res["roofline"], indent=2), flush=True)

json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/roof_numerics.json", "w"), indent=2)
print("\nwrote", sys.argv[1] if len(sys.argv) > 1 else "/tmp/roof_numerics.json", flush=True)
