#!/usr/bin/env python3
"""Mechanism test without a device profiler.

The claim is that the qkv projection's DRAM-out penalty IS its result write, not stalls, not
occupancy, not scheduling. That claim is quantitative and falsifiable: the matmul's oDRAM-minus-oL1
time must equal the cost of moving exactly the result's bytes to DRAM instead of to L1, measured
separately on a pure data-movement op of the same size.

  predicted penalty = 25.17 MB / write-to-DRAM rate  -  25.17 MB / write-to-L1 rate

If the measured matmul penalty comes out well above that, something else is also going on and the
DRAM-write explanation is incomplete. Also dumps the tt-metal device profiler if the build has one.
"""
import json, os, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.tenstorrent as T

N, C_Z, H, D = 128, 256, 8, 32
GF = 2 * (N * N) * C_Z * (3 * H * D) / 1e9
RESULT_BYTES = N * N * 3 * H * D * 2
DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG


def med(x):
    return sorted(x)[len(x) // 2]


def timed(dev, fn, warm=8, pipe=12, reps=7):
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
cfg = T._l1_resident_matmul_config((N * N) // 32, C_Z // 32, (3 * H * D) // 32, 2, True)
print(f"grid={gx}x{gy}  result={RESULT_BYTES/1e6:.2f} MB", flush=True)
res = {"result_MB": round(RESULT_BYTES / 1e6, 2)}

# --- the matmul penalty, identical kernel, only the destination differs ------------------------
mm = {}
for tag, mem in (("oDRAM", DRAM), ("oL1", L1)):
    ms = timed(dev, lambda: ttnn.deallocate(
        ttnn.linear(x, w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                    memory_config=mem, program_config=cfg)))
    mm[tag] = {"ms": round(ms, 5), "tflops": round(GF / (ms / 1e3) / 1e3, 2)}
    print(f"  matmul -> {tag:6s} {ms:8.5f} ms  {mm[tag]['tflops']:7.2f} TFLOP/s", flush=True)
penalty = mm["oDRAM"]["ms"] - mm["oL1"]["ms"]
res["matmul"] = mm
res["measured_penalty_ms"] = round(penalty, 5)

# --- the same bytes moved by a pure data-movement op, by destination ----------------------------
print("\n  pure movement of the same result, by destination:", flush=True)
src = ttnn.from_torch(torch.randn(N, N, 3 * H * D), layout=ttnn.TILE_LAYOUT, device=dev,
                      dtype=ttnn.bfloat16, memory_config=L1)
mv = {}
for tag, mem in (("L1 -> DRAM", DRAM), ("L1 -> L1", L1)):
    ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(src, memory_config=mem)))
    mv[tag] = {"ms": round(ms, 5), "write_GBs": round(RESULT_BYTES / (ms / 1e3) / 1e9, 1)}
    print(f"    clone {tag:11s} {ms:8.5f} ms   {mv[tag]['write_GBs']:7.1f} GB/s counting the write only",
          flush=True)
res["movement"] = mv
predicted = mv["L1 -> DRAM"]["ms"] - mv["L1 -> L1"]["ms"]
res["predicted_penalty_ms"] = round(predicted, 5)
res["predicted_over_measured"] = round(predicted / penalty, 3) if penalty else None

# Two models for what the DRAM-out matmul should cost, given the L1-out matmul and the write:
#   overlapped: the writer streams the result out while the math pipe works  -> max(compute, write)
#   serialized: the result write does not overlap the compute at all         -> compute + write
serial_model = mm["oL1"]["ms"] + mv["L1 -> DRAM"]["ms"]
overlap_model = max(mm["oL1"]["ms"], mv["L1 -> DRAM"]["ms"])
res["serialized_model_ms"] = round(serial_model, 5)
res["overlapped_model_ms"] = round(overlap_model, 5)
res["serialized_model_error_pct"] = round(100 * (serial_model - mm["oDRAM"]["ms"]) / mm["oDRAM"]["ms"], 2)
res["overlapped_model_error_pct"] = round(100 * (overlap_model - mm["oDRAM"]["ms"]) / mm["oDRAM"]["ms"], 2)

print(f"\n  measured matmul penalty  {penalty:.5f} ms", flush=True)
print(f"  the write on its own     {predicted + mv['L1 -> L1']['ms']:.5f} ms "
      f"({100*mv['L1 -> DRAM']['ms']/penalty:.0f}% of the penalty)", flush=True)
print(f"\n  measured  oDRAM matmul                  {mm['oDRAM']['ms']:.5f} ms", flush=True)
print(f"  if the write OVERLAPPED the compute    {overlap_model:.5f} ms  "
      f"({res['overlapped_model_error_pct']:+.1f}%)", flush=True)
print(f"  if the write is SERIALIZED after it    {serial_model:.5f} ms  "
      f"({res['serialized_model_error_pct']:+.1f}%)", flush=True)
ttnn.deallocate(src)

# --- device profiler, if this build has one -----------------------------------------------------
res["device_profiler"] = "not attempted"
if os.environ.get("TT_METAL_DEVICE_PROFILER"):
    try:
        ttnn.ReadDeviceProfiler(dev)
        res["device_profiler"] = "ReadDeviceProfiler returned without error"
    except Exception as e:
        res["device_profiler"] = f"unavailable: {type(e).__name__}: {str(e)[:80]}"
    print(f"  device profiler: {res['device_profiler']}", flush=True)

json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/write_ablation.json", "w"), indent=2)
print("wrote", sys.argv[1] if len(sys.argv) > 1 else "/tmp/write_ablation.json", flush=True)
