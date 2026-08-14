#!/usr/bin/env python3
"""S1 -- THE KILL GATE for the tile-native FFT design.

Measures the sustained fp32 eltwise compute rate on L1-RESIDENT data, with dispatch removed by
trace capture. The whole design rests on 8 TFLOP/s of eltwise throughput, which is DERIVED, not
measured. Box 512 needs 23.6 MFLOP of eltwise per image inside a 9.97 us DRAM budget = 2.37 TFLOP/s.

  PASS      >= 2.4 TFLOP/s  -> path C proceeds as specified
  MARGINAL  1.2 - 2.4       -> radix-4 cross-tile stages only; box 512 fp32 becomes a stretch goal
  FAIL      <  1.2          -> exact-eltwise cross-tile stage cannot service the floor; fall back
                              to path A (accuracy 1.5e-3) and gate the verdict entirely on E1

L1 residency is proved, not assumed: the implied bandwidth must come out far above the 420.7 GB/s
DRAM roof. If it lands near or below the roof, the shard spilled and the number is void.
Run: TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:ttnn-fft-kernel-spike python3 fftprobe/screen_s1.py
"""
import json, time
import torch, ttnn

CHAIN = 64          # eltwise ops per trace; long enough that replay overhead is < 1%
TILES_H = 2         # tiles of height per core -> 3 tensors x 256 KB = 768 KB of 1.5 MB L1
TILES_W = 32
ROOF_GB_S = 420.7   # measured on this card, probe_p1.json

dev = ttnn.open_device(device_id=0, trace_region_size=64 << 20)
grid = dev.compute_with_storage_grid_size()
ncores = grid.x * grid.y
H, W = TILES_H * 32 * ncores, TILES_W * 32
nel = H * W
print(f"grid {grid.x}x{grid.y} = {ncores} cores | tensor {H}x{W} = {nel/1e6:.2f}M el | "
      f"{nel*4/ncores/1024:.0f} KB per core per tensor", flush=True)

shard = ttnn.create_sharded_memory_config(
    shape=(H, W), core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
    strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR)
def mk():
    return ttnn.from_torch(torch.randn(H, W), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=shard)
a, b, c = mk(), mk(), mk()
ttnn.synchronize_device(dev)

R = {"cores": ncores, "elements": nel, "chain": CHAIN, "roof_GB_s": ROOF_GB_S}

# eager, for the contrast: this is what dispatch overhead alone costs
ttnn.mul(a, b, output_tensor=c); ttnn.synchronize_device(dev)
t0 = time.perf_counter()
for _ in range(CHAIN):
    ttnn.mul(a, b, output_tensor=c)
ttnn.synchronize_device(dev)
eager = time.perf_counter() - t0
R["eager"] = {"ms": eager * 1e3, "TFLOP_s": CHAIN * nel / eager / 1e12,
              "us_per_op": eager / CHAIN * 1e6}
print("eager  ", json.dumps(R["eager"]), flush=True)

# traced: dispatch removed, so this is the compute rate
tid = ttnn.begin_trace_capture(dev, cq_id=0)
for _ in range(CHAIN):
    ttnn.mul(a, b, output_tensor=c)
ttnn.end_trace_capture(dev, tid, cq_id=0)
ttnn.execute_trace(dev, tid, cq_id=0, blocking=False); ttnn.synchronize_device(dev)
best = 1e9
for _ in range(10):
    t0 = time.perf_counter()
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    best = min(best, time.perf_counter() - t0)
tflops = CHAIN * nel / best / 1e12
implied_gb_s = CHAIN * 3 * nel * 4 / best / 1e9
R["traced"] = {"ms": best * 1e3, "TFLOP_s": tflops, "us_per_op": best / CHAIN * 1e6,
               "implied_GB_s": implied_gb_s,
               "l1_resident": bool(implied_gb_s > 4 * ROOF_GB_S)}
print("traced ", json.dumps(R["traced"]), flush=True)

if not R["traced"]["l1_resident"]:
    R["verdict"] = "VOID -- implied bandwidth is not far above the DRAM roof, the shard spilled"
elif tflops >= 2.4:
    R["verdict"] = "PASS -- path C proceeds as specified"
elif tflops >= 1.2:
    R["verdict"] = "MARGINAL -- radix-4 cross-tile only, box 512 fp32 is a stretch goal"
else:
    R["verdict"] = "FAIL -- fall back to path A, gate the verdict on E1"
print("S1 VERDICT:", R["verdict"], f"({tflops:.2f} TFLOP/s)", flush=True)
json.dump(R, open("/home/ttuser/.coworker/wt/ttnn-fft-kernel-spike/fftprobe/screen_s1.json", "w"), indent=1)
ttnn.release_trace(dev, tid)
ttnn.close_device(dev)
