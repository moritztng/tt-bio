"""Measured Wormhole roofs on one chip, same instrument as bench.py (back-to-back slope, warm, 10 reps).

  matmul  [8192, 4096] x [4096, 4096], operands bf16 / bfp8 / bfp4, HiFi4 / HiFi3 / HiFi2 / LoFi, fp32 acc off
  dram    ttnn.add of two [1, 1, 8192, 8192] tensors (3 x the tensor's bytes moved), bf16 and bfp8
usage: roof.py OUT CHIP
"""
import json, os, statistics, subprocess, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]).resolve(); CHIP = int(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
W._apply_tt_environment(worker_payload(runtime.build_local_workers("tenstorrent", [object()], [CHIP])[0]))
import torch, ttnn
import tt_bio.tenstorrent as T
dev = T.get_device()
node = sorted(int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
              if os.path.exists(f"/proc/self/fd/{fd}") and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/"))[0]
LOG = open(OUT / "roof.jsonl", "a")
print(json.dumps({"ev": "nodes_open", "nodes": [node]}), flush=True)   # run_on_chip.sh restarts the agent on this line
def log(**kw):
    s = json.dumps(kw, default=str); LOG.write(s + "\n"); LOG.flush(); print(s, flush=True)
def clk():
    return int(open(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read().split()[0])

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
FID = {"HiFi4": ttnn.MathFidelity.HiFi4, "HiFi3": ttnn.MathFidelity.HiFi3, "HiFi2": ttnn.MathFidelity.HiFi2,
       "LoFi": ttnn.MathFidelity.LoFi}

def slope(fn, reps=10):
    for _ in range(2): ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = time.perf_counter(); ttnn.deallocate(fn()); ttnn.synchronize_device(dev); per = time.perf_counter() - t
    n = max(2, int(0.1 / per) + 1); v = []; c = []
    for _ in range(reps):
        ttnn.synchronize_device(dev); t = time.perf_counter()
        for _ in range(n):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev); v.append((time.perf_counter() - t) / n); c.append(clk())
    return statistics.median(v), min(v), max(v), c

M, K, N = 8192, 4096, 4096
a32 = torch.randn(M, K); b32 = torch.randn(K, N)
for dn, dt in DT.items():
    a = ttnn.from_torch(a32, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(b32, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    for fn_, fid in FID.items():
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=False,
                                              packer_l1_acc=True)
        try:
            med, lo, hi, c = slope(lambda: ttnn.matmul(a, b, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                                       core_grid=T.CORE_GRID_MAIN))
            log(kind="matmul", dtype=dn, fid=fn_, ms=med * 1e3, ms_min=lo * 1e3, ms_max=hi * 1e3,
                tflops=2 * M * K * N / med / 1e12, aiclk=[min(c), max(c)])
        except Exception as e:
            log(kind="matmul", dtype=dn, fid=fn_, error=str(e)[:200])
    ttnn.deallocate(a); ttnn.deallocate(b)
x32 = torch.randn(1, 1, 8192, 8192)
for dn in ("bf16", "bfp8"):
    x = ttnn.from_torch(x32, dtype=DT[dn], layout=ttnn.TILE_LAYOUT, device=dev)
    y = ttnn.from_torch(x32, dtype=DT[dn], layout=ttnn.TILE_LAYOUT, device=dev)
    tb = {"bf16": 2048, "bfp8": 1088}[dn] * (8192 // 32) ** 2
    med, lo, hi, c = slope(lambda: ttnn.add(x, y, dtype=DT[dn]))
    log(kind="dram_add", dtype=dn, ms=med * 1e3, ms_min=lo * 1e3, ms_max=hi * 1e3, bytes=3 * tb,
        gbps=3 * tb / med / 1e9, aiclk=[min(c), max(c)])
log(kind="end")
os._exit(0)
