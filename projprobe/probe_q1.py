#!/usr/bin/env python3
"""Phase-1 probes for the Fourier-slice projection / backprojection spike.

1. DRAM bandwidth roof on THIS card (card 1), 3-tensor eltwise-add traffic model.
   Nothing is inherited: the FFT spike measured cards 2 and 3.
2. ttnn.grid_sample -- the only shipped Tensix op that does interpolation at arbitrary
   coordinates. This is the honest stock baseline for a projection kernel, far more so than
   ttnn.gather, and it is also prior art: tt-metal #27904 (perf work, open) and #28513
   (generality, open) say what its authors already know is wrong with it.
   Measured as ns per output point, swept over output count and channel count, because
   channel count is exactly the axis a Fourier-slice extraction does NOT have (complex = 2).
"""
import json, time
import torch, ttnn

R = {}
dev = ttnn.open_device(device_id=0)
def sync(): ttnn.synchronize_device(dev)

def bw_roof(dtype, nb, n=8192):
    a = ttnn.from_torch(torch.randn(n, n), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(n, n), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.add(a, b); sync(); ttnn.deallocate(o)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); o = ttnn.add(a, b); sync(); dt = time.perf_counter() - t0
        ttnn.deallocate(o); best = min(best, dt)
    ttnn.deallocate(a); ttnn.deallocate(b)
    return {"ms": best * 1e3, "GB_s": 3 * n * n * nb / best / 1e9}

R["bw_fp32"] = bw_roof(ttnn.float32, 4)
R["bw_bf16"] = bw_roof(ttnn.bfloat16, 2)
print("BW fp32", R["bw_fp32"], flush=True)
print("BW bf16", R["bw_bf16"], flush=True)

# ---------------- grid_sample ----------------
def gs(Hin, Win, C, Hout, Wout, reps=5):
    x = ttnn.from_torch(torch.randn(1, Hin, Win, C), dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    g = ttnn.from_torch(torch.rand(1, Hout, Wout, 2) * 1.8 - 0.9, dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    o = ttnn.grid_sample(x, g); sync(); ttnn.deallocate(o)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); o = ttnn.grid_sample(x, g); sync()
        best = min(best, time.perf_counter() - t0); ttnn.deallocate(o)
    ttnn.deallocate(x); ttnn.deallocate(g)
    npts = Hout * Wout
    return {"ms": best * 1e3, "points": npts, "C": C,
            "ns_per_point": best * 1e9 / npts,
            "ns_per_point_per_chan": best * 1e9 / npts / C}

for (Hin, Win, C, Ho, Wo) in [
        (512, 512, 32, 256, 256),   # box 256, padded plane, C=32 (the op's minimum tile-aligned C)
        (512, 512, 32, 512, 512),   # 4x the points, same image -> marginal rate
        (512, 512, 64, 256, 256),   # does cost scale with C?
        (1024, 1024, 32, 512, 512), # box 512 shape
]:
    k = f"gs_{Hin}x{Win}_C{C}_{Ho}x{Wo}"
    try:
        R[k] = gs(Hin, Win, C, Ho, Wo)
    except Exception as e:
        R[k] = {"error": f"{type(e).__name__}: {e}"}
    print(k, R[k], flush=True)

ttnn.close_device(dev)
print(json.dumps(R, indent=1))
open("/tmp/probe_q1.json", "w").write(json.dumps(R, indent=1))
