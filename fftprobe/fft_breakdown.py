#!/usr/bin/env python3
"""Where does arm A's time actually go? A 271x speedup is only quotable if arm A is competent.

Arm A runs at 609 GFLOP/s, 0.19% of the matmul roof S2b measured, which is far too slow to be the
matmuls themselves. This times each piece separately at the same shapes so the A/B either stands or
gets rebuilt.
"""
import os, time
import numpy as np, torch, ttnn

N = 256
SL = int(os.environ.get("BD_SL", "130"))
tor, tdt = torch.bfloat16, ttnn.bfloat16


def t2(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).to(tor)
    t = t.reshape(1, 1, *t.shape[-2:]) if t.ndim == 2 else t.reshape(1, 1, -1, t.shape[-1])
    return ttnn.from_torch(t, dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev)


def timed(fn, reps=5):
    fn(); ttnn.synchronize_device(dev)
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); r = fn(); ttnn.synchronize_device(dev)
        b = min(b, time.perf_counter() - t0)
    return b, r


dev = ttnn.open_device(device_id=0)
try:
    rng = np.random.default_rng(0)
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N)
    x = rng.standard_normal((SL, N, N))
    X, Fr = t2(x), t2(F.real)
    flops_mm = SL * 2 * N ** 3

    s, _ = timed(lambda: ttnn.matmul(X, Fr))
    print(f"1 matmul  [1,1,{SL*N},{N}] @ [1,1,{N},{N}]   {s*1e3:8.3f} ms   "
          f"{flops_mm/s/1e12:7.2f} TFLOP/s", flush=True)

    r4 = ttnn.reshape(X, (1, SL, N, N))
    s, _ = timed(lambda: ttnn.reshape(X, (1, SL, N, N)))
    print(f"reshape   [1,1,{SL*N},{N}] -> [1,{SL},{N},{N}]   {s*1e3:8.3f} ms", flush=True)

    s, _ = timed(lambda: ttnn.transpose(r4, -2, -1))
    print(f"transpose [1,{SL},{N},{N}] dims -2,-1          {s*1e3:8.3f} ms", flush=True)

    s, _ = timed(lambda: ttnn.add(X, X))
    print(f"add       [1,1,{SL*N},{N}]                    {s*1e3:8.3f} ms", flush=True)

    # batched-small matmul, the shape the first arm A used, for contrast
    Xb = ttnn.from_torch(torch.from_numpy(x).to(tor).reshape(1, SL, N, N), dtype=tdt,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    Fb = ttnn.from_torch(torch.from_numpy(F.real).to(tor).unsqueeze(0).expand(SL, -1, -1)
                         .contiguous().reshape(1, SL, N, N), dtype=tdt,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    s, _ = timed(lambda: ttnn.matmul(Xb, Fb))
    print(f"batched matmul [1,{SL},{N},{N}] @ same        {s*1e3:8.3f} ms   "
          f"{flops_mm/s/1e12:7.2f} TFLOP/s", flush=True)
finally:
    ttnn.close_device(dev)
