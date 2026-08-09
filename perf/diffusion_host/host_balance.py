"""Host machine balance and per-phase arithmetic intensity for the EDM sampler (D5).

The device is idle for every sampler phase, so the 338 FLOP/byte device machine balance does not
apply. The phases run single-threaded through ATen, so both sides of the HOST balance are measured
single-threaded: peak f32 FLOP/s from a resident square matmul, and the copy roof from a resident
clone. Arithmetic intensity per phase is counted from the source, not estimated.
"""
import time
import torch

PC = time.perf_counter
torch.set_grad_enabled(False)
torch.set_num_threads(1)


def bench(fn, reps, warm=20):
    for _ in range(warm):
        fn()
    t0 = PC()
    for _ in range(reps):
        fn()
    return (PC() - t0) / reps


# --- host compute roof, single thread, resident -------------------------------------------------
best = 0.0
for n in (256, 512, 1024):
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    dt = bench(lambda: torch.mm(a, b), 50)
    gf = 2 * n ** 3 / dt / 1e9
    print(f"mm {n}x{n}: {dt*1e6:9.1f} us -> {gf:7.2f} GFLOP/s (1 thread)")
    best = max(best, gf)

# --- host copy roof, single thread, resident ----------------------------------------------------
x = torch.randn(1, 240000, 3)
bts = x.numel() * 4
dt = bench(lambda: x.clone(), 500)
bw = 2 * bts / dt / 1e9
print(f"clone {bts/1024:.0f} kB: {dt*1e6:.2f} us -> {bw:.2f} GB/s r+w (1 thread)")
print(f"HOST machine balance = {best*1e9/(bw*1e9):.2f} FLOP/byte  ({best:.2f} GFLOP/s / {bw:.2f} GB/s)")

# --- per-phase arithmetic intensity, counted from the source ------------------------------------
N = 2400
phases = [
    # name, FLOPs, bytes touched (read + write)
    ("compute_random_augmentation", 4 * 2 + 4 + 4 * 2 + 18 * 3 + 3, (4 + 4 + 3) * 4 * 2),
    ("mean-centring", 3 * N + 3 + 3 * N, (3 * N + 3 * N) * 4),
    ("rotate (einsum + tr)", 2 * 3 * 3 * N + 3 * N, (3 * N + 9 + 3 * N + 3) * 4),
    ("RNG (randn + scale)", 3 * N, (3 * N) * 4),
    ("x_noisy = x + eps", 3 * N, (3 * 3 * N) * 4),
    ("EDM update (d, x)", 4 * 3 * N, (5 * 3 * N) * 4),
]
print(f"\nper-phase arithmetic intensity at n_atoms={N} (host machine balance "
      f"{best*1e9/(bw*1e9):.2f} FLOP/byte):")
for name, fl, by in phases:
    print(f"  {name:32s} {fl:8d} FLOP / {by:8d} B = {fl/by:6.3f} FLOP/byte")
