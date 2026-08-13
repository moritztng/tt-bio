#!/usr/bin/env python3
"""Step 2 of state/protenix-beat-b200.md 6.2: is `mm_generic`'s 0.68.0 transcription still valid
on qb1's 0.67.4 wheel?

`mm_generic.py` transcribes `minimal_matmul_program_factory.cpp` at the v0.68.0 tag and binds it
onto whatever kernel sources the INSTALLED wheel ships. qb1 runs 0.67.4. If the CB indices, the
semaphore order or the runtime-arg layout moved between the two, the descriptor binds silently
wrong and produces numbers, not a crash. Gate: `torch.equal` against the native op at the exact
trimul tail shape.
"""
import sys, time
from pathlib import Path

import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tt_bio.mm_generic as MG                                              # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device               # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
CZ = 256

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

torch.manual_seed(0)
xt = torch.randn(1, N, N, CZ)
wt = torch.randn(CZ, CZ) * 0.05
x = ttnn.from_torch(xt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
w = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

BLK = (4, 8, 1, 4, 1)          # tenstorrent._MM_BLOCK[8]
gx, gy = COMPUTE_GRID_MAIN
print(f"ttnn kernels from {MG._kernel_dir()}")
print(f"shape [1,{N},{N},{CZ}] @ [{CZ},{CZ}]  block={BLK}  grid={gx}x{gy}")

cfg = ttnn.MinimalMatmulConfig(
    M_block_size=BLK[0], K_block_size=BLK[1], N_block_size=BLK[2],
    subblock_h=BLK[3], subblock_w=BLK[4],
    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy))

native = ttnn.experimental.minimal_matmul(
    input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16, config=cfg)
ref = ttnn.to_torch(native)

out = ttnn.allocate_tensor_on_device(
    ttnn.Shape([int(d) for d in native.shape]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
    ttnn.DRAM_MEMORY_CONFIG)
MG.generic_minimal_matmul(dev, x, w, [out], (BLK, (gx, gy)), MG.ckc_args(ckc))
got = ttnn.to_torch(out)

eq = bool(torch.equal(got, ref))
md = float((got.float() - ref.float()).abs().max())
print(f"PARITY generic_minimal_matmul vs native: torch.equal={eq} max_abs_diff={md:.3e}")


def timed(fn, reps=5):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def run_native():
    o = ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
        config=cfg)
    ttnn.deallocate(o)


def run_generic():
    MG.generic_minimal_matmul(dev, x, w, [out], (BLK, (gx, gy)), MG.ckc_args(ckc))


print(f"native  {timed(run_native):.4f} ms")
print(f"generic {timed(run_generic):.4f} ms")
print("VERDICT:", "TRANSCRIPTION VALID ON THIS WHEEL" if eq else "TRANSCRIPTION INVALID")
sys.exit(0 if eq else 1)
