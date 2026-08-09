"""Where does the per-batch-element block config overtake ttnn's 2D multicast?

The six shipped OF3 shapes split cleanly except for the DiT q@kT (B=16, square 10x10 output),
where the block config is a 0.75x regression while the trunk q@kT at the SAME tile shape and
B=1192 is a 1.8x win. Both take ttnn's 2D-multicast branch, so the only variable is B. This
sweeps B on that one tile shape to locate the crossover instead of guessing a threshold.
"""
import argparse, json, time

import torch
import ttnn

import tt_bio.tenstorrent as T


def mk(dev, shape, seed):
    g = torch.Generator().manual_seed(seed)
    t = (torch.rand(shape, generator=g, dtype=torch.float32) - 0.5) * 0.2
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def time_arm(dev, fn, iters):
    o = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    ttnn.synchronize_device(dev)
    dt = (time.perf_counter() - t0) / iters
    for o in outs:
        ttnn.deallocate(o)
    return dt * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    ttnn.SetDefaultDevice(dev)
    T._configure_active_compute_grid(dev)
    cfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    rows = []
    for B in (16, 32, 64, 128, 256, 512, 1192):
        # DiT q@kT tile shape: Mt=10, Kt=2, Nt=10
        ta, tb = mk(dev, [B, 320, 64], 1), mk(dev, [B, 64, 320], 2)
        ms_auto = time_arm(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg), a.iters)
        pc = T._batched_matmul_program_config(B, 10, 2, 10, T.COMPUTE_GRID_MAIN)
        ms_blk = time_arm(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg,
                                                   program_config=pc), a.iters)
        # per_core_M = Mt (one block per batch element, never split)
        pcM = T._batched_matmul_program_config(10 ** 9, 10, 2, 10, T.COMPUTE_GRID_MAIN)
        ms_mt = time_arm(dev, lambda: ttnn.matmul(ta, tb, compute_kernel_config=cfg,
                                                  program_config=pcM), a.iters)
        row = dict(B=B, per_core_M=pc.per_core_M, ms_auto=ms_auto, ms_block=ms_blk,
                   ms_block_perCoreM_Mt=ms_mt, speedup=ms_auto / ms_blk,
                   speedup_perCoreM_Mt=ms_auto / ms_mt)
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        rows.append(row)
        print(json.dumps(row), flush=True)

    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
