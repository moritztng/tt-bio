#!/usr/bin/env python3
"""z-silu -- the fused-silu A/B probe, at the Transition fc1's own shapes.

One process = one kernel arm, because the arm IS the JIT header and a header edit cannot be
re-picked-up inside a live process. The bare matmul is re-measured in the SAME process every round,
so every arm carries its own bracketing control and the reported silu cost is a within-process
difference. Arms are alternated by the driver, never blocked.

    TT_VISIBLE_DEVICES=2 ... python3 z_probe.py --arm A --shape 298 --out arm.json
"""
from __future__ import annotations
import argparse, json, os, time
import torch, ttnn

SHAPES = {
    "298": (1, 30, 298, 256),   # 9600 padded output tiles at N=1024
    "512": (1, 16, 512, 256),   # 8192 padded output tiles at N=1024
}
N_OUT = 1024


def med(v):
    return sorted(v)[len(v) // 2]


def timed(dev, fn, k, reps, warm=2):
    for _ in range(warm):
        for _ in range(k):
            r = fn()
            ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(k):
            # dealloc is host-side bookkeeping, it does not drain the queue; keeping k outputs
            # alive would be 196 MB of L1 at the 298 shape and the program refuses to enqueue.
            ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / k * 1e6)   # us per call
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--shape", default="298", choices=list(SHAPES))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--roofs", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    shp = SHAPES[a.shape]
    torch.manual_seed(0)
    ta = torch.randn(*shp)
    tb = torch.randn(256, N_OUT)

    dev = ttnn.open_device(device_id=0)
    res = dict(arm=a.arm, shape=list(shp), n_out=N_OUT, k=a.k, reps=a.reps,
               load=[round(v, 2) for v in os.getloadavg()])
    try:
        cfg = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True, packer_l1_acc=True)
        res["cfg"] = dict(math_approx_mode=bool(cfg.math_approx_mode),
                          fp32_dest_acc_en=bool(cfg.fp32_dest_acc_en),
                          packer_l1_acc=bool(cfg.packer_l1_acc),
                          math_fidelity=str(cfg.math_fidelity))
        grid = ttnn.CoreGrid(y=10, x=11)          # CORE_GRID_MAIN, tenstorrent.py:202
        res["core_grid"] = [10, 11]

        A = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        # production weights carry no memory_config (tenstorrent.py:1411) -> DRAM
        B = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)

        fused = lambda: ttnn.linear(A, B, activation="silu", compute_kernel_config=cfg,
                                    memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                    core_grid=grid)
        bare = lambda: ttnn.linear(A, B, compute_kernel_config=cfg,
                                   memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                                   core_grid=grid)

        # interleave the two measurements round by round so drift hits both equally
        f_us, b_us = [], []
        for _ in range(a.reps):
            f_us += timed(dev, fused, a.k, 1, warm=1 if not f_us else 0)
            b_us += timed(dev, bare, a.k, 1, warm=1 if not b_us else 0)
        res["fused_us"] = [round(v, 3) for v in f_us]
        res["bare_us"] = [round(v, 3) for v in b_us]
        res["fused_med_us"] = round(med(f_us), 3)
        res["bare_med_us"] = round(med(b_us), 3)
        res["silu_us"] = round(med(f_us) - med(b_us), 3)

        # padded output tiles: rows padded to 32, cols to 32
        rows = shp[1] * ((shp[2] + 31) // 32 * 32)
        tiles = (rows // 32) * (N_OUT // 32)
        res["padded_output_tiles"] = tiles
        res["silu_us_per_tile"] = round(res["silu_us"] / tiles, 8)
        res["fused_us_per_tile"] = round(res["fused_med_us"] / tiles, 8)

        if a.roofs:
            out_t = fused()
            ttnn.synchronize_device(dev)
            elems = rows * N_OUT
            res["roof_elems"] = elems
            res["roof_bytes_bf16"] = elems * 2
            cl = timed(dev, lambda: ttnn.clone(out_t, memory_config=ttnn.L1_MEMORY_CONFIG), a.k, a.reps)
            res["clone_us"] = [round(v, 3) for v in cl]
            res["clone_med_us"] = round(med(cl), 3)
            res["clone_GBps_one_way"] = round(elems * 2 / (med(cl) * 1e-6) / 1e9, 2)
            si = timed(dev, lambda: ttnn.silu(out_t, memory_config=ttnn.L1_MEMORY_CONFIG), a.k, a.reps)
            res["silu_standalone_us"] = [round(v, 3) for v in si]
            res["silu_standalone_med_us"] = round(med(si), 3)
            res["silu_standalone_Gelem_s_total"] = round(elems / (med(si) * 1e-6) / 1e9, 4)
            res["silu_standalone_Gelem_s_per_core_110"] = round(elems / (med(si) * 1e-6) / 1e9 / 110, 4)
            rl = timed(dev, lambda: ttnn.relu(out_t, memory_config=ttnn.L1_MEMORY_CONFIG), a.k, a.reps)
            res["relu_med_us"] = round(med(rl), 3)
            ttnn.deallocate(out_t)

        # value capture for parity, at the fold's own shape
        res["out_mean_abs"] = float(ttnn.to_torch(fused()).float().abs().mean())
    finally:
        ttnn.close_device(dev)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))


main()
