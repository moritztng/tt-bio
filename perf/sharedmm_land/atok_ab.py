#!/usr/bin/env python3
"""The one site the merged batchedmm-reconcile does NOT cover: openfold3_diffusion_module.py:339.

`ai = atom_to_token_mean @ relu(linear_q(ql))`, rank 3, batch 1, fp32, Mt/Kt/Nt = 10/75/24 at
298 aa. Its exact twin `protenix.py:870` has carried `core_grid=CORE_GRID_MAIN` all along; the
OF3 copy is the only ttnn.matmul in its own file without one.

batched_matmul cannot reach it: rank 3 < 4, and batch 1 < 2. So this is a plain core_grid arm,
not a program-config arm. Arms interleave inside every rep, device synced on both sides of each
timed region, minimum over reps.
"""
import argparse, json, time
import torch, ttnn
import tt_bio.tenstorrent as T

DRAM = ttnn.DRAM_MEMORY_CONFIG


def bench(fn, iters, reps):
    for _ in range(3):
        ttnn.deallocate(fn())
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(DEV)
        out.append((time.perf_counter() - t0) / iters * 1e6)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="perf/sharedmm_land/atok_c2.json")
    a = ap.parse_args()

    DEV = T.get_device()
    T._configure_active_compute_grid(DEV)
    grid = T.CORE_GRID_MAIN
    ckc = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    # 298 aa OpenFold3: n_token padded to 320 (Mt=10), n_atom padded to 2400 (Kt=75), 768 (Nt=24).
    M, K, N = 320, 2400, 768
    torch.manual_seed(0)
    ta, tb = torch.randn(1, M, K), torch.randn(1, K, N)
    A = ttnn.from_torch(ta, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=DRAM)
    B = ttnn.from_torch(tb, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=DEV,
                        memory_config=DRAM)

    shipped = lambda: ttnn.matmul(A, B, compute_kernel_config=ckc)
    tuned = lambda: ttnn.matmul(A, B, compute_kernel_config=ckc, core_grid=grid)

    r_ship, r_tune = bench(shipped, a.iters, a.reps), bench(tuned, a.iters, a.reps)
    # second interleaved pass so host drift cannot land on one arm
    r_tune += bench(tuned, a.iters, a.reps)
    r_ship += bench(shipped, a.iters, a.reps)

    o_s, o_t = ttnn.to_torch(shipped()), ttnn.to_torch(tuned())
    dev = (o_s - o_t).abs()
    ref = ta.to(torch.float64) @ tb.to(torch.float64)
    res = {
        "host": "qb1", "card": 2, "shape": [M, K, N], "dtype": "float32",
        "grid": [grid.x, grid.y], "iters": a.iters, "reps": a.reps,
        "shipped_us": r_ship, "tuned_us": r_tune,
        "shipped_min_us": min(r_ship), "tuned_min_us": min(r_tune),
        "speedup": min(r_ship) / min(r_tune),
        "torch_equal": bool(torch.equal(o_s, o_t)),
        "max_abs_dev": float(dev.max()), "rel_l2_dev": float(dev.norm() / o_s.norm()),
        "out_absmax": float(o_s.abs().max()),
        # which arm is closer to the fp64 reference
        "err_shipped_rel_l2": float((o_s.to(torch.float64) - ref).norm() / ref.norm()),
        "err_tuned_rel_l2": float((o_t.to(torch.float64) - ref).norm() / ref.norm()),
    }
    print(json.dumps(res, indent=2))
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
