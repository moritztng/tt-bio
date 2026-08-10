#!/usr/bin/env python3
"""Atom-window matmul on the PRODUCTION grid.

`_configure_active_compute_grid` swaps COMPUTE_GRID_MAIN from 11x10 to 13x10 at device open on a
p150a, so anything that imported the symbol by value before `get_device()` is holding 110 cores
while `batched_matmul` uses 130. Read the module attribute after the open.

Arms: naive; `batched_matmul` (main); the config main's chooser builds, passed explicitly on the
130-core grid; the same on 110 cores; and, at the sizes where main's `batch < cores` gate declines,
the config it would have built.

Predictions, written before the run:
  P4  main == explicit-130 within noise wherever batch >= 130. A systematic gap is per-call host
      work in `batched_matmul` (it calls `ttnn.get_max_worker_l1_unreserved_size()` on every call).
  P5  explicit-110 beats explicit-130 whenever ceil(batch/110) == ceil(batch/130): same number of
      block rounds, 18% fewer cores to write runtime args to.
  P6  at batch=116 the 130-core grid needs 1 round and the 110-core grid needs 2, so relaxing the
      gate on the PRODUCTION grid should beat the 3.2x/5.5x the 110-core config gave.
"""
import json, statistics as st, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

H, NQ, NK, DH = 4, 32, 128, 32
NBS = [19, 29, 32, 33, 40, 75, 110]
REPS, TRIALS, WARM = 6, 5, 3

dev = T.get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GRID = tuple(T.COMPUTE_GRID_MAIN)          # after the open: the grid batched_matmul actually uses
CORES = GRID[0] * GRID[1]
L1 = int(ttnn.get_max_worker_l1_unreserved_size())
print(f"production grid {GRID} = {CORES} cores, L1/core {L1} B", flush=True)


def cfg_on(grid, k_tiles, n_tiles):
    bw = T._batched_matmul_block_w(1, k_tiles, n_tiles)
    sub_w = max(w for w in range(1, min(4, n_tiles) + 1) if n_tiles % w == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=bw,
        out_subblock_h=1, out_subblock_w=sub_w, per_core_M=1, per_core_N=n_tiles)


def timed(fn):
    for _ in range(WARM):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / REPS


def tt(x, dt):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


# Price batched_matmul's per-call host work on its own: it queries the device for L1 every call.
t0 = time.perf_counter()
for _ in range(2000):
    T._batched_matmul_config(300, 1, 1, 4, 4)
host_us = (time.perf_counter() - t0) / 2000 * 1e6
print(f"_batched_matmul_config host cost {host_us:.2f} us/call", flush=True)

rows = []
for dt, dn in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
    eb = 4 if dt == ttnn.float32 else 2
    for nb in NBS:
        g = torch.Generator().manual_seed(0)
        q = tt(torch.randn(nb, H, NQ, DH, generator=g), dt)
        kt = tt(torch.randn(nb, H, DH, NK, generator=g), dt)
        a = tt(torch.rand(nb, H, NQ, NK, generator=g), dt)
        v = tt(torch.randn(nb, H, NK, DH, generator=g), dt)
        for lbl, x, y, kt_n, nt_n in (("QK^T", q, kt, 1, 4), ("A@V", a, v, 4, 1)):
            batch = nb * H
            applies = T._batched_matmul_search(batch, 1, kt_n, nt_n, eb, GRID, L1) is not None
            c130, c110 = cfg_on(GRID, kt_n, nt_n), cfg_on((11, 10), kt_n, nt_n)
            arms = {
                "naive": lambda: ttnn.matmul(x, y, compute_kernel_config=CKC),
                "main": lambda: T.batched_matmul(x, y, compute_kernel_config=CKC),
                "exp130": lambda: ttnn.matmul(x, y, program_config=c130, compute_kernel_config=CKC),
                "exp110": lambda: ttnn.matmul(x, y, program_config=c110, compute_kernel_config=CKC),
            }
            ref = ttnn.to_torch(arms["naive"]())
            exact = {k: bool(torch.equal(ttnn.to_torch(f()), ref)) for k, f in arms.items()}
            names = list(arms)
            samples = {k: [] for k in names}
            for t in range(TRIALS):
                for k in names[t % len(names):] + names[:t % len(names)]:
                    samples[k].append(timed(arms[k]))
            us = {k: round(st.median(v) * 1e6, 2) for k, v in samples.items()}
            rows.append({"dtype": dn, "nb": nb, "batch": batch, "op": lbl,
                         "rounds_130": -(-batch // CORES), "rounds_110": -(-batch // 110),
                         "main_applies": applies, "us": us, "bit_exact": exact,
                         "vs_naive": {k: round(us["naive"] / us[k], 3) for k in names if k != "naive"}})
            print(f"{dn} nb={nb:3d} b={batch:4d} {lbl:5s} applies={str(applies):5s} "
                  f"naive={us['naive']:8.2f} main={us['main']:8.2f} exp130={us['exp130']:8.2f} "
                  f"exp110={us['exp110']:8.2f} | main {us['naive']/us['main']:5.2f}x "
                  f"e130 {us['naive']/us['exp130']:5.2f}x e110 {us['naive']/us['exp110']:5.2f}x "
                  f"| rounds {-(-batch//CORES)}/{-(-batch//110)} | exact={exact}", flush=True)
        for t in (q, kt, a, v):
            ttnn.deallocate(t)

out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/probe2_qb1c0.json"
json.dump({"grid": list(GRID), "l1": L1, "config_host_us": round(host_us, 2), "rows": rows},
          open(out, "w"), indent=2)
print("wrote", out, flush=True)
