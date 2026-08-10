#!/usr/bin/env python3
"""Why does main's config buy nothing at batch=116 while it buys 6-12x at batch=300?

Arms: naive; `batched_matmul` (main); the same config passed explicitly (to separate "the helper
did not apply it" from "the config is slow here"); and the config chunked so no core ever holds
more than one block.
"""
import json, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import (get_device, CORE_GRID_MAIN, batched_matmul,
                                _batched_matmul_search, _batched_matmul_block_w)

H, NQ, NK, DH = 4, 32, 128, 32
NBS = [8, 19, 26, 27, 28, 29, 30, 33, 40, 55, 75, 110]
REPS, TRIALS, WARM = 6, 5, 3

dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GX, GY = CORE_GRID_MAIN.x, CORE_GRID_MAIN.y
CORES = GX * GY
L1 = int(ttnn.get_max_worker_l1_unreserved_size())


def cfg_of(batch, k_tiles, n_tiles, elem_bytes):
    """The config main's search would build if its `batch * m_tiles < cores` gate never fired."""
    bw = _batched_matmul_block_w(1, k_tiles, n_tiles)
    sub_w = max(w for w in range(1, min(4, n_tiles) + 1) if n_tiles % w == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(GX, GY), in0_block_w=bw,
        out_subblock_h=1, out_subblock_w=sub_w, per_core_M=1, per_core_N=n_tiles)


def chunked(x, y, pc, nb, h):
    """Same config, but split so each launch issues at most CORES blocks."""
    per = max(1, CORES // h)
    g = -(-nb // per)
    per = -(-nb // g)
    if g == 1:
        return ttnn.matmul(x, y, program_config=pc, compute_kernel_config=CKC)
    outs = []
    for c in range(0, nb, per):
        e = min(c + per, nb)
        outs.append(ttnn.matmul(
            ttnn.slice(x, [c, 0, 0, 0], [e, h, x.shape[2], x.shape[3]]),
            ttnn.slice(y, [c, 0, 0, 0], [e, h, y.shape[2], y.shape[3]]),
            program_config=pc, compute_kernel_config=CKC))
    return ttnn.concat(outs, dim=0)


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


rows = []
dt, eb = ttnn.float32, 4
for nb in NBS:
    g = torch.Generator().manual_seed(0)
    q = tt(torch.randn(nb, H, NQ, DH, generator=g), dt)
    kt = tt(torch.randn(nb, H, DH, NK, generator=g), dt)
    a = tt(torch.rand(nb, H, NQ, NK, generator=g), dt)
    v = tt(torch.randn(nb, H, NK, DH, generator=g), dt)
    for lbl, x, y, kt_n, nt_n in (("QK^T", q, kt, 1, 4), ("A@V", a, v, 4, 1)):
        batch = nb * H
        pc = cfg_of(batch, kt_n, nt_n, eb)
        applies = _batched_matmul_search(batch, 1, kt_n, nt_n, eb, (GX, GY), L1) is not None
        arms = {
            "naive": lambda: ttnn.matmul(x, y, compute_kernel_config=CKC),
            "main": lambda: batched_matmul(x, y, compute_kernel_config=CKC),
            "explicit": lambda: ttnn.matmul(x, y, program_config=pc, compute_kernel_config=CKC),
            "chunked": lambda: chunked(x, y, pc, nb, H),
        }
        ref = ttnn.to_torch(arms["naive"]())
        exact = {k: bool(torch.equal(ttnn.to_torch(f()), ref)) for k, f in arms.items()}
        names = list(arms)
        samples = {k: [] for k in names}
        for t in range(TRIALS):
            for k in names[t % len(names):] + names[:t % len(names)]:
                samples[k].append(timed(arms[k]))
        us = {k: round(st.median(v) * 1e6, 2) for k, v in samples.items()}
        rows.append({"nb": nb, "batch": batch, "op": lbl, "main_applies": applies,
                     "us": us, "bit_exact": exact,
                     "vs_naive": {k: round(us["naive"] / us[k], 3) for k in names if k != "naive"}})
        print(f"nb={nb:3d} b={batch:4d} {lbl:5s} applies={str(applies):5s} "
              f"naive={us['naive']:8.2f} main={us['main']:8.2f} explicit={us['explicit']:8.2f} "
              f"chunked={us['chunked']:8.2f} | exp {us['naive']/us['explicit']:5.2f}x "
              f"chunk {us['naive']/us['chunked']:5.2f}x | exact={exact}", flush=True)
    for t in (q, kt, a, v):
        ttnn.deallocate(t)

out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/probe_qb1c0.json"
json.dump(rows, open(out, "w"), indent=2)
print("wrote", out, flush=True)
