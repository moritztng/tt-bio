#!/usr/bin/env python3
"""D2's `_bmm` verbatim on the PRODUCTION grid, against main's `batched_matmul`.

The earlier micro ran D2's replica with an 11x10 grid because it imported CORE_GRID_MAIN before
`get_device()` swapped it to 13x10. D2's real code reads the module attribute at call time, so in
production it chunks against 130 cores, not 110. This closes that gap.
"""
import json, statistics as st, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

H, NQ, NK, DH = 4, 32, 128, 32
REPS, TRIALS, WARM = 6, 7, 3

dev = T.get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
print(f"grid {tuple(T.COMPUTE_GRID_MAIN)}", flush=True)


def d2_bmm(a, b, in0_block_w=1):
    """`AtomTransformer._bmm` from wk/perfwar-atom-window-attention, verbatim, including its
    call-time read of CORE_GRID_MAIN."""
    ncores = T.CORE_GRID_MAIN.x * T.CORE_GRID_MAIN.y
    nb, h = a.shape[0], a.shape[1]
    m_t, k_t, n_t = a.shape[-2] // 32, a.shape[-1] // 32, b.shape[-1] // 32
    if (h > ncores or k_t % in0_block_w or a.shape[-2] % 32 or a.shape[-1] % 32
            or b.shape[-1] % 32 or b.shape[0] != nb or b.shape[1] != h):
        return ttnn.matmul(a, b, compute_kernel_config=CKC)
    per = max(1, ncores // h)
    g = (nb + per - 1) // per
    per = (nb + g - 1) // g
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y),
        in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
        per_core_M=m_t, per_core_N=n_t)
    if g == 1:
        return ttnn.matmul(a, b, program_config=pc, compute_kernel_config=CKC)
    outs = []
    for c in range(0, nb, per):
        e = min(c + per, nb)
        outs.append(ttnn.matmul(
            ttnn.slice(a, [c, 0, 0, 0], [e, h, a.shape[2], a.shape[3]]),
            ttnn.slice(b, [c, 0, 0, 0], [e, h, b.shape[2], b.shape[3]]),
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
for dt, dn in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
    for nb in (29, 75):
        g = torch.Generator().manual_seed(0)
        q = tt(torch.randn(nb, H, NQ, DH, generator=g), dt)
        kt = tt(torch.randn(nb, H, DH, NK, generator=g), dt)
        a = tt(torch.rand(nb, H, NQ, NK, generator=g), dt)
        v = tt(torch.randn(nb, H, NK, DH, generator=g), dt)
        for lbl, x, y, d2_ibw in (("QK^T", q, kt, 1), ("A@V", a, v, 2)):
            arms = {
                "naive": lambda: ttnn.matmul(x, y, compute_kernel_config=CKC),
                "main": lambda: T.batched_matmul(x, y, compute_kernel_config=CKC),
                "d2": lambda: d2_bmm(x, y, d2_ibw),
            }
            ref = ttnn.to_torch(arms["naive"]())
            exact = {k: bool(torch.equal(ttnn.to_torch(f()), ref)) for k, f in arms.items()}
            names = list(arms)
            samples = {k: [] for k in names}
            for t in range(TRIALS):
                for k in names[t % len(names):] + names[:t % len(names)]:
                    samples[k].append(timed(arms[k]))
            us = {k: round(st.median(v) * 1e6, 2) for k, v in samples.items()}
            rows.append({"dtype": dn, "nb": nb, "op": lbl, "us": us, "bit_exact": exact,
                         "main_vs_naive": round(us["naive"] / us["main"], 3),
                         "d2_vs_naive": round(us["naive"] / us["d2"], 3),
                         "main_vs_d2": round(us["d2"] / us["main"], 3)})
            print(f"{dn} nb={nb:3d} {lbl:5s} naive={us['naive']:8.2f} main={us['main']:8.2f} "
                  f"d2={us['d2']:8.2f} | main {us['naive']/us['main']:5.2f}x d2 "
                  f"{us['naive']/us['d2']:5.2f}x | main/d2 {us['d2']/us['main']:5.2f}x | "
                  f"exact={exact}", flush=True)
        for t in (q, kt, a, v):
            ttnn.deallocate(t)

out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/probe4_qb1c0.json"
json.dump(rows, open(out, "w"), indent=2)
print("wrote", out, flush=True)
