#!/usr/bin/env python3
"""Atom-window matmul: main's `batched_matmul` vs D2's chunked `_bmm` vs the naive call.

Four arms on identical operands in one process:
  naive  -- ttnn.matmul(x, y, ckc), the call both helpers fall back to
  main   -- tt_bio.tenstorrent.batched_matmul as merged (E7, 373038e2)
  d2     -- AtomTransformer._bmm from wk/perfwar-atom-window-attention, verbatim
  relax  -- main's chooser with its `batch * m_tiles < cores` early return removed
            (only present at the sizes where main declines)

Arms are rotated per trial so a host load ramp cannot map onto arm order.
"""
import json, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import (get_device, CORE_GRID_MAIN, batched_matmul,
                                _batched_matmul_search, _batched_matmul_block_w)

H, NQ, NK, DH = 4, 32, 128, 32           # AtomTransformer: 4 heads, 32-row window, 128 keys, dh=32
NBS = [75, 29, 19, 8]                    # 298 aa, 117 aa prot.yaml, ubq ~76 aa, a tiny system
REPS, TRIALS, WARM = 6, 7, 3

dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GX, GY = CORE_GRID_MAIN.x, CORE_GRID_MAIN.y
CORES = GX * GY


def search_nogate(batch, m_tiles, k_tiles, n_tiles, elem_bytes, l1):
    """`_batched_matmul_search` verbatim, minus the `batch * m_tiles < cores` early return."""
    if batch < 2:
        return None
    block_w = _batched_matmul_block_w(m_tiles, k_tiles, n_tiles)
    tile, acc_tile = 1024 * elem_bytes, 4096
    legal = []
    for p in range(1, m_tiles + 1):
        if m_tiles % p or (p != m_tiles and batch * m_tiles // p > CORES):
            continue
        if 2 * (p + n_tiles) * block_w * tile + p * n_tiles * (tile + acc_tile) > l1:
            continue
        legal.append(p)
    if not legal:
        return None
    saturating = [p for p in legal if batch * m_tiles // p >= 32]
    per_core_M = max(saturating) if saturating else min(legal)
    sub_w = max(w for w in range(1, min(4, n_tiles) + 1) if n_tiles % w == 0)
    sub_h = max(h for h in range(1, min(4 // sub_w, per_core_M) + 1) if per_core_M % h == 0)
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(GX, GY), in0_block_w=block_w,
        out_subblock_h=sub_h, out_subblock_w=sub_w,
        per_core_M=per_core_M, per_core_N=n_tiles)


def cfg_str(c):
    return None if c is None else (f"ibw={c.in0_block_w} pcM={c.per_core_M} pcN={c.per_core_N} "
                                   f"osh={c.out_subblock_h} osw={c.out_subblock_w}")


def d2_bmm(a, b, in0_block_w=1):
    """`AtomTransformer._bmm` from wk/perfwar-atom-window-attention, verbatim."""
    nb, h = a.shape[0], a.shape[1]
    m_t, k_t, n_t = a.shape[-2] // 32, a.shape[-1] // 32, b.shape[-1] // 32
    if (h > CORES or k_t % in0_block_w or a.shape[-2] % 32 or a.shape[-1] % 32
            or b.shape[-1] % 32 or b.shape[0] != nb or b.shape[1] != h):
        return ttnn.matmul(a, b, compute_kernel_config=CKC)
    per = max(1, CORES // h)
    g = (nb + per - 1) // per
    per = (nb + g - 1) // g
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=(GX, GY),
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


L1 = int(ttnn.get_max_worker_l1_unreserved_size())
rows = []
for dt, dn in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
    eb = 4 if dt == ttnn.float32 else 2
    for nb in NBS:
        g = torch.Generator().manual_seed(0)
        q = tt(torch.randn(nb, H, NQ, DH, generator=g), dt)
        kt = tt(torch.randn(nb, H, DH, NK, generator=g), dt)
        a = tt(torch.rand(nb, H, NQ, NK, generator=g), dt)
        v = tt(torch.randn(nb, H, NK, DH, generator=g), dt)
        for lbl, x, y, kt_n, nt_n, d2_ibw in (("QK^T", q, kt, 1, 4, 1), ("A@V", a, v, 4, 1, 2)):
            batch = nb * H
            real = _batched_matmul_search(batch, 1, kt_n, nt_n, eb, (GX, GY), L1)
            nog = search_nogate(batch, 1, kt_n, nt_n, eb, L1)
            if real is not None:
                assert cfg_str(real) == cfg_str(nog), (cfg_str(real), cfg_str(nog))
            arms = {
                "naive": lambda: ttnn.matmul(x, y, compute_kernel_config=CKC),
                "main": lambda: batched_matmul(x, y, compute_kernel_config=CKC),
                "d2": lambda: d2_bmm(x, y, d2_ibw),
            }
            if real is None and nog is not None:
                arms["relax"] = lambda: ttnn.matmul(x, y, program_config=nog,
                                                    compute_kernel_config=CKC)
            ref = ttnn.to_torch(arms["naive"]())
            exact = {k: bool(torch.equal(ttnn.to_torch(f()), ref)) for k, f in arms.items()}
            names = list(arms)
            samples = {k: [] for k in names}
            for t in range(TRIALS):
                for k in names[t % len(names):] + names[:t % len(names)]:
                    samples[k].append(timed(arms[k]))
            us = {k: round(st.median(v) * 1e6, 2) for k, v in samples.items()}
            rows.append({"dtype": dn, "nb": nb, "batch": batch, "op": lbl,
                         "main_declines": real is None, "main_cfg": cfg_str(real),
                         "relax_cfg": cfg_str(nog), "us": us, "bit_exact": exact,
                         "vs_naive": {k: round(us["naive"] / us[k], 3) for k in names if k != "naive"},
                         "main_vs_d2": round(us["d2"] / us["main"], 3)})
            print(f"{dn} nb={nb:3d} b={batch:4d} {lbl:5s} naive={us['naive']:8.2f} "
                  f"main={us['main']:8.2f} d2={us['d2']:8.2f}"
                  + (f" relax={us['relax']:8.2f}" if "relax" in us else "")
                  + f" | main {us['naive']/us['main']:5.2f}x d2 {us['naive']/us['d2']:5.2f}x"
                  + (f" relax {us['naive']/us['relax']:5.2f}x" if "relax" in us else "")
                  + f" | exact={exact} | cfg={cfg_str(real) or 'DECLINED -> ' + str(cfg_str(nog))}",
                  flush=True)
        for t in (q, kt, a, v):
            ttnn.deallocate(t)

out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/micro_qb1c0.json"
json.dump(rows, open(out, "w"), indent=2)
print("wrote", out, flush=True)
