#!/usr/bin/env python3
"""Atom-window attention: baseline vs batch-parallel variants, qb2 card 1.

V0  today's chain: permute K, matmul QK^T, *scale, +z, +pad_bias, softmax, matmul A@V
V1  identical arithmetic, but both matmuls run as ceil(B/ncores) chunks with an explicit
    MatmulMultiCoreReuseProgramConfig so the batch lands on the core grid instead of a serial
    loop. The chunks stay separate through the elementwise chain; one concat at the end.
V2  V1 without the separate pad_bias add (z and pad_bias are both fold-constants, so they can be
    summed once per fold) -- measured to price that op, NOT bit-exact by construction.
"""
import argparse, json, math, statistics as st, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

NB, H, NQ, NK, DH = 75, 4, 32, 128, 32
dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
NCORES = CORE_GRID_MAIN.x * CORE_GRID_MAIN.y
GRID = (CORE_GRID_MAIN.x, CORE_GRID_MAIN.y)


def timed(fn, warm=3, reps=6, trials=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(trials):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / reps)
    return st.median(o)


def tt(x, dt):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


def bmm(a, b, per_core_n, in0_block_w):
    """Batched matmul with the batch spread over the core grid."""
    nb = a.shape[0]
    per = max(1, NCORES // (a.shape[1] * (b.shape[-1] // 32 // per_core_n)))
    g = math.ceil(nb / per)
    per = math.ceil(nb / g)
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=GRID, in0_block_w=in0_block_w,
        out_subblock_h=1, out_subblock_w=per_core_n, per_core_M=1, per_core_N=per_core_n)
    outs = []
    for c in range(0, nb, per):
        e = min(c + per, nb)
        ac = ttnn.slice(a, [c, 0, 0, 0], [e, a.shape[1], a.shape[2], a.shape[3]])
        bc = ttnn.slice(b, [c, 0, 0, 0], [e, b.shape[1], b.shape[2], b.shape[3]])
        outs.append(ttnn.matmul(ac, bc, program_config=pc, compute_kernel_config=CKC))
    return outs


def run(dt, name, out):
    g = torch.Generator().manual_seed(0)
    q = tt(torch.randn(NB, H, NQ, DH, generator=g), dt)
    k = tt(torch.randn(NB, H, NK, DH, generator=g), dt)
    v = tt(torch.randn(NB, H, NK, DH, generator=g), dt)
    z = tt(torch.randn(NB, H, NQ, NK, generator=g), dt)
    pb = tt(torch.randn(NB, 1, NQ, NK, generator=g), dt)

    def v0():
        sc = ttnn.matmul(q, ttnn.permute(k, (0, 1, 3, 2)), compute_kernel_config=CKC)
        sc = ttnn.multiply(sc, DH ** -0.5)
        sc = ttnn.add(ttnn.add(sc, z), pb)
        return ttnn.matmul(ttnn.softmax(sc, dim=-1), v, compute_kernel_config=CKC)

    def v1():
        kt = ttnn.permute(k, (0, 1, 3, 2))
        outs = []
        nb_c = None
        scs = bmm(q, kt, per_core_n=NK // 32, in0_block_w=1)
        c = 0
        for sc in scs:
            n = sc.shape[0]
            zc = ttnn.slice(z, [c, 0, 0, 0], [c + n, H, NQ, NK])
            pc_ = ttnn.slice(pb, [c, 0, 0, 0], [c + n, 1, NQ, NK])
            vc = ttnn.slice(v, [c, 0, 0, 0], [c + n, H, NK, DH])
            s = ttnn.multiply(sc, DH ** -0.5)
            s = ttnn.add(ttnn.add(s, zc), pc_)
            s = ttnn.softmax(s, dim=-1)
            pcfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=GRID, in0_block_w=NK // 32,
                out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=1)
            outs.append(ttnn.matmul(s, vc, program_config=pcfg, compute_kernel_config=CKC))
            c += n
        return ttnn.concat(outs, dim=0)

    t0 = timed(v0)
    t1 = timed(v1)
    r0 = ttnn.to_torch(v0())
    r1 = ttnn.to_torch(v1())
    ok = torch.equal(r0, r1)
    print(f"--- {name}: V0 {t0*1e6:8.2f} us   V1 {t1*1e6:8.2f} us   {t0/t1:5.2f}x   "
          f"bit-exact={ok}  maxabs={(r0.float()-r1.float()).abs().max().item():.3e}", flush=True)
    print(f"    x1200 calls/fold: {t0*1200*1e3:7.1f} -> {t1*1200*1e3:7.1f} ms/fold  "
          f"(saves {(t0-t1)*1200*1e3:.1f} ms/fold)", flush=True)
    out[name] = {"v0_us": round(t0 * 1e6, 2), "v1_us": round(t1 * 1e6, 2),
                 "speedup": round(t0 / t1, 3), "bit_exact": ok,
                 "saved_ms_per_fold": round((t0 - t1) * 1200 * 1e3, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/atom_window/var_card1.json")
    a = ap.parse_args()
    out = {"card": "qb2-card1", "ncores": NCORES}
    run(ttnn.float32, "fp32", out)
    run(ttnn.bfloat16, "bf16", out)
    json.dump(out, open(a.out, "w"), indent=2)
    print("wrote", a.out, flush=True)
