#!/usr/bin/env python3
"""Which of the two chunked matmuls loses bit-exactness, and can a program-config knob restore it?"""
import json, statistics as st, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

NB, H, NQ, NK, DH = 75, 4, 32, 128, 32
dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GRID = (CORE_GRID_MAIN.x, CORE_GRID_MAIN.y)
PER = 25


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


out = {}
for dt, dn in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
    g = torch.Generator().manual_seed(0)
    q = tt(torch.randn(NB, H, NQ, DH, generator=g), dt)
    kt = tt(torch.randn(NB, H, DH, NK, generator=g), dt)
    a = tt(torch.rand(NB, H, NQ, NK, generator=g), dt)
    v = tt(torch.randn(NB, H, NK, DH, generator=g), dt)
    rows = []
    for lbl, x, y, pcn, k_t in (("QK^T", q, kt, NK // 32, 1), ("A@V", a, v, 1, NK // 32)):
        ref = ttnn.to_torch(ttnn.matmul(x, y, compute_kernel_config=CKC))
        base = timed(lambda: ttnn.deallocate(ttnn.matmul(x, y, compute_kernel_config=CKC)))
        for ibw in sorted({1, 2, k_t}):
            if k_t % ibw:
                continue
            for osw in sorted({1, pcn}):
                if pcn % osw:
                    continue
                try:
                    pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=GRID, in0_block_w=ibw,
                        out_subblock_h=1, out_subblock_w=osw, per_core_M=1, per_core_N=pcn)

                    def go():
                        parts = []
                        for c in range(0, NB, PER):
                            e = min(c + PER, NB)
                            xc = ttnn.slice(x, [c, 0, 0, 0], [e, x.shape[1], x.shape[2], x.shape[3]])
                            yc = ttnn.slice(y, [c, 0, 0, 0], [e, y.shape[1], y.shape[2], y.shape[3]])
                            parts.append(ttnn.matmul(xc, yc, program_config=pc, compute_kernel_config=CKC))
                        return ttnn.concat(parts, dim=0)

                    got = ttnn.to_torch(go())
                    s = timed(lambda: ttnn.deallocate(go()))
                    ok = torch.equal(got, ref)
                    d = (got.float() - ref.float()).abs().max().item()
                    rows.append({"op": lbl, "in0_block_w": ibw, "out_subblock_w": osw,
                                 "base_us": round(base * 1e6, 2), "us": round(s * 1e6, 2),
                                 "speedup": round(base / s, 2), "bit_exact": ok, "maxabs": d})
                    print(f"  {dn} {lbl:5s} ibw={ibw} osw={osw}  {base*1e6:8.2f} -> {s*1e6:8.2f} us "
                          f"({base/s:5.2f}x)  bit-exact={ok}  maxabs={d:.3e}", flush=True)
                except Exception as e:
                    print(f"  {dn} {lbl:5s} ibw={ibw} osw={osw}  ERR {str(e)[:100]}", flush=True)
    out[dn] = rows
json.dump(out, open("perf/atom_window/exact_card1.json", "w"), indent=2)
print("wrote perf/atom_window/exact_card1.json", flush=True)
