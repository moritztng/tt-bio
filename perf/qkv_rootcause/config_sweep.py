#!/usr/bin/env python3
"""Decisive sweep: does widening in0_block_w (K-blocking) and filling the grid
recover the qkv projection's 3x roofline gap?  All configs legality-checked
against num_blocks_total <= num_cores.  Sync-bracketed, warmed, bit-exactness
checked against the production baseline."""
import json, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

N, C_Z, H, D = 128, 256, 8, 32
M_T, K_T, N_T = (N * N) // 32, C_Z // 32, (3 * H * D) // 32   # 512, 8, 24
GF = 2 * (N * N) * C_Z * (3 * H * D) / 1e9


def med(xs): return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=8, pipe=12, reps=5):
    for _ in range(warm): fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe): fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(out)


def main():
    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    print(f"CORE_GRID_MAIN={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}  device={dg.x}x{dg.y}", flush=True)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    ckc_n = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=True)

    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(1, N, N, C_Z), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn(C_Z, 3 * H * D), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    res, ref = {}, None

    def call(cfg, kc):
        kw = {"program_config": cfg} if cfg is not None else {"core_grid": CORE_GRID_MAIN}
        return ttnn.linear(x, w, compute_kernel_config=kc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw)

    def record(name, cfg, kc=ckc):
        nonlocal ref
        try:
            o = call(cfg, kc); t = ttnn.to_torch(o); ttnn.deallocate(o)
            ms = timed(dev, lambda: ttnn.deallocate(call(cfg, kc)))
            ex = None if ref is None else bool(torch.equal(t, ref))
            if ref is None: ref = t
            tf = GF / (ms / 1e3) / 1e3
            res[name] = {"ms": round(ms, 4), "tflops": round(tf, 2), "bit_exact": ex}
            print(f"{name:46s} {ms:8.4f} ms {tf:7.2f} TF/s  exact={ex}", flush=True)
        except Exception as e:
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
            print(f"{name:46s} ERR {str(e)[:95]}", flush=True)

    record("BASELINE_production_autoconfig", None)

    def cfg(gxy, pm, bw, sbh, sbw, pn=N_T):
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=gxy, in0_block_w=bw,
            out_subblock_h=sbh, out_subblock_w=sbw, out_block_h=pm, out_block_w=pn,
            per_core_M=pm, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=False)

    for (gxy, tag) in (((CORE_GRID_MAIN.x, CORE_GRID_MAIN.y), "g11x10"), ((dg.x, dg.y), "g13x10")):
        nc = gxy[0] * gxy[1]
        for pm in (4, 5, 8):
            if -(-M_T // pm) > nc or M_T % pm:      # legality + exact tiling
                continue
            for bw in (1, 2, 4, 8):
                for sbh, sbw in ((1, 4), (2, 2), (1, 2)):
                    if pm % sbh or N_T % sbw:
                        continue
                    record(f"{tag}_M{pm}_bw{bw}_sb{sbh}x{sbw}", cfg(gxy, pm, bw, sbh, sbw))
        # best-case K-blocking without fp32 dest acc (bigger subblocks, bf16 interm)
        for pm in (4, 8):
            if -(-M_T // pm) > nc or M_T % pm:
                continue
            for bw in (1, 8):
                for sbh, sbw in ((2, 4), (1, 8)):
                    if pm % sbh or N_T % sbw:
                        continue
                    record(f"{tag}_M{pm}_bw{bw}_sb{sbh}x{sbw}_NOfp32", cfg(gxy, pm, bw, sbh, sbw), ckc_n)

    ok = {k: v for k, v in res.items() if "ms" in v}
    best = min(ok, key=lambda k: ok[k]["ms"])
    print(f"\nBEST: {best} {ok[best]}   vs baseline {ok['BASELINE_production_autoconfig']}", flush=True)
    print(f"SPEEDUP: {ok['BASELINE_production_autoconfig']['ms'] / ok[best]['ms']:.3f}x", flush=True)
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qkv_sweep2.json", "w"), indent=2)


main()
