#!/usr/bin/env python3
"""The four 298 aa shapes where minimal_matmul beat every ttnn.linear config I could build all
failed the same way: a 1D M-split with the full N strip per core overflows L1, because I left
out_block_h/out_block_w at their per_core defaults. tt_bio already knows better --
_tri_att_qkv_l1_config (tenstorrent.py:287) sets out_block_h/out_block_w explicitly. Retry the 1D
config WITH output blocking and see whether linear can reach minimal_matmul at 298 aa."""
import json, math, os, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
T = 32


def med(xs):
    return sorted(xs)[len(xs) // 2]


def timed(dev, fn, warm=3, pipe=6, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return med(o)


SHAPES = [
    (102400, 128, 512, "298aa pair transition up"),
    (102400, 512, 128, "298aa pair transition down"),
    (102400, 256, 256, "298aa trimul/W6 point"),
    (102400, 128, 256, "298aa trimul in-proj"),
    (16384, 512, 128, "117aa pair transition down"),
]


def main():
    dev = get_device()
    ag = dev.compute_with_storage_grid_size()
    gx, gy = (13 if ag.x >= 13 else ag.x), ag.y
    nc = gx * gy
    grid = ttnn.CoreGrid(y=gy, x=gx)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    out = []
    for M, K, N, label in SHAPES:
        mt, kt, nt = M // T, K // T, math.ceil(N / T)
        at, bt = torch.randn(1, 1, M, K) * 0.1, torch.randn(1, 1, K, N) * 0.1
        ref = at.float() @ bt.float()
        a = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        b = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        gflop, gb = 2 * M * K * N / 1e9, (M * K + K * N + M * N) * 2 / 1e9
        pcm = next(p for p in range(-(-mt // nc), mt + 1) if mt % p == 0)
        print(f"\n=== [{M}x{K}]@[{K}x{N}] {label} mt={mt} kt={kt} nt={nt} per_core_M={pcm} ===", flush=True)
        y = ttnn.experimental.minimal_matmul(a, b, memory_config=DRAM)
        y_mm = ttnn.to_torch(y).float()
        ttnn.deallocate(y)
        ms_mm = timed(dev, lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(a, b, memory_config=DRAM)))
        y = ttnn.linear(a, b, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM)
        y_cg = ttnn.to_torch(y).float()
        ttnn.deallocate(y)
        ms_cg = timed(dev, lambda: ttnn.deallocate(
            ttnn.linear(a, b, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM)))
        print(f"  {'linear_cg':30s} {ms_cg:8.4f} ms  {gb/(ms_cg/1e3):7.1f} GB/s  "
              f"rmsd {float((y_cg-ref).pow(2).mean().sqrt()):.5f}", flush=True)
        print(f"  {'mm_default':30s} {ms_mm:8.4f} ms  {gb/(ms_mm/1e3):7.1f} GB/s  "
              f"rmsd {float((y_mm-ref).pow(2).mean().sqrt()):.5f}", flush=True)
        rec = {"M": M, "K": K, "N": N, "label": label, "ms_cg": ms_cg, "ms_mm": ms_mm, "v": {}}
        for bw in sorted({kt, max(1, kt // 2), min(kt, 8)}, reverse=True):
            for obh in [d for d in (1, 2, 4, 8) if pcm % d == 0]:
                for obw in [d for d in (nt, max(1, nt // 2), 4, 2, 1) if nt % d == 0]:
                    sh = max((h for h in range(min(4, obh), 0, -1) if obh % h == 0), default=1)
                    sw = max((w for w in range(min(4 // sh, obw), 0, -1) if obw % w == 0), default=1)
                    name = f"1d_bw{bw}_obh{obh}_obw{obw}"
                    try:
                        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                            out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
                            per_core_M=pcm, per_core_N=nt, fuse_batch=True,
                            fused_activation=None, mcast_in0=False)
                        z = ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM, program_config=pc)
                        zt = ttnn.to_torch(z).float()
                        ttnn.deallocate(z)
                        ms = timed(dev, lambda: ttnn.deallocate(ttnn.linear(
                            a, b, compute_kernel_config=ckc, memory_config=DRAM, program_config=pc)))
                    except Exception:
                        continue
                    rmsd = float((zt - ref).pow(2).mean().sqrt())
                    eq = bool(torch.equal(zt, y_cg))
                    rec["v"][name] = {"ms": round(ms, 4), "rmsd": rmsd, "bitexact_vs_cg": eq}
                    print(f"  {name:30s} {ms:8.4f} ms  {gb/(ms/1e3):7.1f} GB/s  rmsd {rmsd:.5f}  "
                          f"eq_cg {eq}  {ms_cg/ms:.3f}x vs cg  {ms_mm/ms:.3f}x vs mm", flush=True)
        if rec["v"]:
            bk = min(rec["v"], key=lambda k: rec["v"][k]["ms"])
            rec["best"] = {"variant": bk, **rec["v"][bk],
                           "x_vs_cg": round(ms_cg / rec["v"][bk]["ms"], 3),
                           "x_vs_mm": round(ms_mm / rec["v"][bk]["ms"], 3)}
            print(f"  BEST {bk} {rec['v'][bk]['ms']:.4f} ms = {ms_cg/rec['v'][bk]['ms']:.3f}x vs cg, "
                  f"{ms_mm/rec['v'][bk]['ms']:.3f}x vs mm", flush=True)
        out.append(rec)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocked1d_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
