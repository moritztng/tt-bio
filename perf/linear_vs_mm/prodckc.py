#!/usr/bin/env python3
"""W10 final: production compute_kernel_config (HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True --
tt_bio/tenstorrent.py:3260), which is what every ttnn.linear call site in tt_bio actually passes.

fp32_dest_acc_en halves DEST, so the legal out_subblock cap drops to 4 and both the timing and the
accuracy picture can differ from the fp32_dest_acc_en=False probe. Re-runs the shape sweep, the
in0_block_w ladder, and -- the parity question the recommendation turns on -- whether varying
in0_block_w on a fixed decomposition is bit-exact.
"""
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


def subblock(pm, pn, cap):
    best = (1, 1)
    for w in range(1, min(pn, cap) + 1):
        if pn % w:
            continue
        for h in range(1, min(pm, cap // w) + 1):
            if pm % h:
                continue
            if h * w > best[0] * best[1]:
                best = (h, w)
    return best


def pc_1d(gx, gy, mt, nt, bw, cap):
    pm = math.ceil(mt / (gx * gy))
    h, w = subblock(pm, nt, cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=bw, out_subblock_h=h, out_subblock_w=w,
        per_core_M=pm, per_core_N=nt, fuse_batch=True, fused_activation=None, mcast_in0=False)


def pc_2d(gx, gy, mt, nt, bw, cap):
    pm, pn = math.ceil(mt / gy), math.ceil(nt / gx)
    h, w = subblock(pm, pn, cap)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=bw, out_subblock_h=h, out_subblock_w=w,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)


SHAPES = [
    (16384, 128, 128, "117aa pair proj c_z"),
    (16384, 128, 512, "117aa pair transition up"),
    (16384, 512, 128, "117aa pair transition down"),
    (16384, 128, 32, "117aa pair->bias heads"),
    (16384, 256, 256, "117aa trimul/W6 point"),
    (102400, 128, 128, "298aa pair proj c_z"),
    (102400, 128, 512, "298aa pair transition up"),
    (102400, 512, 128, "298aa pair transition down"),
    (102400, 128, 32, "298aa pair->bias heads"),
    (102400, 256, 256, "298aa trimul/W6 point"),
    (102400, 128, 256, "298aa trimul in-proj"),
    (128, 768, 768, "117aa single proj"),
    (320, 768, 768, "298aa single proj"),
    (320, 768, 3072, "298aa single transition up"),
    (320, 3072, 768, "298aa single transition down"),
    (1792, 128, 128, "117aa atom proj"),
    (4480, 128, 128, "298aa atom proj"),
    (4480, 128, 512, "298aa atom transition up"),
]


def main():
    dev = get_device()
    ag = dev.compute_with_storage_grid_size()
    gx, gy = (13 if ag.x >= 13 else ag.x), ag.y
    grid = ttnn.CoreGrid(y=gy, x=gx)
    cap = 4  # fp32_dest_acc_en=True halves DEST
    print(f"grid {gx}x{gy} = {gx*gy} cores, subblock cap {cap}", flush=True)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    rows = []
    for M, K, N, label in SHAPES:
        mt, kt, nt = math.ceil(M / T), K // T, math.ceil(N / T)
        at = torch.randn(1, 1, M, K) * 0.1
        bt = torch.randn(1, 1, K, N) * 0.1
        ref = at.float() @ bt.float()
        a = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        b = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        gflop = 2 * M * K * N / 1e9
        gb = (M * K + K * N + M * N) * 2 / 1e9
        print(f"\n=== [{M}x{K}]@[{K}x{N}] {label}  mt={mt} kt={kt} nt={nt} "
              f"{gflop:.2f} GFLOP {gb*1e3:.1f} MB ===", flush=True)

        cands = {
            "linear_nocg": lambda: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM),
            "linear_cg": lambda: ttnn.linear(a, b, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM),
            "mm_default": lambda: ttnn.experimental.minimal_matmul(a, b, memory_config=DRAM),
        }
        for bw in [d for d in range(1, kt + 1) if kt % d == 0]:
            for tag, mk in (("1d", pc_1d), ("2d", pc_2d)):
                try:
                    pc = mk(gx, gy, mt, nt, bw, cap)
                except Exception:
                    continue
                cands[f"linear_{tag}_bw{bw}"] = (lambda p: (lambda: ttnn.linear(
                    a, b, compute_kernel_config=ckc, memory_config=DRAM, program_config=p)))(pc)

        rec = {"M": M, "K": K, "N": N, "label": label, "mt": mt, "kt": kt, "nt": nt,
               "gflop": round(gflop, 3), "gb_mb": round(gb * 1e3, 2), "v": {}}
        outs = {}
        for name, fn in cands.items():
            try:
                y = fn()
                yt = ttnn.to_torch(y).float()
                ttnn.deallocate(y)
                ms = timed(dev, lambda: ttnn.deallocate(fn()))
            except Exception as e:
                msg = str(e).split("\n")[0][:70]
                rec["v"][name] = None
                if "bw" not in name:
                    print(f"  {name:18s} ERR {msg}", flush=True)
                continue
            rmsd = float((yt - ref).pow(2).mean().sqrt())
            outs[name] = yt
            rec["v"][name] = {"ms": round(ms, 4), "tflops": round(gflop / (ms / 1e3) / 1e3, 2),
                              "gbs": round(gb / (ms / 1e3), 1), "rmsd": rmsd}
            print(f"  {name:18s} {ms:9.4f} ms {gflop/(ms/1e3)/1e3:7.2f} TF/s {gb/(ms/1e3):7.1f} GB/s "
                  f"rmsd {rmsd:.5f}", flush=True)

        # parity: does in0_block_w change arithmetic? does minimal_matmul?
        base = outs.get("linear_cg")
        if base is not None:
            eq = {k: bool(torch.equal(base, v)) for k, v in outs.items() if k != "linear_cg"}
            rec["bitexact_vs_linear_cg"] = eq
            bw_keys = [k for k in eq if k.startswith("linear_1d_bw") or k.startswith("linear_2d_bw")]
            allbw = all(eq[k] for k in bw_keys) if bw_keys else None
            rec["all_program_configs_bitexact"] = allbw
            mmeq = eq.get("mm_default")
            mmmax = (float((base - outs["mm_default"]).abs().max()) if "mm_default" in outs else None)
            rec["mm_maxabs_vs_cg"] = mmmax
            print(f"  bit-exact: nocg={eq.get('linear_nocg')} all_program_configs={allbw} "
                  f"mm={mmeq} mm_maxabs={mmmax}", flush=True)

        best = min(((k, v["ms"]) for k, v in rec["v"].items() if v), key=lambda kv: kv[1])
        cg = rec["v"].get("linear_cg")
        if cg:
            rec["best"] = {"variant": best[0], "ms": best[1], "speedup_vs_cg": round(cg["ms"] / best[1], 3)}
            print(f"  BEST {best[0]} {best[1]:.4f} ms = {cg['ms']/best[1]:.3f}x vs linear_cg", flush=True)
        rows.append(rec)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        for v in outs.values():
            del v

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prodckc_results.json")
    json.dump(rows, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
