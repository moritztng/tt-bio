#!/usr/bin/env python3
"""W10 mechanism probe + parity.

The sweep showed the linear_cg / minimal_matmul ratio swings 0.48x -> 2.32x with shape, and that
`core_grid=CORE_GRID_MAIN` is itself a pessimisation on the pair track (M tiles >> N tiles) and a
win on the single/atom track (M tiles <= grid rows). Hypothesis: the only thing that matters is the
DECOMPOSITION -- a 1D M-split with the full N per core, versus a 2D MxN split over the grid --
and `core_grid=` forces ttnn onto the 2D path.

Prediction, two-sided:
  * pair track: linear with a hand-built 1D config (in0_block_w = full K) matches minimal_matmul,
    and linear with a hand-built 2D config matches linear_cg.
  * single track: the reverse.
If both hold, the mechanism is program-config selection, not the kernel.

Also emits parity (torch.equal / PCC / rmsd) for every variant against the fp32 torch reference,
because minimal_matmul and linear differ in reduction order and bit-exactness cannot be assumed.
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


def subblock(pm, pn, cap=8):
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


def pc_1d(gx, gy, mt, kt, nt, in0_block_w):
    pm = math.ceil(mt / (gx * gy))
    pn = nt
    h, w = subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w, out_subblock_h=h, out_subblock_w=w,
        per_core_M=pm, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=False)


def pc_2d(gx, gy, mt, kt, nt, in0_block_w):
    pm = math.ceil(mt / gy)
    pn = math.ceil(nt / gx)
    h, w = subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=in0_block_w, out_subblock_h=h, out_subblock_w=w,
        per_core_M=pm, per_core_N=pn, transpose_mcast=False, fused_activation=None)


def divisors_le(kt, cap):
    return [d for d in range(1, kt + 1) if kt % d == 0 and d <= cap]


SHAPES = [
    (102400, 256, 256, "298aa pair, W6 point"),
    (102400, 512, 128, "298aa pair transition down"),
    (102400, 128, 32, "298aa pair->bias heads"),
    (16384, 256, 256, "117aa pair, W6 point"),
    (320, 3072, 768, "298aa single transition down"),
    (4480, 128, 128, "298aa atom proj"),
]


def main():
    dev = get_device()
    ag = dev.compute_with_storage_grid_size()
    gx, gy = (13 if ag.x >= 13 else ag.x), ag.y
    grid = ttnn.CoreGrid(y=gy, x=gx)
    print(f"grid {gx}x{gy} = {gx*gy} cores", flush=True)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False, packer_l1_acc=False)

    torch.manual_seed(0)
    out = []
    for M, K, N, label in SHAPES:
        mt, kt, nt = M // T, K // T, math.ceil(N / T)
        at = torch.randn(1, 1, M, K) * 0.1
        bt = torch.randn(1, 1, K, N) * 0.1
        ref = (at.float() @ bt.float())
        a = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        b = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        gflop = 2 * M * K * N / 1e9
        print(f"\n=== [{M}x{K}]@[{K}x{N}] {label}  mt={mt} kt={kt} nt={nt} ===", flush=True)

        cands = {
            "linear_nocg": lambda: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM),
            "linear_cg": lambda: ttnn.linear(a, b, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM),
            "mm_default": lambda: ttnn.experimental.minimal_matmul(a, b, memory_config=DRAM),
        }
        for bw in divisors_le(kt, kt):
            try:
                pc = pc_1d(gx, gy, mt, kt, nt, bw)
            except Exception as e:
                print(f"  pc_1d bw={bw} build ERR {str(e)[:70]}", flush=True)
                continue
            cands[f"linear_1d_bw{bw}"] = (lambda p: (lambda: ttnn.linear(
                a, b, compute_kernel_config=ckc, memory_config=DRAM, program_config=p)))(pc)
        for bw in divisors_le(kt, kt):
            try:
                pc = pc_2d(gx, gy, mt, kt, nt, bw)
            except Exception as e:
                print(f"  pc_2d bw={bw} build ERR {str(e)[:70]}", flush=True)
                continue
            cands[f"linear_2d_bw{bw}"] = (lambda p: (lambda: ttnn.linear(
                a, b, compute_kernel_config=ckc, memory_config=DRAM, program_config=p)))(pc)

        rec = {"M": M, "K": K, "N": N, "label": label, "mt": mt, "kt": kt, "nt": nt, "v": {}}
        for name, fn in cands.items():
            try:
                y = fn()
                yt = ttnn.to_torch(y).float()
                ttnn.deallocate(y)
                d = (yt - ref)
                rmsd = float(d.pow(2).mean().sqrt())
                pcc = float(torch.corrcoef(torch.stack([yt.flatten(), ref.flatten()]))[0, 1])
                ms = timed(dev, lambda: ttnn.deallocate(fn()))
            except Exception as e:
                print(f"  {name:18s} ERR {str(e)[:100]}", flush=True)
                rec["v"][name] = None
                continue
            rec["v"][name] = {"ms": round(ms, 4), "tflops": round(gflop / (ms / 1e3) / 1e3, 2),
                              "rmsd": rmsd, "pcc": pcc}
            print(f"  {name:18s} {ms:9.4f} ms {gflop/(ms/1e3)/1e3:7.2f} TF/s  "
                  f"rmsd {rmsd:.5f} pcc {pcc:.6f}", flush=True)

        # bit-exactness between the candidate op swaps, on device output
        try:
            y_cg = ttnn.to_torch(ttnn.linear(a, b, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM))
            y_no = ttnn.to_torch(ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM))
            y_mm = ttnn.to_torch(ttnn.experimental.minimal_matmul(a, b, memory_config=DRAM))
            rec["bitexact_cg_vs_nocg"] = bool(torch.equal(y_cg, y_no))
            rec["bitexact_cg_vs_mm"] = bool(torch.equal(y_cg, y_mm))
            rec["maxabs_cg_vs_mm"] = float((y_cg.float() - y_mm.float()).abs().max())
            rec["maxabs_cg_vs_nocg"] = float((y_cg.float() - y_no.float()).abs().max())
            print(f"  bit-exact cg==nocg {rec['bitexact_cg_vs_nocg']}  cg==mm {rec['bitexact_cg_vs_mm']}"
                  f"  maxabs cg-mm {rec['maxabs_cg_vs_mm']:.3e}  cg-nocg {rec['maxabs_cg_vs_nocg']:.3e}",
                  flush=True)
        except Exception as e:
            print("  parity ERR", str(e)[:100], flush=True)

        out.append(rec)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mechanism_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
