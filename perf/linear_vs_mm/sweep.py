#!/usr/bin/env python3
"""W10 -- ttnn.linear(core_grid=...) vs ttnn.experimental.minimal_matmul across the shapes tt-bio
actually runs, plus a program-config probe that names the mechanism.

W6 measured one point ([102400,256]@[256,256] bf16: linear+core_grid 0.678 ms, minimal_matmul
0.408, 1.66x). One shape is an anecdote. This sweeps M, K, N over the pair track, single track and
atom track at 117 aa (N_tok=128) and 298 aa (N_tok=320), and for every shape also runs ttnn.linear
with a HAND-BUILT 1D program config at in0_block_w = full K, and at in0_block_w = 1. If the tuned
program config closes the gap to minimal_matmul, the mechanism is ttnn's program-config SELECTION,
not the kernel.

Host-timed with pipe=6 and a synchronize_device on both sides of the timed region (WARROOM 2.4).
Operands and result in DRAM, which is what the call sites use. qb2 / ttnn 0.68.0: ratios only.
"""
import json, math, os, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
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
    """Largest legal (h, w) with h*w <= cap, pm % h == 0, pn % w == 0."""
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


def pc_1d(grid, mt, kt, nt, in0_block_w):
    """1D multicast-in1 config: in0 row-sharded over cores, in1 multicast. Tall-skinny shape."""
    ncores = grid[0] * grid[1]
    pm = math.ceil(mt / ncores)
    pn = nt
    h, w = subblock(pm, pn)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=h,
        out_subblock_w=w,
        per_core_M=pm,
        per_core_N=pn,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )


# ---- shape families actually present in tt_bio ------------------------------------------------
# pair track: M = N_tok^2 (row-major flattened pair tensor), K/N = c_z and its 4x transition
# single track: M = N_tok, K/N = c_s = 768 (TOKEN_DIM) and c_s*4
# atom track: M = n_atoms ~ 14*N_tok, K/N = ATOM_DIM = 128
def shapes():
    out = []
    for ntok, tag in ((128, "117aa"), (320, "298aa")):
        M = ntok * ntok
        for k, n, what in (
            (128, 128, "pair proj c_z->c_z"),
            (128, 512, "pair transition up 4x"),
            (512, 128, "pair transition down"),
            (128, 32, "pair->bias heads (pad 32)"),
            (256, 256, "W6 point / trimul c_z=256"),
            (128, 256, "trimul in-proj"),
        ):
            out.append((M, k, n, f"{tag} {what}"))
        # single track
        for k, n, what in ((768, 768, "single proj"), (768, 3072, "single transition up"),
                           (3072, 768, "single transition down")):
            out.append((ntok, k, n, f"{tag} {what}"))
        # atom track
        na = 14 * ntok
        for k, n, what in ((128, 128, "atom proj"), (128, 512, "atom transition up")):
            out.append((na, k, n, f"{tag} {what}"))
    return out


def main():
    dev = get_device()
    ag = dev.compute_with_storage_grid_size()
    gx = 13 if ag.x >= 13 else ag.x
    gy = ag.y
    grid = ttnn.CoreGrid(y=gy, x=gx)
    pgrid = ttnn.CoreCoord(gx, gy)
    print(f"device grid {ag.x}x{ag.y}, using {gx}x{gy} = {gx*gy} cores", flush=True)

    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=False, packer_l1_acc=False)

    torch.manual_seed(0)
    rows = []
    for M, K, N, label in shapes():
        mt, kt, nt = M // T, math.ceil(K / T), math.ceil(N / T)
        at = torch.randn(1, 1, M, K) * 0.1
        bt = torch.randn(1, 1, K, N) * 0.1
        a = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        b = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        gflop = 2 * M * K * N / 1e9
        gb = (M * K + K * N + M * N) * 2 / 1e9  # bf16 read+read+write
        rec = {"M": M, "K": K, "N": N, "label": label, "mt": mt, "kt": kt, "nt": nt,
               "gflop": round(gflop, 3), "gb": round(gb, 4)}

        variants = [
            ("linear_nocg", lambda: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM)),
            ("linear_cg", lambda: ttnn.linear(a, b, compute_kernel_config=ckc, core_grid=grid, memory_config=DRAM)),
            ("mm_default", lambda: ttnn.experimental.minimal_matmul(a, b, memory_config=DRAM)),
        ]
        if mt >= 1 and nt >= 1:
            try:
                variants.append(("linear_pc_kfull",
                                 lambda: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM,
                                                     program_config=pc_1d(pgrid, mt, kt, nt, kt))))
                variants.append(("linear_pc_k1",
                                 lambda: ttnn.linear(a, b, compute_kernel_config=ckc, memory_config=DRAM,
                                                     program_config=pc_1d(pgrid, mt, kt, nt, 1))))
            except Exception as e:
                print(f"  pc build err {label}: {str(e)[:80]}", flush=True)

        print(f"[{M}x{K}@{K}x{N}] {label}  mt={mt} kt={kt} nt={nt}  {gflop:.2f} GFLOP {gb*1e3:.1f} MB",
              flush=True)
        for name, fn in variants:
            try:
                ms = timed(dev, lambda: ttnn.deallocate(fn()))
            except Exception as e:
                rec[name] = None
                print(f"    {name:16s} ERR {str(e)[:90]}", flush=True)
                continue
            rec[name] = round(ms, 4)
            print(f"    {name:16s} {ms:9.4f} ms  {gflop/(ms/1e3)/1e3:7.2f} TFLOP/s  "
                  f"{gb/(ms/1e3):7.1f} GB/s", flush=True)
        if rec.get("linear_cg") and rec.get("mm_default"):
            rec["cg_over_mm"] = round(rec["linear_cg"] / rec["mm_default"], 3)
            print(f"    -> linear_cg / mm_default = {rec['cg_over_mm']:.3f}x", flush=True)
        rows.append(rec)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.json")
    json.dump(rows, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
