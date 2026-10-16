"""ESMC-6B's projections at the WH page shape (one 544-row sequence), ttnn.linear's auto program against explicit
1D (in0 multicast, N across every core) and 2D multicast programs, against float64.

cen4/cen5 on WH: the 6B's linears are 198 ms of a 297 ms forward, ~36 TF/s against a 56 TF/s HiFi4 roof, and neither
HiFi2 (fd, -1.3..-2.5 %) nor any minimal_matmul block (mc) moves them. At 544 rows M is 17 tiles, a prime: a 2D grid
of 8 or 9 rows leaves cores with 2 tiles and one with 1, while a 1D program keeps all 17 rows on every core and splits
only N. This times both families per shape (qkv, out, fc1 half, fc2) with the 6B's fidelity and weight dtype.

    python3 perf/spd_gap_whlm/lin6b_sweep.py [--wdtype bfloat8_b] --out lin6b.jsonl
"""
import argparse
import json
import math
import time

import torch
import ttnn

M, D, FF = 544, 2560, 6912
SHAPES = {"qkv": (D, 3 * D), "out": (D, D), "fc1h": (D, FF), "fc2": (FF, D)}


def subblock(h_tiles, w_tiles, cap):
    """Largest out subblock (h, w) with h | h_tiles, w | w_tiles, h * w <= cap, wide first."""
    best = (1, 1)
    for h in range(1, h_tiles + 1):
        for w in range(1, w_tiles + 1):
            if h_tiles % h == 0 and w_tiles % w == 0 and h * w <= cap and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def arms(gx, gy, K, N, cap):
    mt, kt, nt = M // 32, K // 32, N // 32
    cores = gx * gy
    out = {"linear_grid": "grid", "linear_auto": "auto"}
    for kb in (1, 2, 4, 8):
        if kt % kb:
            continue
        pn = math.ceil(nt / cores)
        sh, sw = subblock(mt, pn, cap)
        out[f"1d_k{kb}_n{pn}"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=kb, out_subblock_h=sh, out_subblock_w=sw,
            per_core_M=mt, per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True)
        for y in sorted({gy, 6, 3, 2}):
            if y > gy:
                continue
            pm, pn2 = math.ceil(mt / y), math.ceil(nt / gx)
            sh, sw = subblock(pm, pn2, cap)
            out[f"2d_y{y}_k{kb}_m{pm}n{pn2}"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, y), in0_block_w=kb, out_subblock_h=sh, out_subblock_w=sw,
                per_core_M=pm, per_core_N=pn2, transpose_mcast=False, fused_activation=None, fuse_batch=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--wdtype", default="bfloat16", choices=["bfloat16", "bfloat8_b"])
    ap.add_argument("--fidelity", default="HiFi4", help="the 6B (ESMCLanguageModel) runs HiFi4 in both modes")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    out = open(args.out, "w")
    gen = torch.Generator().manual_seed(0)
    for name in args.shapes.split(","):
        K, N = SHAPES[name]
        x = torch.randn(M, K, generator=gen).to(torch.bfloat16)
        w = (torch.randn(K, N, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        if args.wdtype == "bfloat8_b":  # the reference multiplies the weights the device holds
            w = ttnn.to_torch(ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)).to(torch.bfloat16)
        ref = x.double() @ w.double()
        xd = ttnn.from_torch(x.reshape(1, 1, M, K), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=getattr(ttnn, args.wdtype))
        base = None
        for arm, cfg in arms(g.x, g.y, K, N, cap=4).items():
            kw = {"core_grid": T.CORE_GRID_MAIN} if cfg == "grid" else {} if cfg == "auto" else {"program_config": cfg}
            fn = lambda kw=kw: ttnn.linear(xd, wd, compute_kernel_config=ck, dtype=ttnn.bfloat16, **kw)
            rec = {"wdtype": args.wdtype, "fidelity": args.fidelity, "shape": name, "mkn": [M, K, N], "arm": arm,
                   "grid": [g.x, g.y]}
            try:
                o = fn()
                got = ttnn.to_torch(o).reshape(M, N).double()
                ttnn.deallocate(o)
                ts = []
                for _ in range(args.repeat):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(args.inner):
                        ttnn.deallocate(fn())
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) / args.inner * 1e3)
                ms = sorted(ts)[len(ts) // 2]
                err = got - ref
                rec.update(ms=round(ms, 4), tflops=round(2 * M * K * N / ms / 1e9, 2),
                           gbs_w=round(K * N * (2 if args.wdtype == "bfloat16" else 1.0625) / ms / 1e6, 1),
                           finite=bool(torch.isfinite(got).all()),
                           rel_l2=float(err.norm() / ref.norm()), max_err_peak=float(err.abs().max() / ref.abs().max()))
                if base is None:
                    base = got
                rec["equal_default"] = bool(torch.equal(got, base))
            except Exception as e:  # an L1 refusal or an illegal config is a result, not a crash
                rec["error"] = str(e).splitlines()[0][:200]
            print(json.dumps(rec), flush=True)
            out.write(json.dumps(rec) + "\n")
            out.flush()
        ttnn.deallocate(xd); ttnn.deallocate(wd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
