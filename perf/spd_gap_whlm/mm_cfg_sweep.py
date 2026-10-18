"""ESMC's minimal_matmul projections at the WH page shape, swept over MinimalMatmulConfig blocks, against float64.

The WH census (spd-gap-whlm cen, main fe61d408f) runs ESMC-300M's projections on minimal_matmul's DEFAULT config at
27-36 TF/s against a ~72 TF/s HiFi2 / ~56 HiFi4 roof on a Galaxy chip at 1000 MHz: the fc1 half [3808, 960] x [960, 2560]
0.52 ms a call, the 960 x 960 projections 0.26 ms. perf/spd_gap_esmf/lin_bench.py only timed that default against
ttnn.linear programs; this sweeps the op's own blocks (M, K, N block in tiles, the 2x2 / 1x4 / 4x1 subblock that fits).
Per arm: device ms (median of --repeat, --inner calls per synchronize), TF/s, rel_l2 and max error / peak against
float64 of the same bf16 operands, and whether the output equals the default config's byte for byte.

    python3 perf/spd_gap_whlm/mm_cfg_sweep.py --model esmc-300m [--wdtype bfloat8_b] --out mm.jsonl
"""
import argparse
import itertools
import json
import math
import time

import torch
import ttnn

# (d_model, d_ff, rows of the page batch); fc1 runs as two halves of d_ff each (split SwiGLU).
MODELS = {"esmc-300m": (960, 2560, 7 * 544), "esmc-600m": (1152, 3072, 7 * 544), "esmc-6b": (2560, 6912, 544)}


def _outs(o):
    return o if isinstance(o, list) else [o]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-300m", choices=sorted(MODELS))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=5)
    ap.add_argument("--wdtype", default="bfloat16", choices=["bfloat16", "bfloat8_b"],
                    help="weight dtype on device; --fast loads the projections as bfloat8_b")
    ap.add_argument("--acc", default="fp32", choices=["fp32", "bf16"],
                    help="dest accumulation; bf16 doubles the dest registers, so 8-tile subblocks become legal")
    ap.add_argument("--shapes", default="proj,fc1h,fc2",
                    help="comma list of proj, fc1h, fc2, qkv (q/k/v from one minimal_matmul_split, as the model runs it)")
    ap.add_argument("--fidelity", default="HiFi3", choices=["HiFi2", "HiFi3", "HiFi4"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
                                                math_approx_mode=False, fp32_dest_acc_en=args.acc == "fp32", packer_l1_acc=True)
    d, ff, m = MODELS[args.model]
    shapes = {"proj": (m, d, d), "fc1h": (m, d, ff), "fc2": (m, ff, d), "qkv": (m, d, 3 * d)}
    shapes = {k: shapes[k] for k in args.shapes.split(",")}
    out = open(args.out, "w")
    gen = torch.Generator().manual_seed(0)
    for name, (M, K, N) in shapes.items():
        x = torch.randn(M, K, generator=gen).to(torch.bfloat16)
        w = (torch.randn(K, N, generator=gen) / math.sqrt(K)).to(torch.bfloat16)
        if args.wdtype == "bfloat8_b":  # the reference multiplies the weights the device holds
            w = ttnn.to_torch(ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)).to(torch.bfloat16)
        ref = x.double() @ w.double()
        xd = ttnn.from_torch(x.reshape(1, 1, M, K), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        wd = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=dev, dtype=getattr(ttnn, args.wdtype))
        kt = K // 32
        arms = {"default": None}
        for mb, kb, nb in itertools.product((2, 4, 8, 16), (1, 2, 4, 8), (2, 4, 8, 16)):
            if kt % kb:
                continue
            for sh, sw in ((2, 2), (1, 4), (4, 1)) if args.acc == "fp32" else ((2, 4), (4, 2), (1, 8), (8, 1), (2, 2)):
                if mb % sh == 0 and nb % sw == 0:
                    arms[f"m{mb}k{kb}n{nb}s{sh}x{sw}"] = ttnn.MinimalMatmulConfig(
                        M_block_size=mb, K_block_size=kb, N_block_size=nb, subblock_h=sh, subblock_w=sw,
                        compute_with_storage_grid_size=ttnn.CoreCoord(g.x, g.y))
                    break
        if name == "qkv":  # the model's own config first (tt_bio.esmc._mm_config), then the sweep
            from tt_bio import esmc
            arms = {"default": None, "model": esmc._mm_config(xd, wd).get("config"), **{k: v for k, v in arms.items() if v}}
        else:
            arms["linear"] = "linear"  # what a projection under ESMC_MM_MIN_ROWS (the 6B's 544) runs today
        base = None
        for arm, cfg in arms.items():
            fn = (lambda cfg=cfg: list(ttnn.experimental.minimal_matmul_split(
                    xd, wd, chunks=3, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    **({"config": cfg} if cfg else {})))) if name == "qkv" else \
                (lambda: ttnn.linear(xd, wd, compute_kernel_config=ck, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN)) \
                if cfg == "linear" else (lambda cfg=cfg: ttnn.experimental.minimal_matmul(
                    xd, wd, compute_kernel_config=ck, dtype=ttnn.bfloat16, **({"config": cfg} if cfg else {})))
            rec = {"model": args.model, "wdtype": args.wdtype, "acc": args.acc, "shape": name, "mkn": [M, K, N], "arm": arm, "grid": [g.x, g.y]}
            try:
                o = _outs(fn())
                got = torch.cat([ttnn.to_torch(t).reshape(M, -1) for t in o], dim=-1).double()
                for t in o:
                    ttnn.deallocate(t)
                ts = []
                for _ in range(args.repeat):
                    ttnn.synchronize_device(dev)
                    t0 = time.perf_counter()
                    for _ in range(args.inner):
                        for t in _outs(fn()):
                            ttnn.deallocate(t)
                    ttnn.synchronize_device(dev)
                    ts.append((time.perf_counter() - t0) / args.inner * 1e3)
                ms = sorted(ts)[len(ts) // 2]
                err = got - ref
                rec.update(ms=round(ms, 4), tflops=round(2 * M * K * N / ms / 1e9, 2),
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
