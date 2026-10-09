"""Wrong-pixel rate of one matmul shape on the device under several K blockings and fidelities, vs float64.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_wherr/site_probe.py --m 2560 --k 768 --n 3072 \
        --variants auto,kb1,kb2,kb4 --fid HiFi4,HiFi3 --draws 8 --out probe.json

a is N(0, 1) (a layer-norm output), w is N(0, 1/k) (an initialised projection), both rounded to bf16, so every
output is ~N(0, 1). `auto` is how tt-bio calls it (ttnn.linear with core_grid = the full grid, ttnn picks the
program); `auto3d` the same with the input as [batch, m/batch, k], the shape the fold passes; `kbN` is a 2D multicast program with in0_block_w = N (`kbfull` = all of K) on the same grid. fp32 dest
acc and packer L1 acc on, bf16 output, as the trunk and diffusion run in normal mode. A pixel is wrong under the
same rule as perf/spd_wherr/audit.py: error above 8 bf16 ulps of the float64 value and above 16x the call's rms
error. The time per call (median of 5 after a warm call) rides along, so the cost of a clean config is known.
"""
import argparse, json, statistics, time
from pathlib import Path

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402


def wrong(R, Y):
    err = (Y - R).abs()
    rms = err.pow(2).mean().sqrt().item()
    ulp = torch.exp2(torch.floor(torch.log2(R.abs().clamp_min(1e-30))) - 7)
    bad = (err > 8 * ulp) & (err > 16 * rms)
    big = bad & (err > 0.25)
    worst = [[round(float(R[bad][i]), 4), round(float(Y[bad][i]), 4)] for i in err[bad].topk(min(3, int(bad.sum()))).indices.tolist()] if bad.any() else []
    return int(bad.sum()), int(big.sum()), float(err.max()), rms, worst


def program(variant, mt, kt, nt, grid):
    if variant.startswith("auto"):
        return None
    w = kt if variant == "kbfull" else int(variant[2:])
    if kt % w:
        return "skip"
    gx, gy = grid
    pm, pn = -(-mt // gy), -(-nt // gx)
    sw = max(s for s in range(1, min(4, pn) + 1) if pn % s == 0)
    sh = max(h for h in range(1, min(4 // sw, pm) + 1) if pm % h == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=w, out_subblock_h=sh, out_subblock_w=sw,
        out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn, transpose_mcast=False,
        fused_activation=None, fuse_batch=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--variants", default="auto,kb1,kb2,kb4,kbfull")
    ap.add_argument("--fid", default="HiFi4,HiFi3")
    ap.add_argument("--batch", type=int, default=5, help="leading dim of the auto3d variant's input, as the fold passes it")
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(6)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size(); grid = (g.x, g.y)
    mt, kt, nt = a.m // 32, a.k // 32, a.n // 32
    res = {"m": a.m, "k": a.k, "n": a.n, "grid": grid, "arch": str(dev.arch()), "cells": []}
    try:
        for d in range(a.draws):
            torch.manual_seed(a.seed0 + d)
            A = torch.randn(a.m, a.k).bfloat16(); W = (torch.randn(a.k, a.n) / a.k ** 0.5).bfloat16()
            R = A.double() @ W.double()
            ta = ttnn.from_torch(A, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            ta3 = ttnn.reshape(ta, (a.batch, a.m // a.batch, a.k))
            tw = ttnn.from_torch(W, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            for fid in a.fid.split(","):
                ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid),
                                                            math_approx_mode=False, fp32_dest_acc_en=True,
                                                            packer_l1_acc=True)
                for v in a.variants.split(","):
                    pc = program(v, mt, kt, nt, grid)
                    if pc == "skip":
                        continue
                    kw = dict(compute_kernel_config=ck, dtype=ttnn.bfloat16)
                    if pc is None:
                        kw["core_grid"] = ttnn.CoreGrid(y=grid[1], x=grid[0])
                    else:
                        kw["program_config"] = pc
                    x = ta3 if v == "auto3d" else ta
                    run = lambda: ttnn.linear(x, tw, **kw)
                    try:
                        y = run()
                    except Exception as e:
                        res["cells"].append(dict(draw=d, fid=fid, v=v, error=str(e)[:200])); continue
                    Y = ttnn.to_torch(y).double().reshape(a.m, a.n); ttnn.deallocate(y)
                    nb, nbig, mx, rms, worst = wrong(R, Y)
                    ts = None
                    if d == 0:
                        ts = []
                        for _ in range(5):
                            ttnn.synchronize_device(dev); t0 = time.perf_counter(); y = run(); ttnn.synchronize_device(dev)
                            ts.append(time.perf_counter() - t0); ttnn.deallocate(y)
                        ts = round(statistics.median(ts) * 1e6, 1)
                    cell = dict(draw=d, fid=fid, v=v, wrong=nb, big=nbig, max_err=round(mx, 4), rms_err=round(rms, 6),
                                worst=worst, us=ts)
                    res["cells"].append(cell); print(json.dumps(cell), flush=True)
            ttnn.deallocate(ta); ttnn.deallocate(tw)
    finally:
        ttnn.close_device(dev)
        tot = {}
        for c in res["cells"]:
            if "wrong" in c:
                t = tot.setdefault(f"{c['fid']}/{c['v']}", dict(wrong=0, big=0, el=0, us=None))
                t["wrong"] += c["wrong"]; t["big"] += c["big"]; t["el"] += a.m * a.n
                t["us"] = c["us"] if c["us"] is not None else t["us"]
        res["totals"] = tot
        a.out.write_text(json.dumps(res, indent=1) + "\n")
        for k, t in tot.items():
            print(f"{k:16s} wrong {t['wrong']:4d} big {t['big']:4d} of {t['el']:.2e} el, {t['us']} us")


if __name__ == "__main__":
    main()
