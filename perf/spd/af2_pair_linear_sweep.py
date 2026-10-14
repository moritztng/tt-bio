#!/usr/bin/env python3
"""Block sweep for AF2's pair-track linears on `minimal_matmul`, against the shipped `ttnn.linear`.

The AF2-IG Wormhole device profile (.114 out/af2igwh_devprof_8e053986e) has the pair transition's
two linears at 1139 us (128 -> 512, bias + ReLU) and 850 us (512 -> 128, bias) a call, roof 0.20
and 0.27, 208 calls each a fold: both move ~56 MB, a ~225 us floor at Wormhole's 249 GB/s.
`pair_mm` already serves these on Blackhole inside a BindCraft 2 round with a table swept at 288
tokens; it was never timed on Wormhole, and it declines AF2-IG's ragged 195 x 195 grid
(rows % 32). This times the shipped call, the `pair_mm` table, and a grid of minimal_matmul blocks
on [1, S, S, K] x [K, N] (+ bias, + ReLU as production), every arm with rel L2 against float64.

Each arm: warm 2, median of 5 reps of `--inner` calls between syncs.

`--ckcs` adds the shipped call and the `pair_mm` table again under other fidelity/accumulator
configs (`hifi2-bf16` = HiFi2, bf16 dest). At K = 128 a tile row is four K tiles, so the fp32 dest
buys little accuracy and halves the dest's capacity; this prices that against float64.
"""
import argparse, itertools, json, statistics, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402
from tt_bio import pair_mm  # noqa: E402

WARM, REPS = 2, 5
SHAPES = [(128, 512, "relu"), (512, 128, None), (128, 128, None), (128, 256, None)]


def timed(dev, fn, inner):
    for _ in range(WARM):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for _ in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(inner):
            o = fn(); ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / inner)
    o = fn(); ttnn.synchronize_device(dev)
    return statistics.median(ts), o


def divisors(n, cap):
    return [d for d in range(1, min(n, cap) + 1) if n % d == 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--sizes", default="224,195")
    ap.add_argument("--max-arms", type=int, default=48)
    ap.add_argument("--ckcs", default="", help="extra configs, e.g. hifi4-bf16,hifi2-fp32,hifi2-bf16")
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    ckc_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
    def config(name):
        fid, acc = name.split("-")
        return ckc_cls(math_fidelity=getattr(ttnn.MathFidelity, {"lofi": "LoFi"}.get(fid, fid.replace("hifi", "HiFi"))),
                       math_approx_mode=False, fp32_dest_acc_en=acc == "fp32", packer_l1_acc=True)
    ckc = config("hifi4-fp32")
    extra = [(n, config(n)) for n in a.ckcs.split(",") if n]
    res = {"arch": str(dev.arch()), "grid": [gx, gy], "loadavg": open("/proc/loadavg").read().split()[:3],
           "inner": a.inner, "cells": []}
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pair_mm.PAIR_MM_FUSED = True
    torch.manual_seed(0)
    for S in map(int, a.sizes.split(",")):
        for K, N, act in SHAPES:
            xt = torch.randn(1, S, S, K, dtype=torch.bfloat16)
            wt = (torch.randn(K, N, dtype=torch.bfloat16) / K ** 0.5).to(torch.bfloat16)
            bt = torch.randn(N, dtype=torch.bfloat16) * 0.1
            ref64 = xt.double() @ wt.double() + bt.double()
            if act == "relu":
                ref64 = ref64.clamp_min(0)
            x, w, b = up(xt), up(wt), up(bt.reshape(1, N))
            arms = [("prod", lambda: ttnn.linear(x, w, bias=b, activation=act, compute_kernel_config=ckc,
                                                 dtype=ttnn.bfloat16, core_grid=tt.CORE_GRID_MAIN)),
                    ("pair_mm_table", lambda: pair_mm.matmul(x, w, b, ckc, ttnn.bfloat16, activation=act))]
            for n, c in extra:
                arms += [(f"prod@{n}", lambda c=c: ttnn.linear(x, w, bias=b, activation=act, compute_kernel_config=c,
                                                                dtype=ttnn.bfloat16, core_grid=tt.CORE_GRID_MAIN)),
                         (f"pair_mm_table@{n}", lambda c=c: pair_mm.matmul(x, w, b, c, ttnn.bfloat16, activation=act))]
            Mt = S * ((S + 31) // 32)
            kt, nt = K // 32, N // 32
            grid = []
            for Mb, Kb, Nb in itertools.product((1, 2, 4, 7, 8, 16), divisors(kt, 8), divisors(nt, 8)):
                for sh, sw in ((1, 1), (2, 1), (1, 2), (4, 1), (2, 2), (1, 4)):
                    if Mb % sh or Nb % sw or (Mb, Nb) == (1, 1) and (sh, sw) != (1, 1):
                        continue
                    grid.append((Mb, Kb, Nb, sh, sw))
            # Keep the widest subblock per block and the largest K block first: the sweep's prior.
            grid.sort(key=lambda c: (-(c[3] * c[4]), -c[1], abs(c[0] * c[2] - 16)))
            seen, kept = set(), []
            for c in grid:
                if c[:3] in seen:
                    continue
                seen.add(c[:3]); kept.append(c)
            for Mb, Kb, Nb, sh, sw in kept[:a.max_arms]:
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=Mb, K_block_size=Kb, N_block_size=Nb, subblock_h=sh, subblock_w=sw,
                    compute_with_storage_grid_size=tt._mm_core_coord(gx, gy))
                arms.append((f"mm{Mb}_{Kb}_{Nb}_{sh}x{sw}", lambda cfg=cfg: ttnn.experimental.minimal_matmul(
                    input_tensor=x, weight_tensor=w, bias_tensor=b, compute_kernel_config=ckc,
                    fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if act else None,
                    dtype=ttnn.bfloat16, config=cfg)))
            cell = {"S": S, "K": K, "N": N, "act": act, "Mt": Mt, "arms": []}
            res["cells"].append(cell)      # written after every arm: a hung config keeps the rest
            for name, fn in arms:
                row = {"arm": name}
                try:
                    if fn() is None:
                        row["error"] = "declined"
                    else:
                        s, o = timed(dev, fn, a.inner)
                        got = ttnn.to_torch(o); ttnn.deallocate(o)
                        got = got.reshape(ref64.shape) if got.numel() == ref64.numel() else got
                        d = got.double() - ref64
                        row.update(ms=1e3 * s, rel_l2_f64=float(d.norm() / ref64.norm()))
                except Exception as e:                                          # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                cell["arms"].append(row)
                print(S, K, N, row, flush=True)
                Path(a.out).write_text(json.dumps(res, indent=1))
            base = cell["arms"][0].get("ms")
            for r in cell["arms"]:
                if base and "ms" in r:
                    r["speedup"] = base / r["ms"]
            ok = [r for r in cell["arms"] if "ms" in r]
            cell["best"] = min(ok, key=lambda r: r["ms"])["arm"] if ok else None
            for t in (x, w, b):
                ttnn.deallocate(t)
            Path(a.out).write_text(json.dumps(res, indent=1))
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
