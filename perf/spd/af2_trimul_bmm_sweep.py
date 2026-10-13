#!/usr/bin/env python3
"""Program-config sweep for the triangle multiplication's batched matmul at AF2 sizes, on Wormhole.

The AF2-IG device profile (.114 out/af2igwh_devprof_8e053986e) puts the two `_multiply` matmuls at
~390 us a call, 416 calls a fold, at 0.20 of their roofline. At Kt = 7 (195 and 224 tokens) the
production 2D-mcast config (`_triangle_mul_program_config`) gives each core a 1x1 tile output block
of every channel, so 49 of 72 cores loop over all C channels with a multicast per channel. The
arms here put whole channels on cores instead (1D reuse config, batch folded into M), or let ttnn
plan with a core grid.

Operands [1,C,S,S] @ [1,C,S,S] bf16 in DRAM, AF2's kernel config (HiFi4, fp32 dest acc, packer
L1 acc). Each arm: warm 2, median of 5 reps of `--inner` calls between syncs. `exact` is torch.equal
against the production config; every arm carries rel L2 against float64.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import ttnn  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402

WARM, REPS = 2, 5


def _plan(St, per_channel):
    prev = tt.set_trimul_per_channel(per_channel)
    try:
        return tt._triangle_mul_program_config(St, tt._trimul_ibw_full(), tt._trimul_subblock())
    finally:
        tt.set_trimul_per_channel(prev)


def _mcast_config(St):
    """The 2D multicast plan, the reference every arm is compared against."""
    return _plan(St, False)


def _per_channel_config(St):
    """The plan `TT_BIO_TRIMUL_PER_CHANNEL` actually ships, for this many sequence tiles.

    `prod` is the same call with the lever off, so `exact_vs_prod` on this arm is the bit-exactness
    claim the lever rests on, at whatever shape the sweep is pointed at.
    """
    return _plan(St, True)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--sizes", default="195,224")
    ap.add_argument("--channels", default="64,128")
    ap.add_argument("--mem", default="dram", choices=("dram", "l1"),
                    help="operand/output memory; production is L1 untaped at <= _trimul_l1_max_seq, DRAM taped")
    ap.add_argument("--transpose", default="none", choices=("none", "a", "b"),
                    help="the deferred transpose production hands the matmul (ending: a, starting: b)")
    ap.add_argument("--no-reuse2", action="store_true", help="skip reuse_c2 (hung on .114 chip 12, 10-12)")
    a = ap.parse_args()
    dev = tt.get_device()
    gx, gy = tt.COMPUTE_GRID_MAIN
    ckc_cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
               else ttnn.types.BlackholeComputeKernelConfig)
    ckc = ckc_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                  fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"arch": str(dev.arch()), "grid": [gx, gy], "loadavg": open("/proc/loadavg").read().split()[:3],
           "inner": a.inner, "cells": []}
    mc = ttnn.L1_MEMORY_CONFIG if a.mem == "l1" else ttnn.DRAM_MEMORY_CONFIG
    res.update(mem=a.mem, transpose=a.transpose)
    tkw = {"transpose_a": a.transpose == "a", "transpose_b": a.transpose == "b"}
    up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,  # noqa: E731
                                   memory_config=mc)
    torch.manual_seed(0)
    for S in map(int, a.sizes.split(",")):
        for C in map(int, a.channels.split(",")):
            at = torch.randn(1, C, S, S, dtype=torch.bfloat16)
            bt = torch.randn(1, C, S, S, dtype=torch.bfloat16)
            ref64 = ((at.double().transpose(-1, -2) if tkw["transpose_a"] else at.double())
                     @ (bt.double().transpose(-1, -2) if tkw["transpose_b"] else bt.double()))
            x, y = up(at), up(bt)
            St = (S + 31) // 32
            mm = lambda **kw: (lambda: ttnn.matmul(x, y, memory_config=mc,  # noqa: E731
                                                   dtype=ttnn.bfloat16, compute_kernel_config=ckc,
                                                   **tkw, **kw))
            arms = [("prod", mm(program_config=_mcast_config(St))),
                    ("perchan", mm(program_config=_per_channel_config(St))),
                    ("plain", mm()),
                    (f"grid{gx}x{gy}", mm(core_grid=ttnn.CoreGrid(y=gy, x=gx))),
                    ("grid8x8", mm(core_grid=ttnn.CoreGrid(y=8, x=8)))]
            # 1D reuse: batch folded into M, whole channels per core (per_core_M a multiple of St).
            for chans in ((1,) if a.no_reuse2 else (1, 2)):
                for kw in (1, St):
                    if St % kw:
                        continue
                    arms.append((f"reuse_c{chans}_k{kw}", mm(program_config=ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=(gx, gy), in0_block_w=kw, out_subblock_h=1,
                        out_subblock_w=1, per_core_M=chans * St, per_core_N=St))))
            cell = {"S": S, "C": C, "St": St, "arms": []}
            ref = None
            for name, fn in arms:
                row = {"arm": name}
                try:
                    s, o = timed(dev, fn, a.inner)
                    got = ttnn.to_torch(o); ttnn.deallocate(o)
                    if ref is None:
                        ref = got
                    d = got.double() - ref64
                    row.update(ms=1e3 * s, exact_vs_prod=bool(torch.equal(got, ref)),
                               rel_l2_f64=float(d.norm() / ref64.norm()))
                except Exception as e:                                          # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                cell["arms"].append(row)
                print(S, C, row, flush=True)
            base = cell["arms"][0].get("ms")
            for r in cell["arms"]:
                if base and "ms" in r:
                    r["speedup"] = base / r["ms"]
            res["cells"].append(cell)
            ttnn.deallocate(x); ttnn.deallocate(y)
            Path(a.out).write_text(json.dumps(res, indent=1))
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
