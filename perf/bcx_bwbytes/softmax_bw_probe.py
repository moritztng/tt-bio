#!/usr/bin/env python3
"""bcx-bwbytes: the softmax backward on the score tensor, four ways, graded against float64.

`reach.py` puts 25.7 % of an AF2 Evoformer block backward -- 10.122 GB of 39.394 -- in thirty
calls, and they are all this one expression on `[n, 4, n, n]`. This prices the ways out of it.

    chain      what ships: inner = sum(g*y)/sum(y), dx = y*(g - inner). Ten passes of the score
               tensor with SOFTMAX_BW_RENORM on, six with it off
    chain_bf16 the same expression on bf16 operands with `precise_config()` reductions, so the
               summation precision comes from fp32 accumulation in DST instead of from fp32
               tensors in DRAM. Halves every pass. A PRECISION lever
    moreh      `ttnn.moreh_softmax_backward`, in the 0.68.0 wheel, no build. Three passes -- and
               no renorm, which is default-on since 2026-09-21 on Moritz's ask-9629 ruling.
               THE fp32 ARMS ARE EXPECTED TO BE REFUSED, not to be slow: the op's guard admits
               only BFLOAT16 and BFLOAT8_B (device_operation.cpp:79-84) and `of3t-softbw` already
               returned NO-GO on Route A for exactly that. They are kept so the refusal is
               reproduced here rather than assumed, and a refusal is recorded as an error string,
               never as a timing
    moreh_rn   the renorm kept through an identity: with S = sum(y, -1),
               S * moreh(y/S, g) == y*(g - sum(g*y)/S). Eight passes

Every arm is graded against a float64 reference built from the SAME operands, never against
another device arm. `moreh_layer_norm_backward` is wrong on Blackhole (dx 2.741e+06 rel L2 bf16,
upstream #12349), so a moreh arm that is merely fast has proved nothing here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = pathlib.Path(__file__).resolve().parent


def _head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _dest(out, name):
    """`launch.sh` hands every script a DIRECTORY as --out; a bare path stays a file."""
    p = pathlib.Path(out)
    return str(p / name) if p.is_dir() or p.suffix != ".json" else str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="224,256")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default=str(HERE / "softmax_bw_probe.json"))
    args = ap.parse_args()
    args.out = _dest(args.out, "softmax_bw_probe.json")

    import os
    import torch
    import ttnn
    from perf.bcx_stack import stack as S
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    clock = S.Clock(0.2)
    node, pci = S.sysfs_node()
    cfg = ag.precise_config()
    gen = torch.Generator().manual_seed(0)

    stamp = {"host": os.uname().nodename, "pci": pci, "sysfs": node, "commit": _head(),
             "card": os.environ.get("TT_VISIBLE_DEVICES"), "reps": args.reps,
             "renorm_flag": bool(ag.SOFTMAX_BW_RENORM),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg()}
    print(json.dumps(stamp), flush=True)

    rows = []
    for n in [int(x) for x in args.ns.split(",")]:
        sc = (torch.randn(n, args.heads, n, n, generator=gen) * 3.0)
        gt = torch.randn(n, args.heads, n, n, generator=gen) * 1e-2
        y64 = torch.softmax(sc.double(), dim=-1)
        g64 = gt.double()
        # The reference is the RENORMED expression, because that is what ships and what the
        # gradient of this model means. An arm without the renorm is graded against it too, which
        # is the point: the distance it reads IS the cost of dropping the correction.
        inner64 = (g64 * y64).sum(-1, keepdim=True) / y64.sum(-1, keepdim=True)
        dx64 = y64 * (g64 - inner64)

        y32 = ttnn.from_torch(y64.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        g32 = ttnn.from_torch(gt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        y16 = ttnn.typecast(y32, ttnn.bfloat16)
        g16 = ttnn.typecast(g32, ttnn.bfloat16)

        def chain(y, g, precise_numerator=False):
            """`softmax_bw_inner` + `y*(g - inner)`, written out rather than called.

            Written out for one reason. `autograd.py:119` sums the NUMERATOR `g*y` with no
            `compute_kernel_config`, while line 123 sums the denominator `y` at
            `precise_config()`. On fp32 operands that asymmetry is invisible. On bf16 operands it
            is the whole question, because the point of the bf16 arm is that the summation
            precision comes from fp32 accumulation in DST -- and an unconfigured reduce does not
            provide it. So the arm measures both and does not presume which one ships.
            """
            num = ttnn.sum(ttnn.multiply(g, y), dim=-1, keepdim=True,
                           **({"compute_kernel_config": cfg} if precise_numerator else {}))
            inner = (ttnn.divide(num, ttnn.sum(y, dim=-1, keepdim=True,
                                               compute_kernel_config=cfg))
                     if ag.SOFTMAX_BW_RENORM else num)
            out = ttnn.multiply(y, ttnn.subtract(g, inner))
            ttnn.deallocate(inner)
            return out

        def moreh(y, g):
            return ttnn.moreh_softmax_backward(y, g, dim=len(y.shape) - 1)

        def moreh_rn(y, g):
            s = ttnn.sum(y, dim=-1, keepdim=True, compute_kernel_config=cfg)
            yn = ttnn.divide(y, s)
            d = ttnn.moreh_softmax_backward(yn, g, dim=len(y.shape) - 1)
            ttnn.deallocate(yn)
            out = ttnn.multiply(d, s)
            ttnn.deallocate(d)
            ttnn.deallocate(s)
            return out

        arms = {"chain": lambda: chain(y32, g32),
                "chain_precise": lambda: chain(y32, g32, True),
                "chain_bf16": lambda: chain(y16, g16),
                "chain_bf16_precise": lambda: chain(y16, g16, True),
                "moreh": lambda: moreh(y32, g32),
                "moreh_bf16": lambda: moreh(y16, g16),
                "moreh_rn": lambda: moreh_rn(y32, g32),
                "moreh_rn_bf16": lambda: moreh_rn(y16, g16)}

        rec = {"shape": [n, args.heads, n, n]}
        for name, fn in arms.items():
            try:
                out = fn()
                ot = ttnn.to_torch(out)
                ttnn.deallocate(out)
            except Exception as e:
                rec[name] = {"error": str(e).splitlines()[0][:220]}
                print(json.dumps({name: rec[name]}), flush=True)
                continue
            rel = float((ot.double() - dx64).norm() / dx64.norm())
            ts, spans = [], []
            for _ in range(args.reps):
                t0 = time.time()
                out = fn()
                ttnn.synchronize_device(dev)
                t1 = time.time()
                ts.append((t1 - t0) * 1e3)
                spans.append((t0, t1))
                ttnn.deallocate(out)
            ts.sort()
            rec[name] = {"rel_l2_vs_f64": rel, "min_ms": round(ts[0], 4),
                         "median_ms": round(ts[len(ts) // 2], 4), "max_ms": round(ts[-1], 4),
                         "aiclk": clock.window(spans)}
            print(json.dumps({"n": n, name: rec[name]}), flush=True)
        base = rec.get("chain", {}).get("median_ms")
        if base:
            for name in arms:
                if "median_ms" in rec.get(name, {}):
                    rec[name]["speedup_vs_chain"] = round(base / rec[name]["median_ms"], 3)
        rows.append(rec)
        for t in (y32, g32, y16, g16):
            ttnn.deallocate(t)

    clock.stop()
    stamp["loadavg_end"] = os.getloadavg()
    pathlib.Path(args.out).write_text(json.dumps({"stamp": stamp, "rows": rows}, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
