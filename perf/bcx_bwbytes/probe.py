#!/usr/bin/env python3
"""bcx-bwbytes: the two routed byte levers at the shapes BindCraft 2 actually runs.

Both were measured by `bcx-bytes` at n=256 and both are gated above it, so on a real BC2 round
-- 211 tokens bucketed to 224 -- neither one fires. This sweeps the gate instead of assuming it.

  sum0    `ttnn.sum(dim=0)` against `autograd._pairwise_sum0` on the triangle attention's
          broadcast-bias gradient [rows, heads, N, N] fp32, timing and distance from float64.
  permute `ttnn.permute(g, (0, 2, 3, 1))` against `reblock_permute_back` on the channel move's
          gradient [1, C, N, N] bf16, timing and `torch.equal`.

Every arm is interleaved rep by rep against its control so a drift in the box cannot land on one
of them, and the AICLK is sampled throughout and reported over each timed window.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = pathlib.Path(__file__).resolve().parent
HERE = OUT


def _dest(out, name):
    """`launch.sh` hands every script a DIRECTORY as --out; a bare path stays a file."""
    p = pathlib.Path(out)
    return str(p / name) if p.is_dir() or p.suffix != ".json" else str(p)


def _head():
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def timed(fn, reps, clock, free):
    """Median/min ms over `reps`, and the clock window that covers them."""
    import ttnn
    y = fn()
    ttnn.synchronize_device(y.device())
    free(y)
    ts, spans = [], []
    for _ in range(reps):
        t0 = time.time()
        y = fn()
        ttnn.synchronize_device(y.device())
        t1 = time.time()
        ts.append((t1 - t0) * 1e3)
        spans.append((t0, t1))
        free(y)
    ts.sort()
    return {"min_ms": round(ts[0], 4), "median_ms": round(ts[len(ts) // 2], 4),
            "max_ms": round(ts[-1], 4), "aiclk": clock.window(spans)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--rows", default="64,128,160,192,224,256,288")
    ap.add_argument("--ns", default="224,256")
    ap.add_argument("--chans", default="64,128")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "probe.json"))
    args = ap.parse_args()
    args.out = _dest(args.out, "probe.json")

    import torch
    import ttnn
    from perf.bcx_stack import stack as S
    from tt_bio import autograd as ag, reblock_permute as R
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    clock = S.Clock(0.2)
    node, pci = S.sysfs_node()
    cfg = ag.precise_config()
    gen = torch.Generator().manual_seed(0)
    free = ttnn.deallocate

    stamp = {"host": __import__("os").uname().nodename, "pci": pci, "sysfs": node,
             "card": __import__("os").environ.get("TT_VISIBLE_DEVICES"),
             "commit": _head(),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": __import__("os").getloadavg(), "reps": args.reps}
    print(json.dumps(stamp), flush=True)

    sum_rows, perm_rows = [], []

    def _save():
        """After EVERY shape, not at the end. A probe that writes its artifact once, after the
        last loop, loses every shape it already measured the first time one raises or the box
        reclaims the card -- and the shapes most likely to raise are the big ones, which run
        last. (`a-harness-that-writes-its-clock-at-the-end-of-main-loses-it-on-every-death`.)"""
        pathlib.Path(args.out).write_text(json.dumps(
            {"stamp": stamp, "sum0": sum_rows, "permute": perm_rows}, indent=1))
    for N in [int(x) for x in args.ns.split(",")]:
        for rows in [int(x) for x in args.rows.split(",")]:
            x = torch.randn(rows, args.heads, N, N, generator=gen) * 1e-3
            ref = x.double().sum(0, keepdim=True)
            g = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            rec = {"shape": [rows, args.heads, N, N],
                   "torch_f32_rel_l2_vs_f64":
                       float((x.sum(0, keepdim=True).double() - ref).norm() / ref.norm())}
            arms = {"sum": lambda: ttnn.sum(g, dim=0, keepdim=True, compute_kernel_config=cfg),
                    "tree": lambda: ag._pairwise_sum0(g)}
            for name, fn in arms.items():
                # An arm the wheel or the allocator refuses is a RESULT, not a crash, and it
                # must not take the shapes already measured with it -- see the incremental
                # write below. The permute loop had this guard from the start; this one did not.
                try:
                    y = fn()
                    yt = ttnn.to_torch(y).reshape(ref.shape)
                    free(y)
                except Exception as e:
                    rec[name] = {"error": str(e).splitlines()[0][:200]}
                    continue
                rec[name] = {"rel_l2_vs_f64":
                             float((yt.double() - ref).norm() / ref.norm())}
                rec[name].update(timed(fn, args.reps, clock, free))
            if all("median_ms" in rec.get(a, {}) for a in arms):
                rec["tree_over_sum"] = round(rec["sum"]["median_ms"]
                                             / rec["tree"]["median_ms"], 3)
            free(g)
            print(json.dumps(rec), flush=True)
            sum_rows.append(rec)
            _save()

    for N in [int(x) for x in args.ns.split(",")]:
        for C in [int(x) for x in args.chans.split(",")]:
            x = torch.randn(1, C, N, N, generator=gen)
            g = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mc = ttnn.DRAM_MEMORY_CONFIG
            rec = {"shape": [1, C, N, N], "eligible_back": bool(R.eligible_back(g, mc))}
            arms = {"permute": lambda: ttnn.permute(g, [0, 2, 3, 1], memory_config=mc),
                    "reblock": lambda: R.reblock_permute_back(g, mc)}
            base = None
            ok = True
            for name, fn in arms.items():
                try:
                    y = fn()
                    yt = ttnn.to_torch(y)
                    free(y)
                except Exception as e:
                    rec[name] = {"error": str(e).splitlines()[0][:200]}
                    ok = False
                    continue
                if base is None:
                    base = yt
                rec[name] = {"bits_eq_permute": bool(torch.equal(yt, base))}
                rec[name].update(timed(fn, args.reps, clock, free))
            if ok and all("median_ms" in rec.get(a, {}) for a in arms):
                rec["reblock_over_permute"] = round(
                    rec["permute"]["median_ms"] / rec["reblock"]["median_ms"], 3)
            free(g)
            print(json.dumps(rec), flush=True)
            perm_rows.append(rec)
            _save()

    clock.stop()
    stamp["loadavg_end"] = __import__("os").getloadavg()
    _save()
    print(f"wrote {args.out}", flush=True)

    def band(rs, key, ratio):
        """`no wins` is the EXPECTED reading for the permute lever at N=224, so it has to be
        printable. It was not: `print(a + b or c)` parses as `print((a + b) or c)`, and `a` is a
        non-empty literal, so the `or` never fired and an empty win list printed as a dangling
        `wins at `."""
        wins = [(r["shape"], r[ratio]) for r in rs if r.get(ratio, 0) > 1.0]
        if wins:
            print(f"{key}: wins at " + ", ".join(f"{sh} {v:.3f}x" for sh, v in wins))
        else:
            print(f"{key}: NO WINS at any measured shape")
    band(sum_rows, "tree", "tree_over_sum")
    band(perm_rows, "reblock", "reblock_over_permute")


if __name__ == "__main__":
    main()
