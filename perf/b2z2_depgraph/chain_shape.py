#!/usr/bin/env python3
"""Is the operand daisy chain fill-limited or throughput-limited? The per-core shape says which.

`b2z2-tile-arrival-latency` measured that 67.6 % of the reader's time in tt-bio's matmul kernels is
one core waiting for the core upstream of it. It measured the TOTAL. This reads the same log per
core, which separates two mechanisms that the total cannot tell apart:

  * fill-limited -- wait RISES with the core's index along the chain, because core i pays i hops of
    latency once. Cost is one chain fill per output block and a deeper CB pipeline hides it.
  * throughput-limited -- wait is FLAT across the chain, because every core is waiting on a hop that
    is itself busy forwarding. Cost is the per-hop bandwidth and only a multicast removes it.

Auto-detects the replayed window: the census brackets each replay with 1-core fence ops.
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict

ZONES = ("B2Z2-IN0-CHAINWAIT", "B2Z2-IN1-CHAINWAIT", "B2Z2-IN0-DOWNWAIT", "B2Z2-IN1-DOWNWAIT",
         "B2Z2-IN0-FWD", "B2Z2-IN1-FWD", "B2Z2-IN0-CBRES", "B2Z2-IN1-CBRES",
         "B2Z2-IN0-SRC", "B2Z2-IN1-SRC", "B2Z2-RDBAR",
         "CB-COMPUTE-WAIT-FRONT", "CB-COMPUTE-RESERVE-BACK")


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--ops-per-block", type=int, default=272)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # (op, core) -> {zone: cycles}; plus the cores each op ran on, to find the fences
    acc = defaultdict(lambda: defaultdict(float))
    op_cores = defaultdict(set)
    want = set(ZONES)
    with open(a.log) as fh:
        fh.readline(), fh.readline()
        for line in fh:
            f = line.split(",", 12)
            if len(f) < 12:
                continue
            zone = f[10]
            if zone in want:
                try:
                    op = int(f[7]) // 1024
                except ValueError:
                    continue
                if int(f[5]) == 0:
                    continue
                acc[(op, (int(f[1]), int(f[2])))][zone] += float(f[6])
            elif zone == "BRISC-KERNEL" and f[11] == "ZONE_START":
                try:
                    op_cores[int(f[7]) // 1024].add((int(f[1]), int(f[2])))
                except ValueError:
                    pass

    ops = sorted(op_cores)
    fences = [o for o in ops if len(op_cores[o]) == 1]
    # the replayed region sits between the last two fence runs
    tail = [o for o in fences if o > (fences[0] if fences else 0)]
    lo = max((o for o in fences if o < ops[-1] - a.ops_per_block), default=ops[0]) + 1
    hi = min((o for o in fences if o >= lo + a.ops_per_block), default=ops[-1]) - 1
    window = [o for o in ops if lo <= o <= hi]
    nblocks = len(window) // a.ops_per_block
    if nblocks < 1:
        print(f"window {lo}..{hi} = {len(window)} ops, no whole block", file=sys.stderr)
        return 1
    bi = nblocks // 2
    block = window[bi * a.ops_per_block:(bi + 1) * a.ops_per_block]
    bset = set(block)

    per_core = defaultdict(lambda: defaultdict(float))
    for (op, core), z in acc.items():
        if op in bset:
            for k, v in z.items():
                per_core[core][k] += v

    cores = sorted(per_core)
    xs = sorted({c[0] for c in cores})
    ys = sorted({c[1] for c in cores})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    US = 1e3

    R = {"log": a.log, "window": [lo, hi], "blocks_in_window": nblocks, "block_index": bi,
         "ops_in_block": len(block), "cores": len(cores), "grid": [len(xs), len(ys)], "zones": {}}
    for z in ZONES:
        vals = [per_core[c].get(z, 0.0) for c in cores]
        tot = sum(vals)
        if tot == 0:
            continue
        nz = [v for v in vals if v > 0]
        med = statistics.median(nz)
        cx = [xi[c[0]] for c in cores]
        cy = [yi[c[1]] for c in cores]
        # position correlation only over the cores the zone actually fires on
        idx = [i for i, v in enumerate(vals) if v > 0]
        R["zones"][z] = {
            "total_core_us": round(tot / US, 1),
            "cores_firing": len(nz),
            "per_core_us_max": round(max(nz) / US, 2),
            "per_core_us_med": round(med / US, 2),
            "per_core_us_min": round(min(nz) / US, 2),
            "max_over_median": round(max(nz) / med, 3),
            "rho_x": round(spearman([cx[i] for i in idx], [vals[i] for i in idx]), 3),
            "rho_y": round(spearman([cy[i] for i in idx], [vals[i] for i in idx]), 3),
            "by_core_us": {f"{c[0]},{c[1]}": round(per_core[c].get(z, 0.0) / US, 2) for c in cores},
        }
    open(a.out, "w").write(json.dumps(R, indent=1))

    print(f"window {lo}..{hi}  blocks {nblocks}  block#{bi} ops {len(block)}  "
          f"cores {len(cores)}  grid {len(xs)}x{len(ys)}")
    print(f"{'zone':26s} {'core-us':>10s} {'cores':>6s} {'max':>9s} {'med':>9s} {'min':>9s} "
          f"{'max/med':>8s} {'rho_x':>7s} {'rho_y':>7s}")
    for z, v in R["zones"].items():
        print(f"{z:26s} {v['total_core_us']:10.1f} {v['cores_firing']:6d} "
              f"{v['per_core_us_max']:9.2f} {v['per_core_us_med']:9.2f} {v['per_core_us_min']:9.2f} "
              f"{v['max_over_median']:8.3f} {v['rho_x']:7.3f} {v['rho_y']:7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
