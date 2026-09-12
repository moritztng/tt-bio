#!/usr/bin/env python3
"""Is the fold's largest cost concentrated in a few gating sites, or diffuse across the grid?

Reads percore_map.py's json and answers four questions the ops CSV cannot:

  1. WAIT HISTOGRAM. How is the math thread's input-tile wait distributed over physical cores?
     Flat means there is no hot core to fix. Skewed means the work split itself is lopsided.
  2. GATING. Per op, the core that finishes last gates every other core. Is it the same few cores
     every time (a fan-in) or a different core each op (diffuse)?
  3. SHAPE. In a daisy-chained operand forward, a core's finish time rises with its position along
     the chain axis. Spearman of per-core end-time against grid coordinate tests exactly that.
  4. BUDGET. Where do the block's core-milliseconds go: busy, waiting inside a kernel, idle
     between kernels.
"""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict


def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return 0.0
    return (2 * sum((i + 1) * x for i, x in enumerate(xs))) / (n * s) - (n + 1) / n


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mapjson")
    ap.add_argument("--opscsv", help="ops CSV, to name the ops by OP CODE")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = json.load(open(a.mapjson))
    blk = d["blocks"][len(d["blocks"]) // 2]          # median rep
    span = blk["span"]
    ncore = len(blk["per_core_wait"])
    US = 1e3                                           # cycles at 1 GHz -> ns; /1e3 -> us

    names = {}
    if a.opscsv:
        import csv
        with open(a.opscsv) as fh:
            for row in csv.DictReader(fh):
                try:
                    names[int(row["GLOBAL CALL COUNT"]) // 1024] = row["OP CODE"].strip()
                except (ValueError, KeyError):
                    pass

    R = {"span_ms": span / 1e6, "cores": ncore}

    # ---- 1. wait histogram over physical cores -------------------------------------------------
    w = blk["per_core_wait"]
    vals = sorted(w.values(), reverse=True)
    med = statistics.median(vals)
    tot = sum(vals)
    top10 = vals[:max(1, ncore // 10)]
    R["wait_hist"] = {
        "total_core_ms": tot / 1e6,
        "per_core_mean_ms": tot / ncore / 1e6,
        "max_ms": vals[0] / 1e6, "min_ms": vals[-1] / 1e6, "median_ms": med / 1e6,
        "max_over_median": vals[0] / med,
        "top_decile_share_pct": 100 * sum(top10) / tot,
        "flat_share_pct": 100 * len(top10) / ncore,
        "gini": gini(vals),
        "sorted_core_ms": [round(v / 1e6, 4) for v in vals],
        "by_core_ms": {k: round(v / 1e6, 4) for k, v in sorted(w.items(), key=lambda kv: -kv[1])},
    }

    # ---- 2. gating: who finishes last, and how often --------------------------------------------
    gate = Counter()
    surplus_tot = 0.0
    rows = []
    for o in blk["ops"]:
        gate[tuple(o["gate_core"])] += 1
        surplus_tot += o["straggler_surplus"]
        rows.append(o)
    rows_by_surplus = sorted(rows, key=lambda o: -o["straggler_surplus"])
    cum = 0.0
    n60 = 0
    for o in rows_by_surplus:
        cum += o["straggler_surplus"]
        n60 += 1
        if cum >= 0.60 * surplus_tot:
            break
    gate_counts = sorted(gate.values(), reverse=True)
    R["gating"] = {
        "straggler_surplus_core_ms": surplus_tot / 1e6,
        "surplus_pct_of_core_ms": 100 * surplus_tot / (span * ncore),
        "ops": len(rows),
        "ops_for_60pct_of_surplus": n60,
        "top40_ops_share_pct": 100 * sum(o["straggler_surplus"] for o in rows_by_surplus[:40]) / surplus_tot,
        "distinct_gate_cores": len(gate),
        "gate_uniform_expect": len(rows) / ncore,
        "gate_top_core_count": gate_counts[0],
        "gate_gini": gini(gate_counts),
        "gate_top10": [[f"{c[0]},{c[1]}", n] for c, n in gate.most_common(10)],
        "top_surplus_ops": [
            {"op": o["op"], "i": o["i"], "name": names.get(o["op"], "?"), "cores": o["cores"],
             "span_us": round(o["span"] / US, 2),
             "surplus_core_us": round(o["straggler_surplus"] / US, 1),
             "start_skew_us": round(o["first_to_last_start"] / US, 2),
             "gate_core": o["gate_core"]}
            for o in rows_by_surplus[:25]],
    }

    # by op name
    byname = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for o in rows:
        k = names.get(o["op"], "?")
        byname[k][0] += 1
        byname[k][1] += o["straggler_surplus"]
        byname[k][2] += o["wait_total"]
        byname[k][3] += o["span"]
    R["by_opcode"] = {k: {"n": v[0], "surplus_core_ms": round(v[1] / 1e6, 3),
                          "wait_core_ms": round(v[2] / 1e6, 3), "span_ms": round(v[3] / 1e6, 3)}
                      for k, v in sorted(byname.items(), key=lambda kv: -kv[1][1])}

    # ---- 3. shape: does finish time know where the core sits on the grid? ----------------------
    # rank coordinates so physical NOC coords become logical grid indices
    xs = sorted({int(k.split(",")[0]) for k in blk["per_core_wait"]})
    ys = sorted({int(k.split(",")[1]) for k in blk["per_core_wait"]})
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    shape = []
    for o in rows_by_surplus[:40]:
        pc = o["percore"]
        cx, cy, end, wt = [], [], [], []
        for k, v in pc.items():
            a_, b_ = k.split(",")
            cx.append(xi[int(a_)]); cy.append(yi[int(b_)])
            end.append(v[1]); wt.append(v[2])
        shape.append({
            "op": o["op"], "name": names.get(o["op"], "?"), "cores": o["cores"],
            "surplus_core_us": round(o["straggler_surplus"] / US, 1),
            "rho_end_x": round(spearman(cx, end), 3), "rho_end_y": round(spearman(cy, end), 3),
            "rho_wait_x": round(spearman(cx, wt), 3), "rho_wait_y": round(spearman(cy, wt), 3),
            "end_spread_us": round((max(end) - min(end)) / US, 2),
        })
    R["shape"] = shape
    strong = [s for s in shape if max(abs(s["rho_end_x"]), abs(s["rho_end_y"])) >= 0.5]
    R["shape_summary"] = {
        "ops_examined": len(shape),
        "ops_position_dependent_rho_ge_0.5": len(strong),
        "surplus_share_of_position_dependent_pct":
            100 * sum(s["surplus_core_us"] for s in strong) / max(1e-9, sum(s["surplus_core_us"] for s in shape)),
        "median_abs_rho": round(statistics.median(
            [max(abs(s["rho_end_x"]), abs(s["rho_end_y"])) for s in shape]), 3),
    }

    # ---- 4. budget ------------------------------------------------------------------------------
    busy = sum(blk["per_core_busy"].values())
    grid_core_ms = span * ncore / 1e6
    R["budget"] = {
        "grid_core_ms": grid_core_ms,
        "in_kernel_core_ms": busy / 1e6,
        "in_kernel_pct": 100 * busy / (span * ncore),
        "wait_front_core_ms": tot / 1e6,
        "reserve_back_core_ms": sum(blk["per_core_resv"].values()) / 1e6,
        "idle_between_kernels_core_ms": (span * ncore - busy) / 1e6,
        "idle_between_kernels_pct": 100 * (span * ncore - busy) / (span * ncore),
    }

    js = json.dumps(R, indent=1)
    if a.out:
        open(a.out, "w").write(js)
    print(json.dumps({k: R[k] for k in
                      ("span_ms", "cores", "budget", "shape_summary")}, indent=1))
    h = R["wait_hist"]
    print(f"\nWAIT HISTOGRAM  total {h['total_core_ms']:.1f} core-ms over {ncore} cores")
    print(f"  max {h['max_ms']:.3f}  median {h['median_ms']:.3f}  min {h['min_ms']:.3f}  "
          f"max/median {h['max_over_median']:.3f}  gini {h['gini']:.4f}  "
          f"top-decile share {h['top_decile_share_pct']:.2f}% (uniform = {h['flat_share_pct']:.2f}%)")
    g = R["gating"]
    print(f"\nGATING  straggler surplus {g['straggler_surplus_core_ms']:.1f} core-ms "
          f"= {g['surplus_pct_of_core_ms']:.2f}% of the grid's core-ms")
    print(f"  {g['ops_for_60pct_of_surplus']} of {g['ops']} ops carry 60% of it; "
          f"top 40 ops carry {g['top40_ops_share_pct']:.1f}%")
    print(f"  gate cores: {g['distinct_gate_cores']} distinct, top core gates "
          f"{g['gate_top_core_count']} ops (uniform = {g['gate_uniform_expect']:.1f}), "
          f"gini {g['gate_gini']:.4f}")
    print("  top gate cores:", g["gate_top10"][:6])
    print("\nTOP SURPLUS OPS")
    for o in g["top_surplus_ops"][:12]:
        print(f"  #{o['i']:3d} {o['name'][:34]:34s} cores {o['cores']:3d} span {o['span_us']:8.1f} us "
              f"surplus {o['surplus_core_us']:9.1f} core-us  skew {o['start_skew_us']:7.2f}  gate {o['gate_core']}")
    print("\nBY OP CODE (surplus)")
    for k, v in list(R["by_opcode"].items())[:8]:
        print(f"  {k[:38]:38s} n={v['n']:3d} surplus {v['surplus_core_ms']:8.2f}  "
              f"wait {v['wait_core_ms']:9.2f}  span {v['span_ms']:7.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
