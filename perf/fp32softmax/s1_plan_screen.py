"""Host-only screen for the fp32-softmax height shard: what S1 changes, and where.

S1 lets `_fp32_softmax_l1_plan` pick the shard core count when the tuned 8x8 rectangle cannot
divide any affordable block. The plan is a pure function of `(n_heads, S)` and the byte budget, so
the whole blast radius is enumerable without a device: every model whose triangle attention has the
same head count gets the same plan at the same padded length.

Three things this asserts, per grid, so a regression shows up as a failed row and not as a slow
fold:

1. wherever the tuned rectangle serves a block today, the plan is byte-for-byte the tuned answer
   (same rows, 64 cores), i.e. S1 cannot move a size that is L1-resident today;
2. wherever it does not, the plan is legal: the block height divides the core count exactly, every
   core stays under the byte budget, and the count fits the ACTIVE grid;
3. no plan asks for more cores than the grid has, on a 13x10 or 11x10 Blackhole and on a
   Wormhole 8x8.

Run: PYTHONPATH=<worktree> python3 perf/fp32softmax/s1_plan_screen.py [--json out.json]
"""
import argparse
import json

import tt_bio.tenstorrent as tt

GRIDS = [(13, 10), (11, 10), (8, 8)]
HEADS = [1, 2, 4, 8, 16]
SIZES = [s for s in range(32, 1057, 32)] + [515, 546, 547, 1023]


def screen():
    out = {"budget_per_core": tt._FP32_SOFTMAX_L1_BYTES_PER_CORE,
           "tuned_cores": tt._FP32_SOFTMAX_L1_GRID[0] * tt._FP32_SOFTMAX_L1_GRID[1],
           "core_cap": tt._FP32_SOFTMAX_L1_CORE_CAP, "grids": {}}
    failures = []
    for gx, gy in GRIDS:
        tt.COMPUTE_GRID_MAIN = (gx, gy)
        tt._fp32_softmax_l1_plan.cache_clear()
        grid_cores = gx * gy
        budget = tt._fp32_softmax_core_budget()
        rows_out = []
        for heads in HEADS:
            for S in SIZES:
                hpr = heads * S
                per_row = hpr * S * 4
                tuned = tt._fp32_softmax_l1_rows(per_row, hpr)
                blk, cores = tt._fp32_softmax_l1_plan(per_row, hpr, S)
                row = {"heads": heads, "S": S, "tuned_rows": tuned,
                       "plan_rows": blk, "plan_cores": cores}
                if S % 32:
                    # `_fp32_softmax_shard` refuses a width that is not whole tiles, so a plan
                    # here would cap the block with no shard behind it: measured 0.786x-0.928x.
                    if blk:
                        failures.append(("planned a block for a width no shard can take", row))
                    if tuned:
                        failures.append(("the tuned rectangle serves a ragged width", row))
                elif tuned:
                    if (blk, cores) != (tuned, 64):
                        failures.append(("moved a size the rectangle already serves", row))
                elif blk:
                    height = blk * hpr
                    if cores > grid_cores:
                        failures.append(("plan wants more cores than the grid has", row))
                    if height % (cores * 32):
                        failures.append(("block height does not divide the core count", row))
                    if blk * per_row > cores * tt._FP32_SOFTMAX_L1_BYTES_PER_CORE:
                        failures.append(("per-core bytes over budget", row))
                    row["bytes_per_core"] = blk * per_row // cores
                    row["dark_today"] = True
                rows_out.append(row)
        out["grids"]["%dx%d" % (gx, gy)] = {"cores": grid_cores, "core_budget": budget,
                                           "rows": rows_out}
    out["failures"] = [{"why": w, "row": r} for w, r in failures]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="perf/fp32softmax/s1_plan_screen.json")
    a = ap.parse_args()
    out = screen()
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    for g, d in out["grids"].items():
        lit = [r for r in d["rows"] if r["tuned_rows"]]
        dark = [r for r in d["rows"] if not r["tuned_rows"] and r["plan_rows"]]
        never = [r for r in d["rows"] if not r["tuned_rows"] and not r["plan_rows"]]
        print("grid %s: %d cores, budget %d | unchanged %d  lit-by-S1 %d  still-interleaved %d"
              % (g, d["cores"], d["core_budget"], len(lit), len(dark), len(never)))
        for r in dark:
            if r["heads"] == 4:
                print("   heads=4 S=%4d  ->  %3d rows on %3d cores  (%d KB/core)"
                      % (r["S"], r["plan_rows"], r["plan_cores"], r["bytes_per_core"] >> 10))
    print("FAILURES: %d" % len(out["failures"]))
    for f_ in out["failures"][:20]:
        print("  ", f_)
    return 1 if out["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
