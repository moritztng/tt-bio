#!/usr/bin/env python3
"""What the fold launches: every top-level ttnn op by name, calls, size and its current floor term.

Reuses `roof_true/true_floor.py` verbatim -- same `setup()`, same `Join`, same `op_terms`, so the
op set, the byte column and the two existing terms are the committed ones and not a second
convention. Adds nothing but the shape columns a launch-cost sweep needs to pick its arms:
output tiles, input tiles, and the per-call seconds the existing max() already charges.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import ceil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "roof_true"))
import true_floor as TF                                                       # noqa: E402


def tiles(shape):
    if not shape:
        return 0
    if len(shape) == 1:
        return ceil(shape[0] / 32)
    n = 1
    for d in shape[:-2]:
        n *= d
    return n * ceil(shape[-2] / 32) * ceil(shape[-1] / 32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", type=Path, default=HERE.parent)
    ap.add_argument("--roofs", default="shape_roofs_qb2c3_shipped.json")
    ap.add_argument("--eltwise-rate", type=float, default=None)
    ap.add_argument("--uncovered-lo", type=float, default=None)
    ap.add_argument("--uncovered-hi", type=float, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "op_census_512.json")
    ap.add_argument("--shapes-out", type=Path, default=HERE / "fold_shapes.json",
                    help="the (op class, shape, K) set `shape_ladder.py` measures a launch floor "
                         "for, keyed by true_floor.launch_key so the sweep and the join share one "
                         "definition. Ordered by calls: a partial ladder covers the most calls.")
    a = ap.parse_args()

    perf = a.perf.resolve()
    R = TF.setup(perf, a)
    R["eltwise_rate"] = R["lo_rate"]
    by = R["by"]

    per_name = defaultdict(lambda: defaultdict(float))
    per_shape = defaultdict(lambda: defaultdict(float))
    for sig in TF.TOP:
        calls = by[sig]["calls"]
        J = TF.Join(R, sig)
        for i in range(len(J.ops)):
            name = J.ops[i]["name"]
            (B, F_mm, F_el, t_tr, t_ar, _lo, _hi, _u, _l, _t,
             _la, _ls, _lc, _src) = TF.op_terms(R, J, i)
            t = max(t_tr, t_ar)
            ot = max((tiles(s) for s in J.outs[i]), default=0)
            it = sum(tiles(s) for _n, s in J.ins[i])
            e = per_name[name]
            e["calls"] += calls
            e["B"] += calls * B
            e["s_traffic"] += calls * t_tr
            e["s_arith"] += calls * t_ar
            e["s_floor"] += calls * t
            e["out_tiles"] += calls * ot
            e["in_tiles"] += calls * it
            k = "%s|out=%s|in=%s" % (name,
                                     "x".join(str(d) for d in (J.outs[i][0] if J.outs[i] else ())),
                                     ",".join("x".join(str(d) for d in s) for _n, s in J.ins[i]))
            q = per_shape[k]
            q["calls"] += calls
            q["B"] += calls * B
            q["s_floor"] += calls * t
            q["out_tiles"] = ot
            q["in_tiles"] = it

    ladder = defaultdict(float)
    # An op with no launch_arm falls out of the ladder here, and until 2026-09-17 it fell out
    # SILENTLY. `ttnn.generic_op` -- the fused trimul and SDPA kernel, 3,920 calls, 0.9127 TB, the
    # largest s_floor of any op in the 512 aa fold -- has no LAUNCH_ARM entry, so it never reached
    # fold_shapes.json, so c10-fold-census never priced it and never listed it as refused either.
    # That produced a false sentence in a concluded row ("the 52 refused launch keys: generic_op,
    # permute, ..." -- generic_op is not among them) and left the fold's second-largest block
    # unaccounted. The ladder itself is unchanged; what is added is that the drop is now counted,
    # printed and written to the output, so the next op without an arm is loud instead of invisible.
    unladdered = defaultdict(lambda: defaultdict(float))
    for sig in TF.TOP:
        calls = by[sig]["calls"]
        J = TF.Join(R, sig)
        for i in range(len(J.ops)):
            name = J.ops[i]["name"]
            arm = TF.launch_arm(name, J.ins[i])
            rows = TF.op_shape_rows(R["EF"], name, J.ins[i], J.outs[i])
            if TF.launch_key(arm, name, J.ins[i], J.outs[i], rows) is None:
                u = unladdered[name]
                u["calls"] += calls
                u["B"] += calls * TF.op_terms(R, J, i)[0]
                u["arm_is_none"] += calls if arm is None else 0
                continue
            ladder[(arm, TF.launch_shape(name, J.ins[i], J.outs[i]),
                    rows[0][2] if rows else None)] += calls
    shapes = [{"arm": a2, "shape": list(sh), "K": K, "calls": c}
              for (a2, sh, K), c in sorted(ladder.items(), key=lambda kv: -kv[1])]
    a.shapes_out.write_text(json.dumps(shapes, indent=1, default=float))
    print("launch-ladder keys: %d shapes, %d of %d calls"
          % (len(shapes), sum(e["calls"] for e in shapes),
             sum(e["calls"] for e in per_name.values())))
    if unladdered:
        ul = sorted(unladdered.items(), key=lambda kv: -kv[1]["B"])
        print("NOT IN THE LADDER -- these ops are priced by NO launch key. A census built on "
              "fold_shapes.json cannot see them, and must list them as refused or price them "
              "another way:")
        for name, u in ul:
            print("  %-52s %9d calls  %8.4f TB%s"
                  % (name.replace("ttnn.", ""), u["calls"], u["B"] / 1e12,
                     "  <-- no LAUNCH_ARM entry" if u["arm_is_none"] else ""))

    tot_calls = sum(e["calls"] for e in per_name.values())
    tot_s = sum(e["s_floor"] for e in per_name.values())
    rows = sorted(per_name.items(), key=lambda kv: -kv[1]["calls"])
    print("%-52s %9s %8s %10s %10s %9s" % ("op", "calls", "%calls", "s floor", "us/call", "out tl"))
    for name, e in rows:
        print("%-52s %9d %7.2f%% %10.4f %10.3f %9.1f"
              % (name.replace("ttnn.", ""), e["calls"], 100 * e["calls"] / tot_calls,
                 e["s_floor"], 1e6 * e["s_floor"] / e["calls"], e["out_tiles"] / e["calls"]))
    print("TOTAL %d calls, floor %.4f s" % (tot_calls, tot_s))

    shp = sorted(per_shape.items(), key=lambda kv: -kv[1]["calls"])[:60]
    out = {"total_calls": tot_calls, "total_floor_s": tot_s,
           "by_op": {k: dict(v) for k, v in rows},
           "top_shapes": {k: dict(v) for k, v in shp},
           "unladdered": {k: dict(v) for k, v in
                          sorted(unladdered.items(), key=lambda kv: -kv[1]["B"])}}
    a.out.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
