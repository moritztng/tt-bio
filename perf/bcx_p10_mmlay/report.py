#!/usr/bin/env python3
"""bcx-p10-mmlay leg 1: the round's matmuls ranked by SHAPE, and placed on the roofline.

Reads `cells.py`'s blob and `bcx-p10-calls`'s anchor round, runs `bcx-p10-devmap`'s unchanged
subtraction on the widened key, and scales to the round at the multiplicities the round
measured -- the same three steps `perf/bcx_p10_calls/calls.py` runs, so A_shape and A_verb are
comparable by construction. The wider key clamps more per-key zeros, so the difference is
printed rather than assumed away.

FLOPs per shape are COUNTED: 2 * batch * M * K * N off the padded shapes in the key. The
roofline placement is then measured on both axes at once, and that is the whole point of leg 1:

  intensity < ridge, %dram high   -> bandwidth-bound AT the roof. Nothing to win.
  intensity < ridge, %dram low    -> below both roofs. Neither operand placement nor the
                                     block config is excused; this is where the 60 % is.
  intensity > ridge, %compute low -> a program-config problem, not a placement problem.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import analyze as AN                    # noqa: E402
from perf.bcx_p10_devgap import gap as GAP                       # noqa: E402
from perf.bcx_p10_shape import compare as CMP                    # noqa: E402
from perf.bcx_p10_mmlay import mmkey as MK                       # noqa: E402


def scale(pbv, mult):
    rows = collections.defaultdict(collections.Counter)
    fam_of = collections.defaultdict(collections.Counter)
    unscaled = set()
    for (stack, direction, family, key), v in pbv.items():
        m = mult.get((stack, direction))
        if m is None:
            unscaled.add((stack, direction))
            continue
        c = rows[key]
        c["device_s"] += v["device"] * m
        c["dispatch_s"] += v["enqueue"] * m
        c["calls"] += v["calls"] * m
        c["GB"] += (v["read"] + v["written"]) * m / 1e9
        fam_of[key][CMP.FAMILY.get(family, family)] += v["device"] * m
    return rows, fam_of, unscaled


def verdict(intensity, pct_dram, pct_flop, ridge, near=70.0):
    if intensity is None:
        return "?"
    if intensity >= ridge:
        return "COMPUTE-bound: at roof" if pct_flop >= near else "compute side, BELOW roof: program config"
    if pct_dram >= near:
        return "DRAM-bound: at roof"
    return "BELOW both roofs: placement + config"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--cell", default="E")
    ap.add_argument("--dram", type=float, default=442.3)
    ap.add_argument("--tflops", type=float, default=85.90)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--near", type=float, default=70.0,
                    help="%% of a roof above which a shape counts as AT that roof")
    ap.add_argument("--calls-averb", type=float, default=None,
                    help="bcx-p10-calls' A_verb matmul-family seconds, to reconcile against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ridge = args.tflops * 1e12 / (args.dram * 1e9)

    stamp, rounds, R = GAP.round_column(args.round)
    blob = json.load(open(args.cells))
    cell = blob["cells"][args.cell]
    sub = {"records": cell["records"], "ks": cell["ks"],
           "sync_floor_s": blob["sync_floor_s"],
           "flops_fwd_analytic_padded": cell["flops_fwd_analytic_padded"]}

    print("== the round this table is scaled to ==")
    print(json.dumps({k: stamp.get(k) for k in
                      ("host", "card", "commit", "triatt_hifi", "extra_msa_on_device",
                       "template_on_device", "exact", "binder_pinned", "seed")}))
    print("median over rounds 2..%d: wall %.3f | DEVICE COLUMN %.3f | AICLK med %s min %s | load1 %.1f"
          % (R["n_rounds_medianed"] + 1, R["wall"], R["device"],
             R["aiclk_med"], R["aiclk_min_over_all"], R["load1"]))
    print("== the cell this table is measured on ==")
    print("cell %s, axis %d, reps %d, hifi %s, reach %s"
          % (args.cell, cell["device_axis"], cell["reps"], cell.get("hifi"),
             json.dumps(cell.get("reach", {}).get("fused_hifi", {}))))
    ld = [r["loadavg"] for r in cell["records"]]
    ck = [r["aiclk"]["median"] for r in cell["records"] if r.get("aiclk")]
    lo = [r["aiclk"]["min"] for r in cell["records"] if r.get("aiclk")]
    print("AICLK during the timed windows: median %s, min %s | loadavg1 %.1f-%.1f"
          % (sorted(ck)[len(ck) // 2] if ck else "?", min(lo) if lo else "?", min(ld), max(ld)))
    print("roofs: DRAM %.1f GB/s, %.2f bf16 TFLOP/s at HiFi4, ridge %.1f FLOP/byte"
          % (args.dram, args.tflops, ridge))

    mult = {}
    for stack, module, nb in (("evo", "evoformer", 48), ("extra", "extra_msa", 4)):
        mult[(stack, "fwd")] = nb * (R[f"{module}.taped.n"] + R[f"{module}.primal.n"])
        mult[(stack, "bwd")] = nb * R[f"{module}.backward.n"]

    pbv = AN.per_block(sub, by_verb=True)
    rows, fam_of, unscaled = scale(pbv, mult)
    if unscaled:
        print("UNSCALED: %s" % sorted(unscaled))

    mm = {k: v for k, v in rows.items() if k.split("#")[0] in MK.MM_VERBS}
    A_all = sum(c["device_s"] for c in rows.values())
    A_mm = sum(c["device_s"] for c in mm.values())
    GB_mm = sum(c["GB"] for c in mm.values())
    calls_mm = sum(c["calls"] for c in mm.values())
    flop_mm = sum(MK.parse(k)["flop"] * c["calls"] for k, c in mm.items() if "#" in k)

    print("\n== leg 1: the round's matmul family by SHAPE, %d distinct shapes, top %d =="
          % (len(mm), args.top))
    hdr = ("op", "batch x M x K x N", "fl", "calls", "dev_s", "GB", "GB/s", "%dram",
           "TFLOP/s", "%cmp", "FLOP/B", "roofline verdict")
    print("%-12s %-22s %-3s %6s %7s %7s %7s %6s %8s %5s %7s  %s" % hdr)
    ordered = sorted(mm.items(), key=lambda kv: -kv[1]["device_s"])
    out_rows = []
    for key, c in ordered:
        p = MK.parse(key)
        d, gb, n = c["device_s"], c["GB"], c["calls"]
        gbs = gb / d if d else 0.0
        fl = p["flop"] * n if p["shape"] else 0.0
        tf = fl / d / 1e12 if d else 0.0
        inten = fl / (gb * 1e9) if gb else None
        pd, pc = 100 * gbs / args.dram, 100 * tf / args.tflops
        r = {"key": key, "device_s": d, "GB": gb, "calls": n, "GBs": gbs, "TFLOPs": tf,
             "pct_dram": pd, "pct_compute": pc, "intensity": inten,
             "verdict": verdict(inten, pd, pc, ridge, args.near),
             "called_from": dict(fam_of[key]),
             "a": p.get("a"), "b": p.get("b"), "o": p.get("o"), "flags": p.get("flags"),
             "extra": p.get("extra")}
        out_rows.append(r)
    for r in out_rows[:args.top]:
        p = MK.parse(r["key"])
        print("%-12s %-22s %-3s %6d %7.3f %7.1f %7.1f %5.1f%% %8.2f %4.1f%% %7.1f  %s"
              % (p["op"].replace("experimental.", "exp."), p.get("shape", "-"),
                 p.get("flags", "-")[:3], round(r["calls"]), r["device_s"], r["GB"],
                 r["GBs"], r["pct_dram"], r["TFLOPs"], r["pct_compute"],
                 r["intensity"] or 0, r["verdict"]))
    rest = out_rows[args.top:]
    if rest:
        print("%-12s %-22s %-3s %6d %7.3f %7.1f" % ("... %d more" % len(rest), "", "",
              round(sum(r["calls"] for r in rest)), sum(r["device_s"] for r in rest),
              sum(r["GB"] for r in rest)))
    print("%-12s %-22s %-3s %6d %7.3f %7.1f %7.1f %5.1f%% %8.2f %4.1f%% %7.1f"
          % ("MATMUL FAM", "", "", round(calls_mm), A_mm, GB_mm,
             GB_mm / A_mm, 100 * (GB_mm / A_mm) / args.dram,
             flop_mm / A_mm / 1e12, 100 * (flop_mm / A_mm / 1e12) / args.tflops,
             flop_mm / (GB_mm * 1e9)))

    print("\n== where each top shape is issued ==")
    for r in out_rows[:args.top]:
        p = MK.parse(r["key"])
        src = sorted(r["called_from"].items(), key=lambda kv: -kv[1])
        tot = sum(v for _, v in src) or 1.0
        print("%-12s %-22s %-3s  %s | a %s  b %s  o %s %s"
              % (p["op"].replace("experimental.", "exp."), p.get("shape", "-"),
                 p.get("flags", "-")[:3],
                 ", ".join("%s %.0f%%" % (n, 100 * v / tot) for n, v in src[:3]),
                 r["a"], r["b"], r["o"], ("[" + r["extra"] + "]") if r["extra"] else ""))

    print("\n== reconciliation: the wider key against the op-name key ==")
    print("%-56s %9s" % ("term", "seconds"))
    print("%-56s %9.3f" % ("A_shape: whole column on the widened key", A_all))
    print("%-56s %9.3f" % ("  of which the matmul family", A_mm))
    if args.calls_averb is not None:
        diff = A_mm - args.calls_averb
        print("%-56s %9.3f" % ("bcx-p10-calls' matmul family on the NAME key", args.calls_averb))
        print("%-56s %9.3f  (%+.1f %%)"
              % ("  widening cost: extra per-key zero clamping", diff,
                 100 * diff / args.calls_averb))
    print("%-56s %9.3f" % ("MEASURED DEVICE COLUMN (C) of the round", R["device"]))

    print("\n== the roofline verdict, by seconds ==")
    agg = collections.Counter()
    for r in out_rows:
        agg[r["verdict"]] += r["device_s"]
    for v, s in agg.most_common():
        print("%-46s %7.3f s  %5.1f %% of the family" % (v, s, 100 * s / A_mm))

    print("\n== operand placement, by seconds: what the model actually asks for ==")
    place = collections.Counter()
    for r in out_rows:
        place[(r["a"], r["b"], r["o"])] += r["device_s"]
    for (a, b, o), s in place.most_common(8):
        print("%7.3f s  %5.1f %%   a %s | b %s | o %s" % (s, 100 * s / A_mm, a, b, o))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"round_stamp": stamp, "round_median": R, "cell": args.cell,
             "cell_reach": cell.get("reach"), "sync_floor_s": blob["sync_floor_s"],
             "roofs": {"dram_GBs": args.dram, "tflops": args.tflops, "ridge": ridge},
             "multiplicities": {f"{k[0]}.{k[1]}": v for k, v in mult.items()},
             "shapes": out_rows,
             "totals": {"A_shape": A_all, "A_mm": A_mm, "GB_mm": GB_mm,
                        "calls_mm": calls_mm, "flop_mm": flop_mm, "C": R["device"],
                        "round_wall": R["wall"]}}, indent=1))
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
