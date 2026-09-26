#!/usr/bin/env python3
"""The shape cells, side by side, in `bcx-p10-devmap`s own table.

Reuses `devmap/analyze.py` -- its `per_block`, `to_round` and `roofline` -- so the numbers are
the same arithmetic applied to a different shape, which is the only way the two tables can be
subtracted from each other.
"""
import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import analyze as AN            # noqa: E402

FAMILY = {"tri_att_start": "triangle attention", "tri_att_end": "triangle attention",
          "tri_mul_in": "triangle multiplication", "tri_mul_out": "triangle multiplication",
          "residual": "residual (rne_residual, fp32)", "pair_transition": "pair transition",
          "msa_col_attn": "MSA column attention", "msa_row_attn": "MSA row attention",
          "opm": "outer product mean", "tape_engine": "tape engine",
          "msa_transition": "MSA transition", "mask_bias": "mask bias"}


def table(cell, args):
    blob = {"records": cell["records"], "ks": cell["ks"],
            "sync_floor_s": args.sync_floor,
            "flops_fwd_analytic_padded": cell["flops_fwd_analytic_padded"]}
    rows = AN.roofline(AN.to_round(AN.per_block(blob)), blob, args)
    fam = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        name = FAMILY.get(r["family"], r["family"])
        fam[name]["device_s"] += r["device_s"]
        fam[name]["dispatch_s"] += r["dispatch_s"]
        fam[name]["calls"] += r["calls"]
        fam[name]["GB"] += r["read_GB"] + r["written_GB"]
    split = collections.Counter()
    for r in rows:
        split[(r["stack"], "device")] += r["device_s"]
        split[(r["stack"], "dispatch")] += r["dispatch_s"]
        split[(r["dir"], "device")] += r["device_s"]
        split[(r["dir"], "GB")] += r["read_GB"] + r["written_GB"]
    return fam, split, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--dram", type=float, default=442.3)
    ap.add_argument("--tflops", type=float, default=85.90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.ridge = args.tflops * 1e12 / (args.dram * 1e9)
    blob = json.load(open(args.blob))
    args.sync_floor = blob["sync_floor_s"]
    names = list(blob["cells"])
    print("stamp:", json.dumps(blob["stamp"]))
    print("sync floor %.2f us   cells %s" % (args.sync_floor["median"] * 1e6, names))

    tabs = {n: table(blob["cells"][n], args) for n in names}
    fams = sorted({f for n in names for f in tabs[n][0]},
                  key=lambda f: -tabs[names[-1]][0].get(f, collections.Counter())["device_s"])
    hdr = "%-30s" % "family"
    for n in names:
        hdr += "%10s %9s %9s" % (n + " dev_s", "GB", "disp_s")
    if len(names) > 1:
        hdr += "%10s" % "x(last/first)"
    print(hdr)
    for f in fams:
        line = "%-30s" % f
        for n in names:
            c = tabs[n][0].get(f, collections.Counter())
            line += "%10.3f %9.1f %9.3f" % (c["device_s"], c["GB"], c["dispatch_s"])
        if len(names) > 1:
            a = tabs[names[0]][0].get(f, collections.Counter())["device_s"]
            b = tabs[names[-1]][0].get(f, collections.Counter())["device_s"]
            line += "%10s" % ("%.3f" % (b / a) if a else "-")
        print(line)
    line = "%-30s" % "TOTAL"
    tot = {}
    for n in names:
        d = sum(c["device_s"] for c in tabs[n][0].values())
        g = sum(c["GB"] for c in tabs[n][0].values())
        p = sum(c["dispatch_s"] for c in tabs[n][0].values())
        tot[n] = (d, g, p)
        line += "%10.3f %9.1f %9.3f" % (d, g, p)
    if len(names) > 1:
        line += "%10s" % ("%.3f" % (tot[names[-1]][0] / tot[names[0]][0]))
    print(line)
    for n in names:
        s = tabs[n][1]
        c = blob["cells"][n]
        print("%s  axis %d masks %s ckpt %s | evo %.3f extra %.3f | fwd %.3f bwd %.3f "
              "| fwd %.1f GB bwd %.1f GB | calls %d"
              % (n, c["device_axis"], c["masks"], c["ckpt"], s[("evo", "device")],
                 s[("extra", "device")], s[("fwd", "device")], s[("bwd", "device")],
                 s[("fwd", "GB")], s[("bwd", "GB")],
                 sum(x["calls"] for x in tabs[n][2])))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"stamp": blob["stamp"], "sync_floor_s": args.sync_floor,
             "roofs": {"dram_GBs": args.dram, "tflops": args.tflops, "ridge": args.ridge},
             "cells": {n: {"axis": blob["cells"][n]["device_axis"],
                           "masks": blob["cells"][n]["masks"],
                           "ckpt": blob["cells"][n]["ckpt"],
                           "families": {f: dict(c) for f, c in tabs[n][0].items()},
                           "rows": tabs[n][2]} for n in names}}, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
