#!/usr/bin/env python3
"""bcx-p10-mmroof: rank the arithmetic classes CELL-LOCALLY, to choose what leg 3 sweeps.

This is not leg 2's table and it must not be quoted as one. It sums the cell's own sync-mode
device wall per class with no multiplicity and no subtraction, which is enough to decide which
eight classes are worth 25 warm reps a point and nothing more. The seconds in leg 2's table come
from `roof.py`, which runs `bcx-p10-devmap`'s subtraction and scales by the multiplicities the
composed round measured; every number this row REPORTS comes from there.

Emitted in `roof.py`'s shape so `sweep.py` reads either one.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_mmlay import mmkey as MK                       # noqa: E402
from perf.bcx_p10_mmroof import plankey as PK                    # noqa: E402
from perf.bcx_p10_mmroof import roof as RF                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--cell", default="E")
    ap.add_argument("--grid-cores", type=int, default=110)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blob = json.load(open(args.cells))
    cell = blob["cells"][args.cell]
    sec, calls, read, wrote = (collections.Counter() for _ in range(4))
    reps = collections.Counter()
    for r in cell["records"]:
        if r["mode"] != "sync":
            continue
        reps[r["stack"]] += 1
        for k, v in r["verb_wall"].items():
            verb = k.split("|")[-1]
            if verb.split("#")[0] not in MK.MM_VERBS:
                continue
            sec[verb] += v
            calls[verb] += r["verb_calls"].get(k, 0)
            read[verb] += r["verb_read"].get(k, 0)
            wrote[verb] += r["verb_written"].get(k, 0)

    rows = []
    for k, v in sec.items():
        p = PK.parse(k)
        if not p["shape"]:
            continue
        gb = (read[k] + wrote[k]) / 1e9
        cores, serial, note = RF.occupancy(p, args.grid_cores)
        rows.append({"key": k, "op": p["op"], "shape": p["shape"], "flags": p["flags"],
                     "plan": p["plan"], "factory": p["factory"], "ck": p["ck"],
                     "device_s": v, "calls": calls[k], "GB": gb,
                     "GBs": gb / v if v else 0.0,
                     "pct_dram": 100 * (gb / v) / 442.3 if v else 0.0,
                     "pct_compute": 100 * (p["flop"] * calls[k] / v / 1e12) / 85.90 if v else 0.0,
                     "cores": cores, "serial_passes": serial, "occ_note": note,
                     "binds": "cell-local, see roof.py"})
    rows.sort(key=lambda r: -r["device_s"])
    print("%-10s %-20s %-3s %6s %8s %6s %6s %6s  %s"
          % ("op", "batch x M x K x N", "fl", "calls", "cell_s", "%dram", "%cmp", "cores", "plan"))
    for r in rows[:args.top]:
        print("%-10s %-20s %-3s %6d %8.4f %5.1f%% %5.1f%% %6s  %s"
              % (r["op"].replace("experimental.", "exp."), r["shape"], (r["flags"] or "-")[:3],
                 r["calls"], r["device_s"], r["pct_dram"], r["pct_compute"],
                 r["cores"] if r["cores"] else "?", r["plan"]))
    pathlib.Path(args.out).write_text(json.dumps(
        {"cell_local_unscaled": True, "cell": args.cell, "armed": blob.get("armed"),
         "classes": rows}, indent=1))
    print("\nwrote", args.out, "(%d classes, CELL-LOCAL and unscaled)" % len(rows))


if __name__ == "__main__":
    main()
