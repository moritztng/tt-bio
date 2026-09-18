#!/usr/bin/env python3
"""Extract the fold's RESOLVED matmul program config, per (shape, config) group, from the capture.

An arm that does not carry the fold's configuration cannot price a lever at these sites: session s1
measured that the hard way, reading the fold's own shape 2.2x slow. So the bw ladder does not
hand-write a program config, it reads the fold's out of the committed capture and varies exactly one
field. This writes the table it reads.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

UNIT_CALLS = {"PairformerLayer": 264, "MSALayer": 16, "DiffusionModule": 200}
FIELDS = ("in0_block_w", "out_subblock_h", "out_subblock_w", "out_block_h", "out_block_w",
          "per_core_M", "per_core_N", "fuse_batch", "transpose_mcast", "mcast_in0", "gather_in0",
          "num_global_cb_receivers", "untilize_out")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c12", type=Path, required=True)
    ap.add_argument("--run", action="append", default=[], metavar="DIR:UNIT")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.c12))
    import reduce as R  # noqa: E402

    agg = defaultdict(lambda: {"calls": 0.0, "s": 0.0})
    for spec in a.run:
        d, _, unit = spec.partition(":")
        run = a.c12 / "runs" / d
        rows = R.ops_report(run / "ops_perf_results.csv.gz")
        uj = json.loads((run / "unit.json").read_text())
        if "unit_order" in uj:
            full = list(uj["unit_order"])
            reps = int(uj["units"][unit]["reps"])
        else:
            full = [uj["env"].get("unit")]
            reps = int(uj["env"]["reps"])
        win, meta = R.window(rows, full.index(unit), len(full))
        if win is None:
            continue
        w = UNIT_CALLS[unit] / reps
        for r in win:
            if r.get("OP CODE") != "MatmulDeviceOperation":
                continue
            at = r["ATTRIBUTES"]
            fac = re.search(r"(Matmul\w*ProgramConfig)\(", at)
            if not fac:
                continue
            o = R.rshape(r, "OUTPUT_0", logical=True) or R.rshape(r, "OUTPUT_0")
            i0 = R.rshape(r, "INPUT_0", logical=True) or R.rshape(r, "INPUT_0")
            if not (o and i0):
                continue
            b, m, n = o[0] * o[1], o[2], o[3]
            k = i0[3] if i0[2] == m else i0[2]
            cfg = {}
            for f in FIELDS:
                mm = re.search(r"\b%s=(\d+)" % f, at)
                if mm:
                    cfg[f] = int(mm.group(1))
            key = (b, m, k, n, fac.group(1), cfg.get("in0_block_w", 0))
            e = agg[key]
            e["calls"] += w
            e["s"] += R.fnum(r, "DEVICE KERNEL DURATION [ns]") / 1e9 * w
            e["cfg"] = cfg
            e["factory"] = fac.group(1)
            e["cores"] = int(R.fnum(r, "CORE COUNT"))
            e["grid"] = [int(x) for x in re.search(r"grid_size=\(x=(\d+);y=(\d+)\)", at).groups()]
            e["mem"] = [r["INPUT_0_MEMORY"], r["INPUT_1_MEMORY"], r["OUTPUT_0_MEMORY"]]
            e["bias"] = bool(str(r.get("INPUT_2_Y_PAD[LOGICAL]") or "").strip())
            e["fidelity"] = r["MATH FIDELITY"]
            e["unit"] = unit

    out = []
    for (b, m, k, n, fac, bw), e in agg.items():
        out.append({"shape": [b, m, k, n], "k_tiles": -(-k // 32), "factory": fac,
                    "in0_block_w": bw, "calls": round(e["calls"], 1), "fold_s": round(e["s"], 5),
                    "us_per_call": round(1e6 * e["s"] / e["calls"], 3),
                    "cfg": e["cfg"], "cores": e["cores"], "grid": e["grid"], "mem": e["mem"],
                    "bias": e["bias"], "fidelity": e["fidelity"], "unit": e["unit"]})
    out.sort(key=lambda r: -r["fold_s"])
    a.out.write_text(json.dumps(out, indent=1))
    for r in out[:14]:
        print("%-20s kt=%-3d bw=%-3d %9.4f s %9.3f us %3d cores %s %s"
              % (",".join(str(x) for x in r["shape"]), r["k_tiles"], r["in0_block_w"],
                 r["fold_s"], r["us_per_call"], r["cores"],
                 r["factory"].replace("MatmulMultiCoreReuse", "MMCR").replace("ProgramConfig", ""),
                 "".join("L" if "L1" in x else "D" for x in r["mem"])))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
