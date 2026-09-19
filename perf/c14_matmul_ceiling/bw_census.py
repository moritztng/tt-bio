#!/usr/bin/env python3
"""Which matmuls in the fold resolve to in0_block_w=1, and what do they cost?

`shape_gap.py` ranked the class by slack against its own roofline. This asks a mechanism question
instead: the resolved program config is in every capture row's ATTRIBUTES, so the fold tells us
directly which of its matmuls stream the inner dimension ONE 32x32 tile at a time. That is the
pathology c12-matmul-key-attribution named twice without pricing across the class, and the engine
already fixes it elsewhere -- `_pair_proj_config` picks
`bw = max(d for d in (k_tiles, 8, 4, 2, 1) if d <= cap and k_tiles % d == 0)`.

Weighted by the fold's own integer unit call counts, same as shape_gap.py, so the seconds are fold
seconds. No device, no counter: the config string and the kernel duration both come out of the
committed capture.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

UNIT_CALLS = {"PairformerLayer": 264, "MSALayer": 16, "DiffusionModule": 200,
              "PairConditioningDevice": 1, "RelPosGather": 2, "PairAssemblyDevice": 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c12", type=Path, required=True)
    ap.add_argument("--run", action="append", default=[], metavar="DIR:UNIT")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    sys.path.insert(0, str(a.c12))
    import reduce as R  # noqa: E402

    agg = defaultdict(lambda: {"calls": 0.0, "s": 0.0, "cores": set(), "units": set()})
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
            print("REFUSED %s: %s" % (d, meta.get("why")))
            continue
        w = UNIT_CALLS[unit] / reps
        for r in win:
            if r.get("OP CODE") != "MatmulDeviceOperation":
                continue
            at = r.get("ATTRIBUTES") or ""
            # The config string nests `(x=11;y=10)`, so a `[^)]*` body match stops before
            # in0_block_w and silently reads 0 for every row. Match the fields, not the body.
            m = re.search(r"(Matmul\w*ProgramConfig)\(", at)
            fac = m.group(1) if m else "none"
            body = at
            bw = re.search(r"in0_block_w=(\d+)", body)
            pcm = re.search(r"per_core_M=(\d+)", body)
            pcn = re.search(r"per_core_N=(\d+)", body)
            o = R.rshape(r, "OUTPUT_0", logical=True) or R.rshape(r, "OUTPUT_0")
            i0 = R.rshape(r, "INPUT_0", logical=True) or R.rshape(r, "INPUT_0")
            if not (o and i0):
                continue
            b, mm, n = o[0] * o[1], o[2], o[3]
            k = i0[3] if i0[2] == mm else i0[2]
            kt = -(-k // 32)
            key = (b, mm, k, n, fac, int(bw.group(1)) if bw else 0)
            e = agg[key]
            e["calls"] += w
            e["s"] += R.fnum(r, "DEVICE KERNEL DURATION [ns]") / 1e9 * w
            e["cores"].add(int(R.fnum(r, "CORE COUNT")))
            e["units"].add(unit)
            e["kt"] = kt
            e["per_core"] = (pcm.group(1) if pcm else "?", pcn.group(1) if pcn else "?")

    out = []
    for (b, mm, k, n, fac, bw), e in agg.items():
        out.append({"shape": [b, mm, k, n], "k_tiles": e["kt"], "factory": fac, "in0_block_w": bw,
                    "bw_over_kt": round(bw / e["kt"], 4) if e["kt"] else None,
                    "calls": round(e["calls"], 1), "fold_s": round(e["s"], 5),
                    "us_per_call": round(1e6 * e["s"] / e["calls"], 2) if e["calls"] else None,
                    "cores": sorted(e["cores"]), "per_core_MN": e["per_core"],
                    "units": sorted(e["units"])})
    out.sort(key=lambda r: -r["fold_s"])
    tot = sum(r["fold_s"] for r in out)
    starved = [r for r in out if r["in0_block_w"] == 1 and r["k_tiles"] > 1]
    print("class total %.4f s over %d (shape, config) groups" % (tot, len(out)))
    print("in0_block_w == 1 with K > 1 tile: %d groups, %.4f s = %.1f %% of the class"
          % (len(starved), sum(r["fold_s"] for r in starved),
             100 * sum(r["fold_s"] for r in starved) / tot))
    print("\n%-22s %4s %4s %9s %9s %9s %6s %s"
          % ("shape b,M,K,N", "kt", "bw", "calls", "fold_s", "us/call", "cores", "factory"))
    for r in out:
        print("%-22s %4d %4d %9.0f %9.4f %9.2f %6s %s%s"
              % (",".join(str(x) for x in r["shape"]), r["k_tiles"], r["in0_block_w"], r["calls"],
                 r["fold_s"], r["us_per_call"] or 0, r["cores"][0] if r["cores"] else "?",
                 r["factory"].replace("MatmulMultiCoreReuse", "MMCR").replace("ProgramConfig", ""),
                 "   <-- STARVED" if r["in0_block_w"] == 1 and r["k_tiles"] > 1 else ""))
    if a.out:
        a.out.write_text(json.dumps({"class_s": round(tot, 5), "rows": out}, indent=1,
                                    default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
