#!/usr/bin/env python3
"""Attack 5: `real_traffic.py` charges nothing to an op whose output lands in L1.

Round 1 found `census.py:split_io` drops the read of a read-modify-write, 8.49 % of a block.
`real_traffic.py` -- the counter behind the tip's 2.9449 TB -- does NOT have that defect: it
charges an in-place op a read AND a write of its destination. It has a different one.

    def moves_dram(i):
        name = ops[i]["name"]
        if name in NO_TRAFFIC: return False
        return alloc_by_op[i] > 0 or name.endswith("_")

`alloc_by_op` is built from the DRAM rows only, so it is zero for an op whose output buffer was
allocated in L1. Such an op is then not a "reader" of anything, and every DRAM buffer it consumes
is uncharged at that op. In `cap_PairformerLayer__1x512x384,1x512x512x128` the very first fused
projection, `ttnn.linear` op 19, reads 67.14 MB of DRAM and allocates 67.11 MB in L1; it is charged
zero. 111 of the capture's 410 ops are in this class and they carry 79.3 % of the block's executed
FLOPs.

A DRAM buffer whose ONLY consumers are in that class falls through to the `if not readers:` branch
and is charged one read against its producer, so the total is not lost outright -- it is charged
once where it should be charged once per consuming op. Where a buffer has several such consumers,
or a mix, the reads after the first vanish.

This file re-counts with one line widened: an op moves DRAM if it allocated ANY buffer, DRAM or L1.
Nothing else changes -- same capture, same in-place rule, same NO_TRAFFIC set, same dedupe on
buffer address.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RB = ROOT / "perf" / "_rb_tmp"
sys.path.insert(0, str(RB))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))

import exec_flops as EF                                                       # noqa: E402
from itemize import itemize                                                   # noqa: E402
from real_traffic import counts as published_counts, NO_TRAFFIC               # noqa: E402

TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64",
       "DiffusionModule|"]


def counts_any_alloc(nodes):
    """real_traffic.counts with `moves_dram` widened to any allocation, DRAM or L1."""
    ops, rows = itemize({"nodes": nodes})
    dram = [r for r in rows if r["kind"] == "DRAM"]
    alloc_dram, alloc_any = defaultdict(int), defaultdict(int)
    for r in rows:
        if r["alloc_op_i"] is not None:
            alloc_any[r["alloc_op_i"]] += r["size"]
            if r["kind"] == "DRAM":
                alloc_dram[r["alloc_op_i"]] += r["size"]

    def moves(i):
        n = ops[i]["name"]
        return False if n in NO_TRAFFIC else (alloc_any[i] > 0 or n.endswith("_"))

    w, rd = defaultdict(int), defaultdict(int)
    for r in dram:
        readers = [i for i in r["consumers"] if moves(i)]
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in readers:
            rd[i] += r["size"]
            if ops[i]["name"].endswith("_") and alloc_dram[i] == 0:
                w[i] += r["size"]
        if not readers:
            rd[r["alloc_op_i"] if r["alloc_op_i"] is not None else -1] += r["size"]
    return {"real_MB": (sum(w.values()) + sum(rd.values())) / 1e6,
            "n_ops": len(ops),
            "n_l1_out_ops": sum(1 for i in range(len(ops))
                                if alloc_any[i] > 0 and alloc_dram[i] == 0)}


def main():
    st = json.loads((RB / "stream_roof2.json").read_text())
    pub = json.loads((RB / "roof_budget_512_qb2c2.json").read_text())
    S = max(r["GBps"] for r in st["stream"] if r["op"] == "add" and r["N"] == 8192) * 1e9
    calls = {r["sig"]: r["calls"] for r in pub["rows"]}

    out, B0, B1 = {"stream_roof_GBps": round(S / 1e9, 1), "units": {}}, 0.0, 0.0
    for sig in TOP:
        nodes = EF.nodes_of(str(RB / "captures" / ("cap_" + sig.replace("|", "__") + ".json.gz")))
        a = published_counts({"nodes": nodes})["real_MB"]
        b = counts_any_alloc(nodes)
        n = calls[sig]
        B0 += a * n
        B1 += b["real_MB"] * n
        out["units"][sig] = {"calls": n, "published_MB_per_call": round(a, 1),
                             "corrected_MB_per_call": round(b["real_MB"], 1),
                             "ratio": round(b["real_MB"] / a, 4),
                             "n_ops": b["n_ops"], "n_ops_with_L1_output": b["n_l1_out_ops"]}
    out["published_fold_TB"] = round(B0 / 1e6, 4)
    out["corrected_fold_TB"] = round(B1 / 1e6, 4)
    out["ratio"] = round(B1 / B0, 4)
    out["published_traffic_floor_s"] = round(B0 * 1e6 / S, 3)
    out["corrected_traffic_floor_s"] = round(B1 * 1e6 / S, 3)
    print(json.dumps(out, indent=1))
    (HERE / "l1_output_gap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
