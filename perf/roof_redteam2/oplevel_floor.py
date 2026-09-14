#!/usr/bin/env python3
"""Attack 3: does `roof_budget_table.py` take max(traffic, compute) per unit, or sum traffic?

It sums traffic. `binding_floor_s = fold_B / stream_roof`, with no `max` anywhere in the file.
That is exact only if every op in the fold is bandwidth bound. Three rows of the published table
are not: the Transition rows sit at 303-380 FLOP/byte against a 247.1 FLOP/byte machine balance.

A roofline floor evaluated on an aggregate is at or below the sum of the floors of its parts,
because max(sum_i x_i, sum_i y_i) <= sum_i max(x_i, y_i). This file re-evaluates the same three
captures at three granularities, on the SAME two instruments and the SAME two roofs the published
table uses, changing nothing but where the max is taken:

  aggregate  max over the whole unit                      -- what is published
  child      max per captured sub-module                  -- the table's own rows
  op         max per top-level ttnn op inside the capture -- the finest the capture supports

No device, no new measurement. Bytes: `real_traffic.py` per-op. FLOPs: `exec_flops.py` per-op.
Both index ops through `itemize.top_level_spans`, so op i means the same op in both.
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
from real_traffic import counts as byte_counts                                # noqa: E402

TOP = ["PairformerLayer|1x512x384,1x512x512x128",
       "MSALayer|1x512x512x128,1x1024x512x64",
       "DiffusionModule|"]


def per_op_bytes(nodes):
    """real_traffic.py's REAL rule, kept per op index instead of summed. Same code path."""
    ops, rows = itemize({"nodes": nodes})
    dram = [r for r in rows if r["kind"] == "DRAM"]
    alloc_by_op = defaultdict(int)
    for r in dram:
        if r["alloc_op_i"] is not None:
            alloc_by_op[r["alloc_op_i"]] += r["size"]

    def moves_dram(i):
        n = ops[i]["name"]
        return False if n in EF.FREE and n in {"ttnn.reshape", "ttnn.unsqueeze", "ttnn.squeeze",
                                               "ttnn.deallocate"} else \
            (alloc_by_op[i] > 0 or n.endswith("_"))

    w, rd = defaultdict(int), defaultdict(int)
    for r in dram:
        readers = [i for i in r["consumers"] if moves_dram(i)]
        if r["alloc_op_i"] is not None:
            w[r["alloc_op_i"]] += r["size"]
        for i in readers:
            rd[i] += r["size"]
            if ops[i]["name"].endswith("_") and alloc_by_op[i] == 0:
                w[i] += r["size"]
        if not readers:
            rd[r["alloc_op_i"] if r["alloc_op_i"] is not None else -1] += r["size"]
    return ops, [w[i] + rd[i] for i in range(len(ops))]


def main():
    ctl = json.loads((RB / "instrument_control.json").read_text())
    st = json.loads((RB / "stream_roof2.json").read_text())
    pub = json.loads((RB / "roof_budget_512_qb2c2.json").read_text())
    C = max(r["TFLOPs"] for r in ctl["rows"] if r["label"].startswith("cube")) * 1e12
    S = max(r["GBps"] for r in st["stream"] if r["op"] == "add" and r["N"] == 8192) * 1e9
    calls = {r["sig"]: r["calls"] for r in pub["rows"]}

    out = {"compute_roof_TFLOPs": round(C / 1e12, 2), "stream_roof_GBps": round(S / 1e9, 1),
           "balance_flop_per_byte": round(C / S, 1), "units": {}}
    agg_s = op_s = 0.0
    for sig in TOP:
        cap = RB / "captures" / ("cap_" + sig.replace("|", "__") + ".json.gz")
        nodes = EF.nodes_of(str(cap))
        ops, B = per_op_bytes(nodes)
        rows = EF.per_op(nodes)
        F = []
        for name, _log, padf, kind in rows:
            F.append(padf if kind in ("matmul", "eltwise", "noshape") else 0)
        assert len(F) == len(B) == len(ops), (len(F), len(B), len(ops))

        n = calls[sig]
        Fs, Bs = sum(F), sum(B)
        s_agg = max(Fs / C, Bs / S) * n
        s_op = sum(max(f / C, b / S) for f, b in zip(F, B)) * n
        # which ops are compute bound and what they cost
        cb = [(i, ops[i]["name"], F[i], B[i]) for i in range(len(ops))
              if B[i] and F[i] / B[i] > C / S]
        cb_extra = sum(F[i] / C - B[i] / S for i, _, _, _ in cb) * n
        agg_s += s_agg
        op_s += s_op
        out["units"][sig] = {
            "calls": n, "TFLOP_per_call": round(Fs / 1e12, 4), "MB_per_call": round(Bs / 1e6, 1),
            "AI": round(Fs / Bs, 2), "n_ops": len(ops),
            "s_fold_floor_aggregate": round(s_agg, 4), "s_fold_floor_per_op": round(s_op, 4),
            "n_compute_bound_ops": len(cb),
            "compute_bound_op_names": sorted({c[1] for c in cb}),
            "compute_bound_share_of_call_FLOP": round(sum(c[2] for c in cb) / Fs, 4),
            "s_fold_added_by_compute_bound_ops": round(cb_extra, 4),
        }
    out["fold_floor_aggregate_s"] = round(agg_s, 4)
    out["fold_floor_per_op_s"] = round(op_s, 4)
    out["published_floor_s"] = pub["summary"]["binding_floor_s"]
    out["ratio_per_op_over_published"] = round(op_s / pub["summary"]["binding_floor_s"], 4)
    out["cell_s"] = pub["summary"]["cell_of_record_s"]
    out["cell_above_published_floor_s"] = round(out["cell_s"] - out["published_floor_s"], 3)
    out["cell_above_per_op_floor_s"] = round(out["cell_s"] - op_s, 3)
    print(json.dumps(out, indent=1))
    (HERE / "oplevel_floor.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
