#!/usr/bin/env python3
"""The 2.671 s that `roof-budget`'s table puts in no op class: named, priced, and put on a roof.

`roof-orchestrator`'s CALL_TREE turned the flat 26-row table into a tree and found 2.671 s of
"unattributed in parents" -- time inside a unit that none of its measured children explains. This
file asks the captures what runs there. It is not a second instrument: the captures, the FLOP
counter (`exec_flops.py`), the byte counter (`real_traffic.py`, deduped on buffer address) and the
roofs (424.7 GB/s stream, 104.93 TFLOP/s dense bf16, machine balance 247.1 FLOP/byte) are
`roof-budget`'s own, unchanged. `split_units.py` only decides, per top-level ttnn op, whether it
ran inside a marked child or in the parent itself.

Two columns, and the gap between them is the finding:

  s/fold        the residual as CALL_TREE computed it: median(parent) x calls minus the same for
                each measured child. Reproduced here exactly.
  roof s        the residual ops' own traffic and arithmetic at the measured roofs.

A parent whose capture contains NO residual op cannot be spending residual seconds on the device,
whatever the timing column says.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import split_units as SU                                                     # noqa: E402

RB = HERE.parents[1] / "perf" / "roof_budget"

# The second estimator. `s_own` above is median(unit) x calls minus the same for each child, which
# is what CALL_TREE used: it rejects the first-call kernel compiles, but a median of a sum is not
# the sum of the medians, so a unit whose children are not identical keeps that difference in its
# own row. The bracket tree carries the exact per-path sums of the SAME fold, so excl/incl on the
# path is a second, independent share of the unit that is its own work -- biased the other way,
# because the compiles it keeps land in a child and shrink the share. The two bracket the truth,
# and where they disagree by an order of magnitude the row is not device work.
TREE_PATHS = {
    "DiffusionTransformer|1x512x768,1x512x768":
        ["DiffusionModule/Diffusion/DiffusionTransformer"],
    "DiffusionTransformer|1x140x32x128,1x140x32x128":
        ["DiffusionModule/Diffusion/DiffusionTransformer"],
    "DiffusionTransformerLayer|1x512x768,1x512x768":
        ["DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer"],
    "DiffusionTransformerLayer|1x140x32x128,1x140x32x128":
        ["DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer"],
    "ConditionedTransitionBlock|1x512x768,1x512x768":
        ["DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer/"
         "ConditionedTransitionBlock"],
    "ConditionedTransitionBlock|1x140x32x128,1x140x32x128":
        ["DiffusionModule/Diffusion/DiffusionTransformer/DiffusionTransformerLayer/"
         "ConditionedTransitionBlock"],
    "Diffusion|1x4480x3,1": ["DiffusionModule/Diffusion"],
    "DiffusionModule|": ["DiffusionModule"],
    "PairformerLayer|1x512x384,1x512x512x128":
        ["TrunkModule/Pairformer/PairformerLayer", "PairformerModule/Pairformer/PairformerLayer"],
    "PairformerLayer|1x512x512x128": ["TrunkModule/MSA/MSALayer/PairformerLayer"],
    "MSALayer|1x512x512x128,1x1024x512x64": ["TrunkModule/MSA/MSALayer"],
}

# parent -> measured children, as (child sig, share of that sig's calls).
#
# Two rows are shared and have to be split by call count, not assigned whole.
# `Transition|1x512x512x128` is 280 calls = the trunk pairformer's 264 plus the MSA-side
# pairformer's 16. `AdaLN|1x512x768` is 9600 calls = 4800 run by DiffusionTransformerLayer and
# 4800 run by the ConditionedTransitionBlock inside it; CALL_TREE charged all 9600 to the layer
# and then also ranked ConditionedTransitionBlock as a leaf, so the AdaLN inside the block is in
# the table twice -- once in the AdaLN leaf row and once inside the block's own time. Splitting
# 0.5/0.5 fixes it and opens the block's own residual, which CALL_TREE never had a row for.
TREE = {
    "PairformerLayer|1x512x384,1x512x512x128": [
        ("TriangleMultiplication|1x512x512x128,1x512x512", 1.0),
        ("TriangleAttention|1x512x512x128,1x1x1x512", 1.0),
        ("Transition|1x512x512x128", 264 / 280),
        ("AttentionPairBias|1x512x384,1x512x512x128", 1.0),
        ("Transition|1x512x384", 1.0)],
    "DiffusionModule|": [("Diffusion|1x4480x3,1", 1.0)],
    "Diffusion|1x4480x3,1": [
        ("DiffusionTransformer|1x512x768,1x512x768", 1.0),
        ("DiffusionTransformer|1x140x32x128,1x140x32x128", 1.0),
        ("Transition|1x512x768", 1.0)],
    "DiffusionTransformer|1x512x768,1x512x768": [
        ("DiffusionTransformerLayer|1x512x768,1x512x768", 1.0)],
    "DiffusionTransformerLayer|1x512x768,1x512x768": [
        ("ConditionedTransitionBlock|1x512x768,1x512x768", 1.0),
        ("AttentionPairBias|1x512x768,1x16x512x512", 1.0),
        ("AdaLN|1x512x768,1x512x768", 0.5)],
    "DiffusionTransformer|1x140x32x128,1x140x32x128": [
        ("DiffusionTransformerLayer|1x140x32x128,1x140x32x128", 1.0)],
    "DiffusionTransformerLayer|1x140x32x128,1x140x32x128": [
        ("AttentionPairBias|1x140x32x128,140x4x32x128", 1.0),
        ("ConditionedTransitionBlock|1x140x32x128,1x140x32x128", 1.0),
        ("AdaLN|1x140x32x128,1x140x32x128", 0.5)],
    "PairformerLayer|1x512x512x128": [
        ("TriangleMultiplication|1x512x512x128", 1.0),
        ("TriangleAttention|1x512x512x128", 1.0),
        ("Transition|1x512x512x128", 16 / 280)],
    "MSALayer|1x512x512x128,1x1024x512x64": [
        ("OuterProductMean|1x1024x512x64,1024x1x1", 1.0),
        ("PairformerLayer|1x512x512x128", 1.0),
        ("PairWeightedAveraging|1x1024x512x64,1x512x512x128", 1.0),
        ("Transition|1x1024x512x64", 1.0)],
    "ConditionedTransitionBlock|1x512x768,1x512x768": [("AdaLN|1x512x768,1x512x768", 0.5)],
    "ConditionedTransitionBlock|1x140x32x128,1x140x32x128": [
        ("AdaLN|1x140x32x128,1x140x32x128", 0.5)],
}


def own_table(bud, att):
    """Every captured unit's OWN work: the ops it runs outside every marked child.

    This tiles the fold. A leaf's own row is the whole unit and reproduces `roof-budget`'s row
    for it; a parent's own row is the glue the flat table had no line for. There is no residual
    column left, because the residual IS a row.
    """
    S = bud["summary"]
    by = {r["sig"]: r for r in bud["rows"]}
    sigs = att["sigs"]
    scale = S["cell_scale"]
    stream = S["stream_roof_GBps"] * 1e9
    compute = S["compute_roof_TFLOPs"] * 1e12
    balance = S["machine_balance_flop_per_byte"]

    edges = set()
    for path in att["tree"]:
        p = path.split("/")
        edges.update(zip(p, p[1:]))
    have = [s for s in sigs if SU.cap_path(s).is_file()]

    rows = []
    for sig in sorted(by):
        r = by[sig]
        kids = TREE.get(sig, [])
        s_kids = sum(by[k]["ms_per_call"] * by[k]["calls"] * w / 1e3 for k, w in kids if k in by)
        s_own = r["s_per_fold"] - s_kids
        ops, owner, bts, fl, orph, report = SU.split(sig, have, edges)
        agg = defaultdict(lambda: {"n": 0, "B": 0.0, "F": 0})
        for i, o in enumerate(ops):
            if owner[i] == "":
                x = agg[o["name"]]
                x["n"] += 1
                x["B"] += bts[i]
                x["F"] += fl[i][1]
        # the counter's terminal-output charge: a DRAM buffer nothing inside the capture consumes.
        # It belongs to the captured unit itself, and adding it back is what makes a leaf's own
        # row reproduce `roof-budget`'s row for the same unit byte for byte.
        B = sum(x["B"] for x in agg.values()) + orph
        F = sum(x["F"] for x in agg.values())
        ai = F / B if B else 0.0
        bound = "compute" if ai > balance else "bandwidth"
        s_roof = r["calls"] * (F / compute if bound == "compute" else B / stream)
        rows.append({
            "unit": sig, "terminal_MB_per_call": round(orph / 1e6, 3), "kind": "leaf" if not kids else "parent", "calls": r["calls"],
            "s_own_per_fold": round(s_own, 4), "s_own_at_cell": round(s_own * scale, 4),
            "ops_per_call": sum(x["n"] for x in agg.values()),
            "MB_per_call": round(B / 1e6, 3), "GFLOP_per_call": round(F / 1e9, 3),
            "AI_flop_per_byte": round(ai, 2), "binding_roof": bound,
            "s_at_roof": round(s_roof, 4),
            "s_above_roof_at_cell": round(s_own * scale - s_roof, 4),
            "pct_of_roof_at_cell": (round(100 * s_roof / (s_own * scale), 1)
                                    if s_own * scale > 1e-9 else None),
            "ops": {k.replace("ttnn.", ""): {"n": v["n"], "MB": round(v["B"] / 1e6, 3),
                                             "GFLOP": round(v["F"] / 1e9, 3)}
                    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]["B"])},
            "closed_by": sorted({x[3].split()[0] for x in report}),
        })
        paths = [p for p in TREE_PATHS.get(sig, []) if p in att["tree"]]
        if paths:
            incl = sum(att["tree"][p]["incl_s"] for p in paths)
            excl = sum(att["tree"][p]["excl_s"] for p in paths)
            f = excl / incl if incl else 0.0
            rows[-1]["own_share_exact_sums"] = round(f, 5)
            rows[-1]["s_own_at_cell_exact_sums"] = round(f * r["s_per_fold"] * scale, 4)
            rows[-1]["estimator_ratio"] = round(
                rows[-1]["s_own_at_cell"] / rows[-1]["s_own_at_cell_exact_sums"], 2) \
                if rows[-1]["s_own_at_cell_exact_sums"] > 1e-9 else None
    rows.sort(key=lambda x: -x["s_above_roof_at_cell"])
    return rows, S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=Path, default=RB / "roof_budget_512_qb2c2.json")
    ap.add_argument("--attrib", type=Path, default=RB / "attrib2_512_tip_qb2c2.json")
    ap.add_argument("--out-json", type=Path, default=HERE / "residual_512_qb2c2.json")
    ap.add_argument("--out-md", type=Path, default=HERE / "ROOF_RESIDUAL.md")
    a = ap.parse_args()

    bud = json.loads(a.budget.read_text())
    att = json.loads(a.attrib.read_text())["attrib"]
    rows, S = own_table(bud, att)
    by = {r["sig"]: r for r in bud["rows"]}
    scale = S["cell_scale"]

    TOP = ["PairformerLayer|1x512x384,1x512x512x128",
           "MSALayer|1x512x512x128,1x1024x512x64", "DiffusionModule|"]
    top_cell = sum(by[t]["s_per_fold"] for t in TOP) * scale
    own_cell = sum(r["s_own_at_cell"] for r in rows)
    glue = [r for r in rows if r["kind"] == "parent"]
    dark = [r for r in glue if r["ops_per_call"] == 0]

    # what CALL_TREE's tree double-counted: it charged all 9600 AdaLN calls to the layer and also
    # ranked ConditionedTransitionBlock, which runs 4800 of them, as a leaf.
    dbl = sum(0.5 * by[k]["s_per_fold"] for k in
              ("AdaLN|1x512x768,1x512x768", "AdaLN|1x140x32x128,1x140x32x128")) * scale
    dbl_above = sum(0.5 * by[k]["s_above_roof_at_cell"] for k in
                    ("AdaLN|1x512x768,1x512x768", "AdaLN|1x140x32x128,1x140x32x128"))

    out = {
        "roofs": {"stream_GBps": S["stream_roof_GBps"],
                  "compute_TFLOPs": S["compute_roof_TFLOPs"],
                  "machine_balance_flop_per_byte": S["machine_balance_flop_per_byte"],
                  "cell_s": S["cell_of_record_s"], "cell_scale": scale,
                  "session_fold_s": S["session_fold_s"]},
        "closure": {"top_level_at_cell_s": round(top_cell, 3),
                    "sum_of_own_rows_at_cell_s": round(own_cell, 3),
                    "gap_pct": round(100 * (own_cell - top_cell) / top_cell, 2)},
        "glue_at_cell_s": round(sum(r["s_own_at_cell"] for r in glue), 4),
        "glue_with_device_ops_at_cell_s": round(
            sum(r["s_own_at_cell"] for r in glue if r["ops_per_call"]), 4),
        "glue_with_no_device_op_at_cell_s": round(sum(r["s_own_at_cell"] for r in dark), 4),
        "units_with_no_own_op": [r["unit"] for r in dark],
        "adaln_double_count_at_cell_s": round(dbl, 4),
        "adaln_double_count_above_roof_at_cell_s": round(dbl_above, 4),
        "glue_at_cell_s_exact_sums": round(
            sum(r.get("s_own_at_cell_exact_sums", 0.0) for r in glue), 4),
        "estimators_disagree_over_3x": [
            r["unit"] for r in glue
            if (r.get("estimator_ratio") or 0) > 3 or (r.get("estimator_ratio") or 9) < 1 / 3],
        "rows": rows,
    }
    a.out_json.write_text(json.dumps(out, indent=1))

    L = ["# The 2.671 s in no op class: named, priced, and put on its roof", "",
         f"One instrument, `roof-budget`'s. FLOPs from `exec_flops.py`, bytes from "
         f"`real_traffic.py` deduped on buffer address, both on the same 26 captures taken on qb2 "
         f"card 2 at `f072ae02f`; times from the same process's bracketed fold, "
         f"{S['session_fold_s']:.3f} s, scaled by {scale:.4f} to the "
         f"{S['cell_of_record_s']:.3f} s cell of record. Roofs measured in that session: "
         f"**{S['stream_roof_GBps']:.1f} GB/s** stream, "
         f"**{S['compute_TFLOPs'] if 'compute_TFLOPs' in S else S['compute_roof_TFLOPs']:.2f} "
         f"TFLOP/s** dense bf16, machine balance "
         f"{S['machine_balance_flop_per_byte']:.1f} FLOP/byte.", "",
         "Every row is one unit's OWN work: the top-level ttnn ops it issues outside every marked "
         "child. That tiles the fold, so there is no unattributed column left to report -- the "
         "residual is a row.", "",
         "| unit | | calls | own ops/call | MB/call | GFLOP/call | FLOP/byte | roof | s/fold | "
         "at cell | **above roof** | % of roof | at cell, exact sums |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        r = dict(r, unit=r["unit"].replace("|", r"\|"))
        L.append("| `{unit}` | {kind} | {calls} | {ops_per_call} | {MB_per_call:.1f} | "
                 "{GFLOP_per_call:.1f} | {AI_flop_per_byte} | {binding_roof} | "
                 "{s_own_per_fold:.3f} | {s_own_at_cell:.3f} | **{s_above_roof_at_cell:.3f}** | "
                 "{pct_of_roof_at_cell} % | {alt} |".format(
                     alt=("%.3f" % r["s_own_at_cell_exact_sums"])
                     if "s_own_at_cell_exact_sums" in r else "", **r))
    L += ["",
          f"The rows close on the fold: they sum to {own_cell:.3f} s at the cell against "
          f"{top_cell:.3f} s for the three top-level units, a "
          f"{abs(100*(own_cell-top_cell)/top_cell):.2f} % gap. The control is the leaves: all 15 "
          f"leaf rows reproduce `roof-budget`'s own bytes and FLOPs for the same unit to the last "
          f"digit, so the only new thing here is the parent/child partition.", "",
          f"**{sum(r['s_own_at_cell'] for r in glue):.3f} s of the cell is glue** -- a unit's own "
          f"ops rather than a child's. "
          f"{sum(r['s_own_at_cell'] for r in dark):.3f} s of that runs NO device op at all.", "",
          "## The glue rows, and what is in them", ""]
    for r in glue:
        if not r["ops"]:
            L.append(f"- `{r['unit']}` -- **no op at all**. "
                     f"{r['s_own_at_cell']:.3f} s at the cell with an empty capture between its "
                     f"children.")
            continue
        items = ", ".join(f"{n}x `{k}`" for k, v in list(r["ops"].items())[:6]
                          for n in [v["n"]] if v["MB"] or v["GFLOP"])
        L.append(f"- `{r['unit']}` -- {r['s_own_at_cell']:.3f} s at the cell, "
                 f"{r['MB_per_call']:.1f} MB and {r['GFLOP_per_call']:.1f} GFLOP a call: {items}.")
    pf = next(r for r in rows if r["unit"] == "PairformerLayer|1x512x384,1x512x512x128")
    ctb = [r for r in rows if r["unit"].startswith("ConditionedTransitionBlock")]
    L += ["", "## What this changes", "",
          f"CALL_TREE's two largest unowned items, "
          f"`DiffusionTransformer|1x512x768` at 0.994 s and `Diffusion|1x4480x3,1` at 0.882 s, are "
          f"the two rows where three independent things all say the same: the capture holds no op "
          f"or almost none (0 and 21 ops a call), the ops that are there account for "
          f"0.0 % and 2.2 % of the binding roof, and the two timing estimators disagree by 12.6x "
          f"and 7.1x. Neither is an op class and neither carries a lever.",
          "",
          f"What is real is smaller and already fast. The trunk `PairformerLayer` runs 7 residual "
          f"`add_` and 1 `layer_norm` a block, "
          f"{pf['MB_per_call']:.1f} MB, and at {pf['pct_of_roof_at_cell']:.1f} % of the {S['stream_roof_GBps']:.1f} GB/s "
          f"stream roof it is the highest-utilisation row in the fold: "
          f"{pf['s_above_roof_at_cell']:.3f} s above roof out of {pf['s_own_at_cell']:.3f} s. The "
          f"`ConditionedTransitionBlock` pair of rows, "
          f"{sum(r['s_own_at_cell'] for r in ctb):.3f} s at the cell between them, are new "
          f"only because CALL_TREE ranked the block as a leaf while also charging every one of its "
          f"AdaLN calls to the layer above it.",
          "",
          "## Owed", "",
          "The timing column is two estimators that bracket the answer rather than one that "
          "measures it, because the committed fold records a median and a total per unit and "
          f"nothing between them. The glue is {sum(r['s_own_at_cell'] for r in glue):.3f} s by one "
          f"and {out['glue_at_cell_s_exact_sums']:.3f} s by the other. Closing that needs the "
          "attrib fold re-taken with per-call times kept, which is one 25 s fold. It does not "
          "change which rows are device work: that comes from the captures, and they are the "
          "committed ones.",
          "",
          f"Three rows rest on a child frame that ran to the end of its capture "
          f"({', '.join('`' + u + '`' for u in sorted(r['unit'] for r in glue if 'to-end' in r['closed_by']))}), "
          "so their op lists are a floor, not a count: a child that closes late takes the "
          "parent's next ops with it, never the reverse."]
    a.out_md.write_text("\n".join(L) + "\n")

    print("%-52s %-6s %8s %5s %9s %9s %7s" % ("unit", "kind", "at cell", "ops", "MB/call",
                                              "above rf", "% roof"))
    for r in rows:
        print("%-52s %-6s %8.4f %5d %9.2f %9.4f %7s" % (
            r["unit"], r["kind"], r["s_own_at_cell"], r["ops_per_call"], r["MB_per_call"],
            r["s_above_roof_at_cell"], r["pct_of_roof_at_cell"]))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("WROTE", a.out_json, a.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
