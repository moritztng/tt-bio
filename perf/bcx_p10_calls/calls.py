#!/usr/bin/env python3
"""bcx-p10-calls: the composed round's device column censused by ttnn OP NAME, not by family.

Every census this campaign has taken aggregates through `devmap.py`'s label context, which
tags a call with the Evoformer component that issued it. A `ttnn.permute` inside triangle
multiplication is charged to triangle multiplication and is invisible as a permute. This adds
the op name as a SECOND key beside that context -- `analyze.per_block(..., by_verb=True)` off
the `verb_*` counters `OpTimer` already keeps -- and runs the identical subtraction

    device_i = synced_i - free_i - lambda * calls_i

on it. Same records, same medians, same multiplicities read off the round, so the verb table
and `bcx-p10-devgap`'s family table are two readings of one measurement.

One number does not transfer: `device` is clamped at zero per key, and a verb key is smaller
than a family key, so more of them clamp. `A_verb >= A_family` by construction and the report
prints both with the difference named.
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

#: Ops that move bytes and do no arithmetic. A call in here pays full DRAM traffic for zero
#: FLOPs and another launch gap, which is the whole thesis of this row. `clone`,
#: `to_memory_config` and `reallocate` are copies; `fill_implicit_tile_padding` writes pad.
MOVEMENT = {
    "reshape", "permute", "transpose", "to_layout", "tilize", "untilize", "typecast",
    "concat", "slice", "pad", "clone", "copy", "to_memory_config", "reallocate",
    "fill_implicit_tile_padding", "repeat", "repeat_interleave", "unsqueeze", "squeeze",
    "chunk", "split", "experimental.view", "experimental.nlp_concat_heads",
    "experimental.nlp_create_qkv_heads", "transformer.concatenate_heads",
    "transformer.split_query_key_value_and_split_heads",
}
#: Ops whose seconds buy FLOPs. Everything else (eltwise, normalisation, softmax) is neither
#: pure movement nor a matmul and is reported as its own class rather than forced into one.
ARITHMETIC = {
    "matmul", "linear", "generic_op", "experimental.minimal_matmul",
    "transformer.scaled_dot_product_attention", "sum", "mean", "max",
}


def klass(verb):
    if verb in MOVEMENT:
        return "movement"
    if verb in ARITHMETIC:
        return "arithmetic"
    return "eltwise"


def scale_verbs(pbv, mult):
    """Per-block per-verb -> per-round, at the multiplicities the round measured."""
    rows = collections.defaultdict(lambda: collections.Counter())
    fam_of = collections.defaultdict(lambda: collections.Counter())
    unscaled = set()
    for (stack, direction, family, verb), v in pbv.items():
        m = mult.get((stack, direction))
        if m is None:
            unscaled.add((stack, direction))
            continue
        c = rows[verb]
        c["device_s"] += v["device"] * m
        c["dispatch_s"] += v["enqueue"] * m
        c["calls"] += v["calls"] * m
        c["GB"] += (v["read"] + v["written"]) * m / 1e9
        fam_of[verb][CMP.FAMILY.get(family, family)] += v["device"] * m
    return rows, fam_of, unscaled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, help="round_events.json of the composed arm")
    ap.add_argument("--cells", required=True, help="shape.py cells blob, --verb-records")
    ap.add_argument("--cell", default="E")
    ap.add_argument("--dram", type=float, default=442.3)
    ap.add_argument("--tflops", type=float, default=85.90)
    ap.add_argument("--bar", type=float, default=10.0, help="reconciliation tolerance, %%")
    ap.add_argument("--devgap-column", type=float, default=9.538,
                    help="bcx-p10-devgap's device column, the number this must reconcile to")
    ap.add_argument("--top", type=int, default=28)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp, rounds, R = GAP.round_column(args.round)
    blob = json.load(open(args.cells))
    cell = blob["cells"][args.cell]
    sub = {"records": cell["records"], "ks": cell["ks"],
           "sync_floor_s": blob["sync_floor_s"],
           "flops_fwd_analytic_padded": cell["flops_fwd_analytic_padded"]}

    print("== the round ==")
    print(json.dumps({k: stamp.get(k) for k in
                      ("host", "card", "commit", "triatt_hifi", "extra_msa_on_device",
                       "template_on_device", "exact", "binder_pinned", "seed")}))
    print("reach:", json.dumps({k: stamp.get(k) for k in
                                ("kernel_entry_stats", "fused_hifi_stats",
                                 "extra_msa_swapped", "template_calls")}))
    print("median over rounds 2..%d (round 1 = %.3f s, dropped): wall %.3f | host_in_sg %.3f "
          "| DEVICE COLUMN %.3f | evo %.3f extra %.3f template %.3f | AICLK med %s min %s "
          "| load1 %.1f"
          % (R["n_rounds_medianed"] + 1, R["round1_wall"], R["wall"], R["host_in_sg"],
             R["device"], R["evoformer.s"], R["extra_msa.s"], R["template.s"],
             R["aiclk_med"], R["aiclk_min_over_all"], R["load1"]))

    mult = {}
    for stack, module, nb in (("evo", "evoformer", 48), ("extra", "extra_msa", 4)):
        mult[(stack, "fwd")] = nb * (R[f"{module}.taped.n"] + R[f"{module}.primal.n"])
        mult[(stack, "bwd")] = nb * R[f"{module}.backward.n"]
    print("multiplicities from the round: "
          + " ".join(f"{k[0]}.{k[1]}={v}" for k, v in sorted(mult.items())))

    # ---- the two readings of one measurement --------------------------------------------
    fam_pb = AN.per_block(sub)
    fam, _ = GAP.scale(fam_pb, mult, cell["flops_fwd_analytic_padded"])
    A_fam = sum(c["device_s"] for c in fam.values())

    pbv = AN.per_block(sub, by_verb=True)
    verbs, fam_of, unscaled = scale_verbs(pbv, mult)
    if unscaled:
        print("UNSCALED (no multiplicity): %s" % sorted(unscaled))
    A = sum(c["device_s"] for c in verbs.values())
    GB = sum(c["GB"] for c in verbs.values())
    calls = sum(c["calls"] for c in verbs.values())
    C = R["device"]
    C_blocks = R["evoformer.s"] + R["extra_msa.s"]
    C_tmpl = R["template.s"]

    # ---- leg 1: the table ----------------------------------------------------------------
    print("\n== leg 1: the round's device column by ttnn op name, cell %s (hifi %s) =="
          % (args.cell, cell.get("hifi")))
    print("%-46s %7s %9s %9s %7s  %s"
          % ("ttnn op", "class", "calls", "device_s", "GB", "%dram  called from"))
    ordered = sorted(verbs.items(), key=lambda kv: -kv[1]["device_s"])
    for verb, c in ordered[:args.top]:
        d = c["device_s"]
        src = sorted(fam_of[verb].items(), key=lambda kv: -kv[1])
        tot = sum(v for _, v in src) or 1.0
        where = ", ".join("%s %.0f%%" % (n, 100 * v / tot) for n, v in src[:3])
        print("%-46s %7s %9d %9.3f %9.1f %6.1f%%  %s"
              % (verb, klass(verb), round(c["calls"]), d, c["GB"],
                 100 * (c["GB"] / d) / args.dram if d else 0, where))
    if len(ordered) > args.top:
        rest = ordered[args.top:]
        print("%-46s %7s %9d %9.3f %9.1f" % ("... %d more ops" % len(rest), "",
                                             sum(round(c["calls"]) for _, c in rest),
                                             sum(c["device_s"] for _, c in rest),
                                             sum(c["GB"] for _, c in rest)))
    print("%-46s %7s %9d %9.3f %9.1f %6.1f%%"
          % ("TOTAL (A_verb)", "", round(calls), A, GB,
             100 * (GB / A) / args.dram if A else 0))

    # ---- the fitness rule ------------------------------------------------------------------
    F = GB * 1e9 / (args.dram * 1e9)
    G = C_blocks - A
    resid_col = C - (A + max(G, 0.0) + C_tmpl)
    print("\n== reconciliation, the rule that caught this campaign twice ==")
    print("%-52s %9s" % ("term", "seconds"))
    for name, v in (("A_verb: card running an attributed op, by op name", A),
                    ("A_family: the same subtraction by family (devgap)", A_fam),
                    ("  clamping difference A_verb - A_family, named", A - A_fam),
                    ("card not running an attributed op (C_blocks - A_verb)", max(G, 0.0)),
                    ("template seam, outside the block harness", C_tmpl),
                    ("residual", resid_col),
                    ("MEASURED DEVICE COLUMN (C), this row's round", C),
                    ("bcx-p10-devgap's device column", args.devgap_column)):
        print("%-52s %9.3f" % (name, v))
    off = 100 * (C - args.devgap_column) / args.devgap_column
    print("this round's C against devgap's %.3f s: %+.1f %%  -> %s"
          % (args.devgap_column, off, "AGREES" if abs(off) <= args.bar else "DOES NOT AGREE"))
    fit = 100 * resid_col / C
    print("residual %.3f s = %+.1f %% of C, bar %.0f %% -> %s"
          % (resid_col, fit, args.bar, "FITS" if abs(fit) <= args.bar else "UNFIT"))

    # ---- leg 2: classify and size ----------------------------------------------------------
    cls = collections.defaultdict(lambda: collections.Counter())
    for verb, c in verbs.items():
        k = cls[klass(verb)]
        k["device_s"] += c["device_s"]
        k["GB"] += c["GB"]
        k["calls"] += c["calls"]
        k["ops"] += 1
    print("\n== leg 2: movement against arithmetic ==")
    print("%-12s %5s %10s %7s %10s %7s %10s %7s"
          % ("class", "ops", "calls", "%calls", "GB", "%GB", "device_s", "%dev"))
    for name in ("arithmetic", "eltwise", "movement"):
        c = cls[name]
        print("%-12s %5d %10d %6.1f%% %10.1f %6.1f%% %10.3f %6.1f%%"
              % (name, c["ops"], round(c["calls"]), 100 * c["calls"] / calls,
                 c["GB"], 100 * c["GB"] / GB, c["device_s"], 100 * c["device_s"] / A))
    mv = cls["movement"]
    print("movement is %.1f %% of the %d ttnn calls, %.1f %% of the %.0f GB and %.3f s "
          "= %.1f %% of the %.3f s round"
          % (100 * mv["calls"] / calls, round(calls), 100 * mv["GB"] / GB, GB,
             mv["device_s"], 100 * mv["device_s"] / R["wall"], R["wall"]))

    print("\n== Amdahl on the %.3f s ROUND, not on a block ==" % R["wall"])
    print("%-46s %9s %9s %9s" % ("removing", "device_s", "p", "speedup if removed"))
    top3 = [v for v, _ in ordered[:3]]
    for label, secs in ([(v, verbs[v]["device_s"]) for v in top3]
                        + [("top three together", sum(verbs[v]["device_s"] for v in top3)),
                           ("all movement ops", mv["device_s"])]):
        p = secs / R["wall"]
        print("%-46s %9.3f %9.4f %9.4fx" % (label, secs, p, 1.0 / (1.0 - p)))
    print("(speedup = 1/((1-p)+p/s) at s -> inf, the ceiling; a real lever gets less)")

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"round_stamp": stamp, "round_median": R, "cell": args.cell,
             "cell_reach": cell.get("reach"), "sync_floor_s": blob["sync_floor_s"],
             "multiplicities": {f"{k[0]}.{k[1]}": v for k, v in mult.items()},
             "roofs": {"dram_GBs": args.dram, "tflops": args.tflops},
             "verbs": {v: dict(c, klass=klass(v), called_from=dict(fam_of[v]))
                       for v, c in verbs.items()},
             "classes": {k: dict(v) for k, v in cls.items()},
             "families": {k: dict(v) for k, v in fam.items()},
             "terms": {"C": C, "C_blocks": C_blocks, "C_template": C_tmpl, "A_verb": A,
                       "A_family": A_fam, "F": F, "G": G, "GB": GB, "calls": calls,
                       "residual": resid_col, "round_wall": R["wall"]}}, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
