#!/usr/bin/env python3
"""bcx-p10-devgap: where the composed `hifi` round's device column goes, and what it sums to.

The round meter (`perf/bcx_round/meter.py`) gives the device column as a WALL: the interval
spent inside `EvoformerOnDevice`/`ExtraMsaOnDevice`/`TemplateOnDevice`'s three seams. The
block harness (`perf/bcx_p10_shape/shape.py` cell E, the fold's own program plus the hifi
route) gives, per op family, the card's own seconds and the dispatch loop's, from the same
subtraction `bcx-p10-devmap` used:

    device_i = synced_i - free_i - lambda * calls_i

The two are different quantities and the difference is the point. Write

    C = the measured device column        the seam wall, round meter, median over rounds
    A = sum of device_i, scaled to a round   the card actually executing an attributed op
    F = bytes / the measured DRAM roof       the floor `bcx-p10-bytes` derived
    P = sum of free_i, scaled to a round     the ttnn dispatch loop's host cost

then C - A is time inside the seam with no attributed op running on the card, and A - F is
time the card spent running ops below the roof. Those are the two halves of the gap and
they have completely different levers: C - A is fewer and larger calls, A - F is bytes and
kernel efficiency. C - A is compared against P because an unhidden dispatch loop and a card
idling for some other reason look identical in a wall and are not the same problem.

The multiplicities are READ OFF THE ROUND, not assumed: `bcx-p10-devmap`'s analyze.py hard-codes
2 taped forwards and 1 backward over 48 + 4 blocks, and the shipped five-model pool does not
have to run that. A harness scaled by the wrong multiplicity produces a table that sums by
luck.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import analyze as AN                    # noqa: E402
from perf.bcx_p10_shape import compare as CMP                    # noqa: E402

PHASES = ("taped", "backward", "primal")
MODULES = ("evoformer", "extra_msa", "template")


def med(xs):
    return st.median(xs) if xs else 0.0


def round_column(path, drop_first=True):
    """The device column, split by on-card stack and by seam, per round.

    A round is bounded by two consecutive `sequence_gradients` entries, and the last round has
    no closing boundary, so N requested rounds give N-1 walls. Round 1 carries BindCraft 2's
    jit compile AND the lazy trunk load, so it is reported apart and never medianed in.
    """
    d = json.load(open(path))
    ev, stamp, clk = d["events"], d["stamp"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    rows = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        sg = [e for e in inside if e["phase"] == "sequence_gradients"]
        sg_t = sum(e["dt"] for e in sg)
        dev = [e for e in inside if e["kind"] == "device"]
        cell = {"round": starts[i]["round"], "wall": t1 - t0, "sg": sg_t}
        for m in MODULES:
            for p in PHASES:
                xs = [e for e in dev if e.get("module") == m and e["phase"] == p]
                cell[f"{m}.{p}.s"] = sum(e["dt"] for e in xs)
                cell[f"{m}.{p}.n"] = len(xs)
            cell[f"{m}.s"] = sum(cell[f"{m}.{p}.s"] for p in PHASES)
        cell["device"] = sum(cell[f"{m}.s"] for m in MODULES)
        cell["host_in_sg"] = sg_t - sum(
            e["dt"] for e in dev
            if sg and sg[0]["t0"] <= e["t0"] and e["t1"] <= sg[0]["t1"])
        s = [(c, l) for t, c, l in clk if t0 <= t <= t1]
        cell["aiclk_min"] = min((c for c, _ in s), default=None)
        cell["aiclk_med"] = med(sorted(c for c, _ in s)) if s else None
        cell["load1"] = sum(l for _, l in s) / len(s) if s else None
        rows.append(cell)
    body = rows[1:] if (drop_first and len(rows) > 1) else rows
    summ = {k: med([r[k] for r in body]) for k in rows[0] if k not in ("round",)
            if all(isinstance(r[k], (int, float)) for r in body)}
    summ["n_rounds_medianed"] = len(body)
    summ["round1_wall"] = rows[0]["wall"]
    summ["aiclk_min_over_all"] = min(r["aiclk_min"] for r in body if r["aiclk_min"])
    return stamp, rows, summ


def harness(path, cell_name, args):
    """Cell `cell_name`'s per-family table, per ONE block of each stack, plus its reach."""
    blob = json.load(open(path))
    cell = blob["cells"][cell_name]
    sub = {"records": cell["records"], "ks": cell["ks"],
           "sync_floor_s": blob["sync_floor_s"],
           "flops_fwd_analytic_padded": cell["flops_fwd_analytic_padded"]}
    return blob, cell, AN.per_block(sub), sub


def scale(per_block, mult, flops):
    """Per-block seconds -> per-round, at the multiplicities the ROUND measured.

    `mult[(stack, dir)]` is blocks x calls-per-round for that stack and direction. FLOPs come
    from the harness's analytic count at the padded shape, doubled on the backward because a
    matmul's VJP is two matmuls of the same size -- `bcx-p10-devmap`'s `roofline` convention,
    kept so the two tables can be subtracted.
    """
    fam = collections.defaultdict(lambda: collections.Counter())
    missing = set()
    for (stack, direction, family), v in per_block.items():
        m = mult.get((stack, direction))
        if m is None:
            missing.add((stack, direction))
            continue
        name = CMP.FAMILY.get(family, family)
        fam[name]["device_s"] += v["device"] * m
        fam[name]["dispatch_s"] += v["enqueue"] * m
        fam[name]["synced_s"] += v["synced"] * m
        fam[name]["calls"] += v["calls"] * m
        fam[name]["GB"] += (v["read"] + v["written"]) * m / 1e9
        f = flops.get(family)
        fam[name]["TFLOP"] += (f * m * (1.0 if direction == "fwd" else 2.0) / 1e12) if f else 0.0
    return fam, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, help="round_events.json of the composed arm")
    ap.add_argument("--cells", required=True, help="shape.py cells blob")
    ap.add_argument("--cell", default="E")
    ap.add_argument("--dram", type=float, default=442.3, help="measured DRAM roof, GB/s")
    ap.add_argument("--tflops", type=float, default=85.90)
    ap.add_argument("--bar", type=float, default=10.0, help="sum tolerance, %%")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.ridge = args.tflops * 1e12 / (args.dram * 1e9)

    stamp, rows, R = round_column(args.round)
    blob, cell, pb, sub = harness(args.cells, args.cell, args)

    print("== the round ==")
    print(json.dumps({k: stamp.get(k) for k in
                      ("host", "card", "commit", "triatt_hifi", "extra_msa_on_device",
                       "template_on_device", "exact", "binder_pinned", "seed")}))
    print("reach:", json.dumps({k: stamp.get(k) for k in
                                ("kernel_entry_stats", "fused_hifi_stats",
                                 "extra_msa_swapped", "template_calls")}))
    for r in rows:
        print(" ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}"
                       for k, v in r.items() if not k.endswith(".n")))
    print("\nmedian over rounds 2..%d (round 1 = %.3f s, jit + lazy trunk load, dropped)"
          % (R["n_rounds_medianed"] + 1, R["round1_wall"]))
    print(" round wall %.3f | host_in_sg %.3f | DEVICE COLUMN %.3f "
          "| evo %.3f extra %.3f template %.3f | AICLK med %s min %s | load1 %.1f"
          % (R["wall"], R["host_in_sg"], R["device"], R["evoformer.s"], R["extra_msa.s"],
             R["template.s"], R["aiclk_med"], R["aiclk_min_over_all"], R["load1"]))

    # ---- the multiplicities, read off the round -----------------------------------
    n_evo_blocks = 48
    n_extra_blocks = 4
    mult = {}
    for stack, module, nb in (("evo", "evoformer", n_evo_blocks),
                              ("extra", "extra_msa", n_extra_blocks)):
        fwd = R[f"{module}.taped.n"] + R[f"{module}.primal.n"]
        bwd = R[f"{module}.backward.n"]
        mult[(stack, "fwd")] = nb * fwd
        mult[(stack, "bwd")] = nb * bwd
    print("\nmultiplicities from the round: "
          + " ".join(f"{k[0]}.{k[1]}={v}" for k, v in sorted(mult.items())))

    fam, missing = scale(pb, mult, cell["flops_fwd_analytic_padded"])
    if missing:
        print("UNSCALED (no multiplicity): %s" % sorted(missing))

    A = sum(c["device_s"] for c in fam.values())
    P = sum(c["dispatch_s"] for c in fam.values())
    GB = sum(c["GB"] for c in fam.values())
    TFLOP = sum(c["TFLOP"] for c in fam.values())
    calls = sum(c["calls"] for c in fam.values())
    lam = blob["sync_floor_s"]["median"]
    F = GB * 1e9 / (args.dram * 1e9)
    C = R["device"]
    C_blocks = R["evoformer.s"] + R["extra_msa.s"]
    C_tmpl = R["template.s"]
    G = C_blocks - A

    print("\n== cell %s: the fold's own program, hifi %s ==" % (args.cell, cell.get("hifi")))
    print("warm reach: %s" % json.dumps(cell.get("reach", {})))
    print("the floor is max(bytes/DRAM, FLOP/compute): whichever roof the family actually sits on")
    print("%-32s %8s %8s %8s %7s %7s %8s %8s %8s"
          % ("family", "device_s", "GB", "TFLOP", "%dram", "%comp", "floor_s", "headroom", "disp_s"))
    FL = 0.0
    for name, c in sorted(fam.items(), key=lambda kv: -kv[1]["device_s"]):
        td, tc = c["GB"] * 1e9 / (args.dram * 1e9), c["TFLOP"] * 1e12 / (args.tflops * 1e12)
        floor = max(td, tc)
        FL += floor
        d = c["device_s"]
        c["floor_s"], c["t_dram_s"], c["t_compute_s"] = floor, td, tc
        c["headroom_s"] = d - floor
        print("%-32s %8.3f %8.1f %8.1f %6.1f%% %6.1f%% %8.3f %8.3f %8.3f"
              % (name, d, c["GB"], c["TFLOP"],
                 100 * (c["GB"] / d) / args.dram if d else 0,
                 100 * (c["TFLOP"] / d) / args.tflops if d else 0,
                 floor, c["headroom_s"], c["dispatch_s"]))
    print("%-32s %8.3f %8.1f %8.1f %6.1f%% %6.1f%% %8.3f %8.3f %8.3f"
          % ("TOTAL", A, GB, TFLOP, 100 * (GB / A) / args.dram if A else 0,
             100 * (TFLOP / A) / args.tflops if A else 0, FL, A - FL, P))
    print("ttnn calls a round: %d   sync floor lambda = %.2f us   lambda*calls = %.3f s"
          % (calls, lam * 1e6, lam * calls))

    B = A - F
    close = C_blocks - (A + G)
    table = [
        ("bytes at the measured %.1f GB/s roof" % args.dram, F),
        ("ops below the roof (A - F)", B),
        ("card not running an attributed op (C_blocks - A)", G),
    ]
    print("\n== the device column, summed ==")
    print("%-52s %9s %8s" % ("term", "seconds", "% of C"))
    for name, v in table:
        print("%-52s %9.3f %7.1f%%" % (name, v, 100 * v / C))
    print("%-52s %9.3f %7.1f%%" % ("evoformer + extra-MSA seams (C_blocks)", C_blocks,
                                   100 * C_blocks / C))
    print("%-52s %9.3f %7.1f%%" % ("template seam, not in the block harness", C_tmpl,
                                   100 * C_tmpl / C))
    print("%-52s %9.3f" % ("MEASURED DEVICE COLUMN (C)", C))
    resid = C - (F + B + G + C_tmpl)
    print("%-52s %9.3f %7.1f%%" % ("residual, named or the instrument is unfit", resid,
                                   100 * resid / C))
    verdict = "FITS" if abs(100 * resid / C) <= args.bar else "UNFIT"
    print("sum bar %.0f %% -> %s" % (args.bar, verdict))

    print("\n== the gap G, against the dispatch loop ==")
    print("G = %.3f s over %d ttnn calls = %.1f us a call" % (G, calls, 1e6 * G / calls if calls else 0))
    print("P (dispatch loop, free mode) = %.3f s = %.1f us a call"
          % (P, 1e6 * P / calls if calls else 0))
    print("G - P = %.3f s  (>0 = idle the dispatch loop does not explain; "
          "<=0 = the loop is fully behind the card and G is launch gap)" % (G - P))

    # ---- leg 2: the residual, split the three ways whose levers differ ------------------
    gap = max(G, 0.0)
    print("\n== leg 2: the 4.70 s, split ==")
    print("%-52s %9s %8s" % ("term", "seconds", "% of C"))
    for name, v in (("roofline floor, one roof per family", FL),
                    ("below its own roof (A - floor): kernels and layout", A - FL),
                    ("card not running an attributed op: launch gap + idle", gap),
                    ("template seam, outside the block harness", C_tmpl)):
        print("%-52s %9.3f %7.1f%%" % (name, v, 100 * v / C))
    print("%-52s %9.3f %7.1f%%" % ("over-attribution (A > C_blocks), the instrument's own",
                                   C_blocks - A if A > C_blocks else 0.0,
                                   100 * (C_blocks - A) / C if A > C_blocks else 0.0))
    print("%-52s %9.3f" % ("MEASURED DEVICE COLUMN (C)", C))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"round_stamp": stamp, "rounds": rows, "round_median": R,
             "cell": args.cell, "cell_hifi": cell.get("hifi"), "cell_reach": cell.get("reach"),
             "sync_floor_s": blob["sync_floor_s"], "multiplicities":
                 {f"{k[0]}.{k[1]}": v for k, v in mult.items()},
             "roofs": {"dram_GBs": args.dram, "tflops": args.tflops},
             "families": {k: dict(v) for k, v in fam.items()},
             "leg2": {"roofline_floor_s": FL, "below_roof_s": A - FL,
                      "gap_s": max(G, 0.0), "template_s": C_tmpl,
                      "over_attribution_s": max(A - C_blocks, 0.0)},
             "terms": {"C": C, "C_blocks": C_blocks, "C_template": C_tmpl, "A": A, "F": F,
                       "B": B, "G": G, "P": P, "GB": GB, "calls": calls,
                       "residual": resid, "verdict": verdict}}, indent=1))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
