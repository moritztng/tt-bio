#!/usr/bin/env python3
"""bcx-p10-mmroof leg 2: place every arithmetic class on the roofline and name which roof binds.

`bcx-p10-devgap` proved the compute roof never binds for the round AS A WHOLE -- 3.1 TFLOP/s
against 85.90, 4.3 %. That is a statement about a 9.5 s aggregate, and an aggregate cannot say
which roof binds a class. This does it per class, on the composed round, with the plan in the key.

Three roofs, not two, and the third is the one this row exists to measure:

  DRAM     442.3 GB/s across the whole card.
  COMPUTE  85.90 bf16 TFLOP/s at HiFi4 across the whole card, ridge 194.2 FLOP/byte.
  GRID     both of the above are CARD roofs, and a matmul only reaches a card roof if its plan
           puts work on the whole card. A plan that engages 9 of 110 cores has a ceiling of
           9/110 of the compute roof no matter how much bandwidth is free, and a class sitting
           at 40 % of the DRAM roof while it occupies 8 % of the grid is not bandwidth-starved,
           it is grid-starved. Occupancy is READ OFF THE PLAN -- `per_core_M`, `per_core_N` and
           the factory decide the block decomposition, so the engaged-core count is a property
           of the config the model passes and not something that needs another measurement.

Occupancy per factory, from tt-metal's own block decomposition:

  RMC2D  MatmulMultiCoreReuseMultiCast: M-blocks on grid rows, N-blocks on grid columns.
         cores = ceil(Mt/per_core_M) * ceil(Nt/per_core_N). With `fuse_batch=False` the batch
         is walked SERIALLY inside that grid, so a batched call pays `batch` sequential passes.
  REUSE  MatmulMultiCoreReuse: one output block per core, batch included in the block count.
         cores = batch * ceil(Mt/per_core_M) * ceil(Nt/per_core_N).
  RMC1D  MatmulMultiCoreReuseMultiCast1D: one dimension blocked across the grid.
  none   ttnn's own planner picked, and what it picked is not in the call. Reported as unknown
         rather than guessed: `bcx-p10-mmlay` already measured what these cost (3.04-14.34x) and
         that measurement is the evidence about them, not a model.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_devmap import analyze as AN                    # noqa: E402
from perf.bcx_p10_devgap import gap as GAP                       # noqa: E402
from perf.bcx_p10_mmlay import mmkey as MK                       # noqa: E402
from perf.bcx_p10_mmlay import report as RP                      # noqa: E402
from perf.bcx_p10_mmroof import plankey as PK                    # noqa: E402

TILE = 32


def _tiles(x):
    return -(-int(x) // TILE)


def occupancy(p, grid_cores):
    """(engaged cores, serial batch passes, note) for one parsed class, off its plan.

    Returns `(None, None, why)` when the plan does not determine it, which is every call that
    carries no program config at all.
    """
    f = PK.plan_fields(p["plan"])
    fac = f.get("factory", "none")
    Mt, Nt = _tiles(p["M"]), _tiles(p["N"])
    pm, pn = f.get("M"), f.get("N")
    if fac == "none":
        return None, None, "no plan: ttnn's planner, not in the call"
    if not isinstance(pm, int) or not isinstance(pn, int) or pm <= 0 or pn <= 0:
        return None, None, "plan carries no per_core_M/N"
    # `fuse_batch` folds the batch into M BEFORE the block decomposition, so a plan that carries
    # it has batch*Mt tiles of M to spread and runs one pass, not `batch` of them. Reading
    # `per_core_M` against the unfused Mt is what makes a 104-core plan read as a 1-core plan.
    fb = bool(f.get("fb"))
    if fb:
        Mt *= max(1, p["batch"])
    mb, nb = -(-Mt // pm), -(-Nt // pn)
    if fac == "REUSE":
        cores = (1 if fb else p["batch"]) * mb * nb
        return (min(cores, grid_cores), max(1, math.ceil(cores / grid_cores)),
                "one output block per batch element")
    if fac == "RMC2D":
        serial = 1 if fb else max(1, p["batch"])
        return (min(mb * nb, grid_cores), serial,
                "batch serial inside the grid" if serial > 1 else "batch fused into M")
    if fac == "RMC1D":
        serial = 1 if fb else max(1, p["batch"])
        return min(mb * nb, grid_cores), serial, "1D multicast"
    return min(mb * nb, grid_cores), (1 if fb else max(1, p["batch"])), fac


def binds(intensity, ridge, occ):
    """Which roof a class SHOULD hit. Occupancy caps both card roofs by the same factor, so it
    does not move the ridge -- it moves how much of whichever roof binds is reachable at all."""
    if intensity is None:
        return "?"
    side = "DRAM" if intensity < ridge else "COMPUTE"
    if occ is not None and occ < 0.5:
        return side + "/GRID"
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--cell", default="E")
    ap.add_argument("--dram", type=float, default=442.3)
    ap.add_argument("--tflops", type=float, default=85.90)
    ap.add_argument("--grid-cores", type=int, default=110, help="qb2 Blackhole is 11x10")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-ms", type=float, default=20.0,
                    help="classes costing less than this a round are folded into a tail row")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ridge = args.tflops * 1e12 / (args.dram * 1e9)

    stamp, rounds, R = GAP.round_column(args.round)
    blob = json.load(open(args.cells))
    cell = blob["cells"][args.cell]
    sub = {"records": cell["records"], "ks": cell["ks"],
           "sync_floor_s": blob["sync_floor_s"],
           "flops_fwd_analytic_padded": cell["flops_fwd_analytic_padded"]}

    print("== the composed round this table is scaled to ==")
    print(json.dumps({k: stamp.get(k) for k in
                      ("host", "card", "commit", "triatt_hifi", "triatt_bw_fused", "rne_kernel",
                       "extra_msa_on_device", "template_on_device", "exact", "binder_pinned")}))
    print("median over rounds 2..%d: wall %.3f | DEVICE COLUMN %.3f | AICLK med %s min %s | load1 %.1f"
          % (R["n_rounds_medianed"] + 1, R["wall"], R["device"],
             R["aiclk_med"], R["aiclk_min_over_all"], R["load1"]))
    print("== the cell the classes are measured on, levers %s ==" % json.dumps(blob.get("armed")))
    ld = [r["loadavg"] for r in cell["records"]]
    ck = [r["aiclk"]["median"] for r in cell["records"] if r.get("aiclk")]
    lo = [r["aiclk"]["min"] for r in cell["records"] if r.get("aiclk")]
    print("AICLK during the timed windows: median %s, min %s | loadavg1 %.1f-%.1f"
          % (sorted(ck)[len(ck) // 2] if ck else "?", min(lo) if lo else "?", min(ld), max(ld)))
    print("reach: %s" % json.dumps(blob.get("reach")))
    print("roofs: DRAM %.1f GB/s | %.2f bf16 TFLOP/s HiFi4 | ridge %.1f FLOP/B | grid %d cores"
          % (args.dram, args.tflops, ridge, args.grid_cores))

    mult = {}
    for stack, module, nb in (("evo", "evoformer", 48), ("extra", "extra_msa", 4)):
        mult[(stack, "fwd")] = nb * (R[f"{module}.taped.n"] + R[f"{module}.primal.n"])
        mult[(stack, "bwd")] = nb * R[f"{module}.backward.n"]

    rows, fam_of, unscaled = RP.scale(AN.per_block(sub, by_verb=True), mult)
    if unscaled:
        print("UNSCALED: %s" % sorted(unscaled))

    mm = {k: v for k, v in rows.items() if k.split("#")[0] in MK.MM_VERBS}
    A_all = sum(c["device_s"] for c in rows.values())
    A_mm = sum(c["device_s"] for c in mm.values())

    out_rows = []
    for key, c in mm.items():
        p = PK.parse(key)
        d, gb, n = c["device_s"], c["GB"], c["calls"]
        if not p["shape"] or not d:
            continue
        gbs = gb / d
        fl = p["flop"] * n
        tf = fl / d / 1e12
        inten = fl / (gb * 1e9) if gb else None
        cores, serial, note = occupancy(p, args.grid_cores)
        occ = cores / args.grid_cores if cores else None
        pd, pc = 100 * gbs / args.dram, 100 * tf / args.tflops
        out_rows.append({
            "key": key, "op": p["op"], "shape": p["shape"], "flags": p["flags"],
            "plan": p["plan"], "factory": p["factory"], "ck": p["ck"],
            "a": p.get("a"), "b": p.get("b"), "o": p.get("o"),
            "device_s": d, "GB": gb, "calls": n, "flop": fl,
            "GBs": gbs, "TFLOPs": tf, "pct_dram": pd, "pct_compute": pc,
            "intensity": inten, "cores": cores, "occupancy": occ, "serial_passes": serial,
            "occ_note": note,
            # The reachable ceiling for this class: the binding card roof, scaled by how much of
            # the grid the plan engages. %of_binding_roof against it is the real efficiency.
            "pct_binding_roof": (pd if (inten or 0) < ridge else pc),
            "pct_binding_roof_gridcorrected": ((pd if (inten or 0) < ridge else pc) / occ
                                               if occ else None),
            "binds": binds(inten, ridge, occ),
            "called_from": dict(fam_of[key])})
    out_rows.sort(key=lambda r: -r["device_s"])

    big = [r for r in out_rows if r["device_s"] * 1e3 >= args.min_ms]
    small = [r for r in out_rows if r["device_s"] * 1e3 < args.min_ms]
    print("\n== leg 2: the arithmetic classes costing >= %.0f ms a round (%d of %d classes) =="
          % (args.min_ms, len(big), len(out_rows)))
    print("%-10s %-20s %-3s %6s %7s %7s %6s %6s %7s %6s %5s %6s  %s"
          % ("op", "batch x M x K x N", "fl", "calls", "dev_s", "GB/s", "%dram", "%cmp",
             "FLOP/B", "cores", "occ", "serial", "binds"))
    for r in big[:args.top]:
        print("%-10s %-20s %-3s %6d %7.3f %7.1f %5.1f%% %5.1f%% %7.1f %6s %4s%% %6s  %s"
              % (r["op"].replace("experimental.", "exp."), r["shape"], (r["flags"] or "-")[:3],
                 round(r["calls"]), r["device_s"], r["GBs"], r["pct_dram"], r["pct_compute"],
                 r["intensity"] or 0,
                 r["cores"] if r["cores"] else "?",
                 "%.0f" % (100 * r["occupancy"]) if r["occupancy"] else "?",
                 r["serial_passes"] if r["serial_passes"] else "?", r["binds"]))
    if small:
        print("%-10s %-20s %-3s %6d %7.3f" % ("(%d tail)" % len(small), "", "",
              round(sum(r["calls"] for r in small)), sum(r["device_s"] for r in small)))
    print("%-10s %-20s %-3s %6d %7.3f" % ("FAMILY", "", "",
          round(sum(r["calls"] for r in out_rows)), A_mm))

    print("\n== the plan each class actually carries ==")
    for r in big[:args.top]:
        print("%-10s %-20s  %s | ck %s" % (r["op"].replace("experimental.", "exp."),
                                           r["shape"], r["plan"], r["ck"]))
        src = sorted(r["called_from"].items(), key=lambda kv: -kv[1])
        tot = sum(v for _, v in src) or 1.0
        print("%-32s issued by %s | a %s b %s o %s | %s"
              % ("", ", ".join("%s %.0f%%" % (n, 100 * v / tot) for n, v in src[:2]),
                 r["a"], r["b"], r["o"], r["occ_note"]))

    print("\n== which roof binds, by seconds ==")
    agg = collections.Counter()
    for r in out_rows:
        agg[r["binds"]] += r["device_s"]
    for v, s in agg.most_common():
        print("%-18s %7.3f s  %5.1f %% of the family" % (v, s, 100 * s / A_mm))

    print("\n== the family against each roof ==")
    GB = sum(r["GB"] for r in out_rows)
    FL = sum(r["flop"] for r in out_rows)
    wocc = sum(r["device_s"] * (r["occupancy"] or 0) for r in out_rows)
    known = sum(r["device_s"] for r in out_rows if r["occupancy"] is not None)
    print("seconds %.3f | %.1f GB | %.2f TFLOP | %.1f GB/s = %.1f %% of DRAM | %.2f TFLOP/s = %.1f %% of compute"
          % (A_mm, GB, FL / 1e12, GB / A_mm, 100 * (GB / A_mm) / args.dram,
             FL / A_mm / 1e12, 100 * (FL / A_mm / 1e12) / args.tflops))
    print("family arithmetic intensity %.1f FLOP/B against a ridge of %.1f: the family is %s-bound"
          % (FL / (GB * 1e9), ridge, "DRAM" if FL / (GB * 1e9) < ridge else "COMPUTE"))
    if known:
        print("seconds-weighted grid occupancy over the %.1f %% of the family whose plan says: %.1f %%"
              % (100 * known / A_mm, 100 * wocc / known))
        print("so the grid-corrected DRAM ceiling for that part is %.1f %% of %.1f GB/s = %.1f GB/s"
              % (100 * wocc / known, args.dram, args.dram * wocc / known))

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"round_stamp": stamp, "round_median": R, "cell": args.cell,
             "armed": blob.get("armed"), "reach": blob.get("reach"),
             "roofs": {"dram_GBs": args.dram, "tflops": args.tflops, "ridge": ridge,
                       "grid_cores": args.grid_cores},
             "multiplicities": {f"{k[0]}.{k[1]}": v for k, v in mult.items()},
             "classes": out_rows,
             "totals": {"A_plan": A_all, "A_mm": A_mm, "GB_mm": GB, "flop_mm": FL,
                        "C": R["device"], "round_wall": R["wall"]}}, indent=1))
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
