#!/usr/bin/env python3
"""Per-shape IN-SITU device seconds for the matmul class, against each shape own roofline.

Why this exists. Every ceiling figure C13 and C14 quote for the matmul class is built on
`op_census_512.json`, a STANDALONE REPLAY of recorded launch keys. `c12-profiled-fold` already
measured the same classes inside a live fold and found the replay overprices them: linear
4.6604 -> 3.39915 s and matmul 1.4790 -> 0.57134 s. So "the class runs at 21.38 TFLOP/s today"
is a replay rate, and the gap to the 34.69 TFLOP/s ceiling is partly a pricing artifact.

This reducer re-opens c12-profiled-fold own committed Tracy captures and splits
MatmulDeviceOperation by SHAPE instead of by class, weights each unit by the fold own integer
call count, and joins the result onto c13 per-shape roofline. It opens no device and reads no
counter: every FLOP and byte figure is a closed form over the shape.

Controls, all of which must pass before the table is quotable:
  C1  class seconds reproduce composed.json MatmulDeviceOperation total
  C2  class call count reproduces the census 110,040 linear+matmul calls
  C3  class FLOPs reproduce c13 131.269 TFLOP, counted twice by independent routes
  C4  the fence/rep controls of the underlying capture still pass
  C5  no shape carries negative slack against its own roof unless its operands are not all DRAM
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

TILE = 32
BF16 = 2

# fold call counts per unit, from c12_profiled_fold/runs/composed.json (the fold own integer
# counts, taken from the counts phase of a live fold).
UNIT_CALLS = {
    "PairformerLayer": 264,
    "MSALayer": 16,
    "DiffusionModule": 200,
    "PairConditioningDevice": 1,
    "RelPosGather": 2,
    "PairAssemblyDevice": 2,
}
CENSUS_CLASS_CALLS = 108608 + 1432     # linear + matmul, c10-fold-census
CENSUS_CLASS_S = 4.6604 + 1.4790       # the replay price every ceiling figure is built on
INSITU_CLASS_S = 3.97049               # composed.json MatmulDeviceOperation
C13_CLASS_TFLOP = 131.26876790784


def tiles(n: int) -> int:
    return (n + TILE - 1) // TILE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c12", type=Path, required=True, help="perf/c12_profiled_fold tree")
    ap.add_argument("--roof", type=Path, required=True, help="c13 class_roof_s3.json")
    ap.add_argument("--run", action="append", default=[], metavar="DIR:UNIT[,UNIT]",
                    help="runs/<name>:<unit order>, repeatable")
    ap.add_argument("--target", type=int, default=1350)
    ap.add_argument("--node", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(a.c12))
    import reduce as R   # noqa: E402  the committed c12 reducer, reused wholesale

    per_shape = defaultdict(lambda: {"calls": 0.0, "s": 0.0, "cores": set(), "fid": set(),
                                     "mem": set(), "units": defaultdict(float)})
    controls = {"units": []}
    for spec in a.run:
        d, _, units = spec.partition(":")
        run = a.c12 / "runs" / d
        order = [u for u in units.split(",") if u]
        rows = R.ops_report(run / "ops_perf_results.csv.gz")
        clock = R.load_clock(run / "clock.jsonl")
        uj = json.loads((run / "unit.json").read_text())
        if "unit_order" in uj:
            umap = [(n, uj["units"][n]) for n in uj["unit_order"]]
        else:
            umap = [(uj["env"].get("unit"), {"reps": int(uj["env"]["reps"]),
                                             "t_region0": uj.get("t_region0"),
                                             "t_region1": uj.get("t_region1")})]
        if order:
            umap = [(n, u) for n, u in umap if n in order]
        full = [n for n, _ in ([(x, uj["units"][x]) for x in uj["unit_order"]]
                               if "unit_order" in uj else umap)]
        for name, u in umap:
            k = full.index(name)
            win, wmeta = R.window(rows, k, len(full))
            ctl = {"run": d, "unit": name, "window": wmeta}
            if win is None:
                controls["units"].append(ctl | {"ok": False})
                print("REFUSED %s/%s: %s" % (d, name, wmeta.get("why")))
                continue
            reps = int(u["reps"])
            ctl["reps"] = reps
            ctl["divisible"] = len(win) % reps == 0
            q = R.qualify(clock, u.get("t_region0"), u.get("t_region1"), a.target, a.node)
            ctl["clock"] = {kk: q.get(kk) for kk in ("ok", "n", "min", "max", "read_errors",
                                                     "max_gap_ms", "uncovered_frac", "why")}
            mm = [r for r in win if r.get("OP CODE") == "MatmulDeviceOperation"]
            ctl["matmul_rows"] = len(mm)
            ctl["matmul_ms_per_call"] = round(sum(R.fnum(r, "DEVICE KERNEL DURATION [ns]")
                                                  for r in mm) / 1e6 / reps, 5)
            ctl["all_ms_per_call"] = round(sum(R.fnum(r, "DEVICE KERNEL DURATION [ns]")
                                               for r in win) / 1e6 / reps, 5)
            ctl["ok"] = bool(ctl["divisible"] and q.get("ok"))
            controls["units"].append(ctl)
            w = UNIT_CALLS[name] / reps
            for r in mm:
                o = R.rshape(r, "OUTPUT_0", logical=True) or R.rshape(r, "OUTPUT_0")
                i0 = R.rshape(r, "INPUT_0", logical=True) or R.rshape(r, "INPUT_0")
                i1 = R.rshape(r, "INPUT_1", logical=True) or R.rshape(r, "INPUT_1")
                if not (o and i0 and i1):
                    continue
                b, m, n = o[0] * o[1], o[2], o[3]
                # in0 is (.., M, K) or (.., K, M): the axis that is not M is K
                kk = i0[3] if i0[2] == m else i0[2]
                e = per_shape[(b, m, kk, n)]
                e["calls"] += w
                e["s"] += R.fnum(r, "DEVICE KERNEL DURATION [ns]") / 1e9 * w
                e["cores"].add(int(R.fnum(r, "CORE COUNT")))
                e["fid"].add(r.get("MATH FIDELITY"))
                e["mem"].add((r.get("INPUT_0_MEMORY"), r.get("INPUT_1_MEMORY"),
                              r.get("OUTPUT_0_MEMORY")))
                e["units"][name] += R.fnum(r, "DEVICE KERNEL DURATION [ns]") / 1e9 * w

    roof = json.loads(a.roof.read_text())
    rmap = {(r["batch"], r["m"], r["k"], r["n"]): r for r in roof["rows"]}
    cube, dram = roof["cube_tflops"], roof["dram_gbs"]

    out = []
    for (b, m, k, n), e in per_shape.items():
        calls = e["calls"]
        flop = 2.0 * b * m * k * n * calls
        flop2 = 2.0 * b * (tiles(m) * tiles(k) * tiles(n)) * TILE ** 3 * calls
        gb = BF16 * (b * tiles(m) * tiles(k) + tiles(k) * tiles(n) + b * tiles(m) * tiles(n)) \
            * TILE ** 2 * calls / 1e9
        t_tr = gb / dram
        t_ar = flop / 1e12 / cube
        r = rmap.get((b, m, k, n))
        t_roof = max(t_tr, t_ar)
        out.append({"batch": b, "m": m, "k": k, "n": n, "calls": round(calls, 1),
                    "insitu_s": round(e["s"], 5),
                    "insitu_tflops": round(flop / 1e12 / e["s"], 3) if e["s"] else None,
                    "tflop": round(flop / 1e12, 5), "tflop_tilewise": round(flop2 / 1e12, 5),
                    "gb": round(gb, 4), "ai": round(flop / (gb * 1e9), 2) if gb else None,
                    "t_traffic_s": round(t_tr, 5), "t_arith_s": round(t_ar, 5),
                    "t_roof_s": round(t_roof, 5), "binding": "traffic" if t_tr > t_ar else "arith",
                    "shape_roof_tflops": round(flop / 1e12 / t_roof, 3) if t_roof else None,
                    "slack_s": round(e["s"] - t_roof, 5),
                    "attain": round(t_roof / e["s"], 4) if e["s"] else None,
                    "cores": sorted(e["cores"]), "fidelity": sorted(x for x in e["fid"] if x),
                    "mem": sorted(set(e["mem"])),
                    "c13_roof_s": r["t_roof_s"] if r else None,
                    "c13_calls": r["calls"] if r else None,
                    "units": {kk: round(v, 5) for kk, v in e["units"].items()}})
    out.sort(key=lambda r: -r["slack_s"])

    # The census key set, canonicalised the same way, so a shape the fold does not issue is
    # named rather than silently absorbed into a class total. `class_roof_s3.json` rows ARE the
    # census key set (c13 merged 34 census entries into 33 canonical shapes).
    insitu_keys = {(r["batch"], r["m"], r["k"], r["n"]) for r in out}
    census_only = [{"shape": list(k), "calls": v["calls"], "tflop": round(v["tflop"], 4),
                    "t_roof_s": v["t_roof_s"]}
                   for k, v in rmap.items() if k not in insitu_keys]
    census_only.sort(key=lambda r: -r["tflop"])
    insitu_only = [{"shape": [r["batch"], r["m"], r["k"], r["n"]], "calls": r["calls"],
                    "tflop": r["tflop"], "insitu_s": r["insitu_s"]}
                   for r in out if (r["batch"], r["m"], r["k"], r["n"]) not in rmap]
    insitu_only.sort(key=lambda r: -r["tflop"])

    tot_s = sum(r["insitu_s"] for r in out)
    tot_calls = sum(r["calls"] for r in out)
    tot_flop = sum(r["tflop"] for r in out)
    tot_flop2 = sum(r["tflop_tilewise"] for r in out)
    tot_roof = sum(r["t_roof_s"] for r in out)
    pos = sum(r["slack_s"] for r in out if r["slack_s"] > 0)
    controls["C1_class_s"] = {"measured": round(tot_s, 5), "composed": INSITU_CLASS_S,
                              "ratio": round(tot_s / INSITU_CLASS_S, 4)}
    controls["C2_class_calls"] = {"measured": round(tot_calls, 1), "census": CENSUS_CLASS_CALLS,
                                  "ratio": round(tot_calls / CENSUS_CLASS_CALLS, 4)}
    controls["C3_class_tflop"] = {"shape_route": round(tot_flop, 5),
                                  "tile_route": round(tot_flop2, 5),
                                  "c13": round(C13_CLASS_TFLOP, 5),
                                  "ratio_to_c13": round(tot_flop / C13_CLASS_TFLOP, 4)}
    controls["C5_negative_slack_shapes"] = [
        {"shape": [r["batch"], r["m"], r["k"], r["n"]], "slack_s": r["slack_s"],
         "mem": r["mem"], "insitu_s": r["insitu_s"], "t_roof_s": r["t_roof_s"]}
        for r in out if r["slack_s"] < 0]
    # CEILING. Per shape the achievable time is its own roof, except where the fold already
    # beats that roof (an operand is not in DRAM, so the closed-form traffic roof does not bound
    # it) -- there the achievable time is what the fold already does. No shape is granted a rate
    # measured on another shape, another batch or another replay configuration.
    ach = sum(min(r["insitu_s"], max(r["t_roof_s"], 0.0)) if r["slack_s"] < 0 else r["t_roof_s"]
              for r in out)
    ach_s = sum((r["insitu_s"] if r["slack_s"] < 0 else r["t_roof_s"]) for r in out)
    summary = {"class_insitu_s": round(tot_s, 5),
               "class_achievable_floor_s": round(ach_s, 5),
               "class_achievable_floor_tflops": round(tot_flop / ach_s, 3),
               "addressable_s": round(tot_s - ach_s, 5),
               "class_insitu_tflops": round(tot_flop / tot_s, 3),
               "class_census_replay_s": round(CENSUS_CLASS_S, 5),
               "class_census_replay_tflops": round(tot_flop / CENSUS_CLASS_S, 3),
               "replay_overprice_x": round(CENSUS_CLASS_S / tot_s, 4),
               "sum_own_roof_s": round(tot_roof, 5),
               "sum_own_roof_tflops": round(tot_flop / tot_roof, 3),
               "positive_slack_s": round(pos, 5),
               "n_shapes": len(out)}
    res = {"summary": summary, "controls": controls, "rows": out,
           "census_only_shapes": census_only, "insitu_only_shapes": insitu_only,
           "cube_tflops": cube, "dram_gbs": dram}
    if a.out:
        a.out.write_text(json.dumps(res, indent=1, default=str))

    print("\nCONTROLS")
    for kk in ("C1_class_s", "C2_class_calls", "C3_class_tflop"):
        print("  %-16s %s" % (kk, controls[kk]))
    for u in controls["units"]:
        print("  unit %-22s %-24s ok=%s reps=%s mm_rows=%s mm_ms/call=%s clock=%s"
              % (u["run"], u["unit"], u.get("ok"), u.get("reps"), u.get("matmul_rows"),
                 u.get("matmul_ms_per_call"),
                 (u.get("clock") or {}).get("ok")))
    print("  C5 shapes under their own roof: %d" % len(controls["C5_negative_slack_shapes"]))
    print("\nCENSUS SHAPES THE FOLD DOES NOT ISSUE (top 8 by TFLOP)")
    for r in census_only[:8]:
        print("  %-24s calls %8.0f  %8.4f TFLOP  roof %7.4f s" %
              (",".join(str(x) for x in r["shape"]), r["calls"], r["tflop"], r["t_roof_s"]))
    print("IN-SITU SHAPES THE CENSUS DOES NOT HAVE (top 8 by TFLOP)")
    for r in insitu_only[:8]:
        print("  %-24s calls %8.0f  %8.4f TFLOP  insitu %7.4f s" %
              (",".join(str(x) for x in r["shape"]), r["calls"], r["tflop"], r["insitu_s"]))
    print("\nSUMMARY", json.dumps(summary, indent=1))
    print("\n%-22s %8s %9s %9s %9s %9s %8s %s"
          % ("shape b,M,K,N", "calls", "insitu_s", "roof_s", "slack_s", "TFLOP/s", "attain", "cores"))
    for r in out:
        print("%-22s %8.0f %9.4f %9.4f %9.4f %9.2f %8.3f %s"
              % ("%d,%d,%d,%d" % (r["batch"], r["m"], r["k"], r["n"]), r["calls"], r["insitu_s"],
                 r["t_roof_s"], r["slack_s"], r["insitu_tflops"] or 0, r["attain"] or 0,
                 r["cores"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
