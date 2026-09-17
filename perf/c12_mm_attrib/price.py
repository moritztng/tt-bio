#!/usr/bin/env python3
"""Price the census's `matmul`-arm keys against the fold's OWN configuration, in cycles.

Three committed instruments, joined on the launch key, nothing new measured here:

  census replay   c10_fold_census sweep2, qb2 node 1, clock pinned and during-sampled at
                  1350 MHz, 107 of 107 intervals qualified. Standalone replay of a recorded
                  (class, out shape, K) with DRAM operands, in two grid arms.
  graph capture   perf/roof_budget/captures, an executed 512 aa fold on qb2 -- operand shapes,
                  buffer types, dtypes, transposes, core_grid, output memory config, and the
                  `unit::` module each op ran inside.
  in-situ profile perf/k10_diffusion/src/ops_perf_step_qb2c0.csv.gz, a Tracy-armed Blackhole
                  capture of the same 512 aa cell on a 110-core p300c -- per-op DEVICE KERNEL
                  DURATION, engaged CORE COUNT, and the program config ttnn RESOLVED.

Why the third instrument decides the question: the census replays the key with ttnn's automatic
config or a `core_grid`, and for four of the six keys the fold pins something else. A replay
number is a hypothesis about the fold, and this is the fold.

CYCLES ARE THE CURRENCY. The profiler's ns are derived at a 1350 MHz constant (the FW cycle
delta over the FW ns is 1.350 to three decimals on every row), so what it reports is cycles; the
seconds below are those cycles at 1350 MHz, which is the clock the census pinned. The
cross-instrument control in `control()` is what says the two are comparable: over the 27 keys
both instruments carry, they agree on the class total to 4.5 %, and the per-key ratios run from
0.31x to 1.78x in BOTH directions -- a clock deficit can only be one-sided.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
from collections import defaultdict
from math import prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "perf/k10_diffusion/src/ops_perf_step_qb2c0.csv.gz"

MHZ = 1350.0                     # the census's pinned clock and the profiler's ns constant
CUBE_TFLOPS = 122.28508737459467  # sweep2 roofs.cube8192_TFLOPs, 8192^3 bf16, 110 cores
DRAM_GBS = 442.8767243360721      # sweep2 roofs.dram_GBs, starved 8192^2 add, streaming
GRID_CORES = 110                  # p300c compute grid the cube roof was measured on
# The best rate any REAL fold shape reached in the census session, at the same clock, over all
# 107 qualified blocks. Used as the argued achievable roof: a dense streaming add and a dense
# cube are not access patterns these keys have, and borrowing them is the error that self-refuted
# the 78.24 TFLOP/s matmul_ceiling.
BEST_FOLD_GBS = 282.01            # matmul|out=1x128x4480|K=512, grid=None
BEST_FOLD_CUBE_FRAC = 94.05 / CUBE_TFLOPS   # matmul|out=16384x16384|K=1024, [11,10], 110 cores


def _shape(x, i):
    v = [x.get(f"INPUT_{i}_{a}[LOGICAL]", "") for a in ("W_PAD", "Z_PAD", "Y_PAD", "X_PAD")]
    if not v[0]:
        return None
    return tuple(int(s.split("[")[0]) for s in v if s)


def _out(x):
    return tuple(int(str(x.get(f"OUTPUT_0_{k}[LOGICAL]", "0")).split("[")[0])
                 for k in ("W_PAD", "Z_PAD", "Y_PAD", "X_PAD"))


def load_profile(path=PROFILE):
    """Matmul rows of the armed capture, grouped by every launch key they could carry.

    A `ttnn.linear` and a `ttnn.matmul` are one `MatmulDeviceOperation` in the profiler, and the
    census key records which Python op issued it, so the group is offered under both arms and the
    census decides. Leading unit dims are stripped progressively because the census key keeps the
    rank the Python call had and the profiler always reports rank 4.
    """
    rows = list(csv.DictReader(io.TextIOWrapper(gzip.open(path))))
    g = defaultdict(list)
    for x in rows:
        if x["OP CODE"] != "MatmulDeviceOperation":
            continue
        a = _shape(x, 0)
        if not a:
            continue
        t = list(_out(x))
        forms = []
        while True:
            forms.append(tuple(t))
            if len(t) > 2 and t[0] == 1:
                t = t[1:]
            else:
                break
        # K is a[-1], or a[-2] when the op took `transpose_a` -- the profiler reports the
        # operand as stored, so both are offered and the census key set disambiguates.
        for arm in ("matmul", "linear"):
            for f in forms:
                for k in {a[-1], a[-2]}:
                    g["%s|out=%s|K=%d" % (arm, "x".join(map(str, f)), k)].append(x)
    return g


def summarise(group):
    d = sorted(float(y["DEVICE KERNEL DURATION [ns]"]) for y in group
               if y["DEVICE KERNEL DURATION [ns]"])
    pc = re.search(r"'program_config':\s*'([^']*)'", group[0]["ATTRIBUTES"])
    return {"n": len(group), "us": d[len(d) // 2] / 1000, "us_min": d[0] / 1000,
            "us_max": d[-1] / 1000,
            "cores": sorted({int(y["CORE COUNT"]) for y in group}),
            "program_config": pc.group(1) if pc else None}


def real_bytes(operands, out_shape):
    """Compulsory DRAM traffic per call from the EXECUTED operands: every DRAM operand read once,
    the result written once. An L1-resident operand moves no DRAM bytes and is excluded, which is
    where the census's reconstruction and the fold part company on one key."""
    b = 0
    for o in operands:
        if o["buffer"] == "DRAM":
            b += prod(o["shape"]) * 2
    return b + prod(out_shape) * 2


def roofs(flops, dram_bytes, cores):
    """Two roofs at two levels of honesty. Dense: the census's own measured cube and streaming
    add, which no key here has the access pattern of. Argued: the arithmetic roof capped to the
    cores the resolved program config actually engages and scaled by the best cube fraction a real
    fold shape reached in that session, and the traffic roof at the best rate a real fold shape
    reached. Both are still upper bounds on the rate, so both give upper bounds on the prize."""
    dense_a = flops / (CUBE_TFLOPS * 1e12)
    dense_t = dram_bytes / (DRAM_GBS * 1e9)
    arg_a = flops / (BEST_FOLD_CUBE_FRAC * (cores / GRID_CORES) * CUBE_TFLOPS * 1e12)
    arg_t = dram_bytes / (BEST_FOLD_GBS * 1e9)
    return {"dense_arith_s": dense_a, "dense_traffic_s": dense_t,
            "dense_s": max(dense_a, dense_t),
            "dense_binding": "traffic" if dense_t > dense_a else "arithmetic",
            "argued_arith_s": arg_a, "argued_traffic_s": arg_t,
            "argued_s": max(arg_a, arg_t),
            "argued_binding": "traffic" if arg_t > arg_a else "arithmetic"}


def control(prof, census):
    """Every key both instruments carry, both directions. The load-bearing instrument control:
    the armed capture is a different session on a different card with no during-sampled clock, so
    it is only usable if it agrees with the pinned replay where the two run the same program."""
    out, tc, ti = [], 0.0, 0.0
    for e in census:
        if e["K"] is None or e["arm"] not in ("matmul", "linear"):
            continue
        k = "%s|out=%s|K=%d" % (e["arm"], "x".join(map(str, e["out"])), e["K"])
        if k not in prof:
            continue
        s = summarise(prof[k])
        out.append({"key": k, "arm": e["arm"], "calls": e["calls"], "grid": e["grid"],
                    "census_us": e["us_per_call"], "insitu_us": s["us"],
                    "ratio": s["us"] / e["us_per_call"], "cores": s["cores"], "n": s["n"]})
        tc += e["fold_s"]
        ti += s["us"] * e["calls"] / 1e6
    out.sort(key=lambda r: -r["census_us"] * r["calls"])
    # The independent leg: `linear` keys are not under test in this row, so their agreement is a
    # control on the instrument rather than a restatement of the result.
    lin = [r for r in out if r["arm"] == "linear"]
    lc = sum(e["fold_s"] for e in census
             if e["arm"] == "linear" and any(r["key"].endswith("|K=%d" % e["K"])
                                             and r["census_us"] == e["us_per_call"] for r in lin))
    li = sum(r["insitu_us"] * r["calls"] / 1e6 for r in lin)
    return {"keys": len(out), "census_fold_s": tc, "insitu_fold_s": ti,
            "ratio": ti / tc if tc else None,
            "over_unit": sum(1 for r in out if r["ratio"] > 1),
            "under_unit": sum(1 for r in out if r["ratio"] <= 1),
            "linear_only": {"keys": len(lin), "census_fold_s": lc, "insitu_fold_s": li,
                            "ratio": li / lc if lc else None,
                            "within_10pct": sum(1 for r in lin if 0.9 <= r["ratio"] <= 1.1),
                            "over_unit": sum(1 for r in lin if r["ratio"] > 1)},
            "rows": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=Path, required=True, help="c10 sweep2 budget.json")
    ap.add_argument("--replay", type=Path, required=True, help="c10 sweep2 replay.json")
    ap.add_argument("--sites", type=Path,
                    default=Path(__file__).resolve().parent / "mm_sites.json.gz")
    ap.add_argument("--shapes", type=Path, default=ROOT / "perf/roof_launch/fold_shapes.json",
                    help="the fold's own call census -- the authority on calls, and the place a "
                         "key the budget dropped is still visible")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "priced.json")
    a = ap.parse_args()

    census = json.load(open(a.budget))["keys"]
    replay = json.load(open(a.replay))["rows"]
    op = gzip.open if a.sites.name.endswith(".gz") else open
    sites = json.load(op(a.sites, "rt"))["rows"]
    prof = load_profile()

    by_key = {r["key"]: r for r in sites if r["key"]}
    arms = defaultdict(dict)
    for r in replay:
        if r["arm"] in ("matmul", "linear"):
            arms[r["key"]]["default" if r["grid"] is None else "grid_11x10"] = r

    rows = []
    for e in census:
        if e["arm"] != "matmul":
            continue
        key = "matmul|out=%s|K=%d" % ("x".join(map(str, e["out"])), e["K"])
        cap, p = by_key.get(key), prof.get(key)
        if not cap or not p:
            rows.append({"key": key, "attributed": False})
            continue
        s = summarise(p)
        rb = real_bytes(cap["operands"], cap["out_shape"])
        R = roofs(e["TFLOP"] / e["calls"] * 1e12, rb, max(s["cores"]))
        insitu_s = s["us"] * e["calls"] / 1e6
        rows.append({
            "key": key, "attributed": True, "calls": e["calls"],
            "census_arm_booked": "default" if e["grid"] is None else "grid_11x10",
            "census_us": e["us_per_call"], "census_fold_s": e["fold_s"],
            "census_Mcycles": e["Mcycles"], "census_min_bytes": e["GB"] / e["calls"] * 1e9,
            "census_above_roof_s": e["above_roof_s"],
            "replay_default_us": arms[key].get("default", {}).get("s_per_call", 0) * 1e6,
            "replay_grid_us": arms[key].get("grid_11x10", {}).get("s_per_call", 0) * 1e6,
            "real_bytes": rb, "bytes_ratio": (e["GB"] / e["calls"] * 1e9) / rb,
            "insitu_us": s["us"], "insitu_spread": [s["us_min"], s["us_max"]],
            "insitu_n": s["n"], "cores": s["cores"],
            "insitu_fold_s": insitu_s,
            "insitu_fold_Mcycles": insitu_s * MHZ,
            "census_minus_insitu_s": e["fold_s"] - insitu_s,
            "operands": [(o["shape"], o["buffer"], o["dtype"]) for o in cap["operands"]],
            "core_grid": cap["core_grid"], "out_mem": cap["out_mem"],
            "transpose_a": cap["transpose_a"], "transpose_b": cap["transpose_b"],
            "resolved_program_config": s["program_config"],
            "compute_kernel": cap["compute_kernel"], "unit": cap["unit"],
            **R,
            "prize_dense_s": max(0.0, (s["us"] / 1e6 - R["dense_s"])) * e["calls"],
            "prize_argued_s": max(0.0, (s["us"] / 1e6 - R["argued_s"])) * e["calls"],
        })
    rows.sort(key=lambda r: -(r.get("insitu_fold_s") or 0))

    # Keys the fold issues, the budget never priced. `op_census.py` records K=None whenever the
    # transpose flag moves the contraction off the last axis, and `census.py` refuses an arm it
    # cannot count FLOPs for, so the key leaves the budget silently. The capture has the operands.
    unpriced = []
    for sh in json.load(open(a.shapes)):
        if sh["arm"] != "matmul" or sh["K"] is not None:
            continue
        out = tuple(sh["shape"])
        cap = next((c for c in sites
                    if c["name"] == "ttnn.matmul" and tuple(c["out_shape"] or ()) == out), None)
        if not cap:
            unpriced.append({"out": list(out), "calls": sh["calls"], "attributed": False})
            continue
        key = "matmul|out=%s|K=%d" % ("x".join(map(str, out)), cap["K"])
        p2 = prof.get(key)
        if not p2:
            unpriced.append({"key": key, "calls": sh["calls"], "attributed": False})
            continue
        s2 = summarise(p2)
        flops = 2 * prod(out) * cap["K"]
        rb = real_bytes(cap["operands"], cap["out_shape"])
        R = roofs(flops, rb, max(s2["cores"]))
        ins = s2["us"] * sh["calls"] / 1e6
        unpriced.append({"key": key, "attributed": True, "calls": sh["calls"],
                         "census_fold_s": 0.0, "insitu_us": s2["us"], "insitu_fold_s": ins,
                         "insitu_fold_Mcycles": ins * MHZ, "cores": s2["cores"],
                         "flops_per_call": flops, "real_bytes": rb,
                         "operands": [(o["shape"], o["buffer"], o["dtype"]) for o in cap["operands"]],
                         "core_grid": cap["core_grid"], "transpose_a": cap["transpose_a"],
                         "transpose_b": cap["transpose_b"], "unit": cap["unit"],
                         "resolved_program_config": s2["program_config"], **R,
                         "prize_dense_s": max(0.0, s2["us"] / 1e6 - R["dense_s"]) * sh["calls"],
                         "prize_argued_s": max(0.0, s2["us"] / 1e6 - R["argued_s"]) * sh["calls"]})
    doc = {"clock_MHz": MHZ, "roofs": {"cube_TFLOPs": CUBE_TFLOPS, "dram_GBs": DRAM_GBS,
                                       "best_fold_GBs": BEST_FOLD_GBS,
                                       "best_fold_cube_frac": BEST_FOLD_CUBE_FRAC},
           "control": control(prof, census), "keys": rows, "unpriced_by_census": unpriced}
    a.out.write_text(json.dumps(doc, indent=1))

    c = doc["control"]
    print("CONTROL  %d keys both instruments carry: census %.4f s vs in-situ %.4f s, ratio %.3f"
          % (c["keys"], c["census_fold_s"], c["insitu_fold_s"], c["ratio"]))
    print("         %d keys in-situ slower, %d faster -- a clock deficit is one-sided, this is not"
          % (c["over_unit"], c["under_unit"]))
    L = c["linear_only"]
    print("         independent leg (%d `linear` keys, none under test here): census %.4f s vs "
          "in-situ %.4f s, ratio %.3f, %d within 10 %%, %d slower"
          % (L["keys"], L["census_fold_s"], L["insitu_fold_s"], L["ratio"], L["within_10pct"],
             L["over_unit"]))
    print("         hand-off: `linear` keys whose two instruments differ by over 0.05 fold s")
    for r in c["rows"]:
        if r["arm"] != "linear":
            continue
        d = (r["insitu_us"] - r["census_us"]) * r["calls"] / 1e6
        if abs(d) > 0.05:
            print("           %-34s %6.0f calls %9.2f -> %9.2f us  %+.4f s" % (
                r["key"].replace("linear|out=", ""), r["calls"], r["census_us"], r["insitu_us"], d))
    print()
    hdr = ("key", "calls", "arm", "cen_us", "situ_us", "cen_s", "situ_s", "delta_s", "cores",
           "B/B", "dense", "argued")
    print("%-31s %5s %-10s %9s %9s %7s %7s %8s %5s %5s %7s %7s" % hdr)
    tc = ti = pd = pa = 0.0
    for r in rows:
        if not r["attributed"]:
            print("%-31s  NOT ATTRIBUTED" % r["key"]); continue
        print("%-31s %5.0f %-10s %9.2f %9.2f %7.4f %7.4f %+8.4f %5d %5.2f %7.4f %7.4f" % (
            r["key"].replace("matmul|out=", "").replace("|K=", " K"), r["calls"],
            r["census_arm_booked"], r["census_us"], r["insitu_us"], r["census_fold_s"],
            r["insitu_fold_s"], -r["census_minus_insitu_s"], max(r["cores"]), r["bytes_ratio"],
            r["prize_dense_s"], r["prize_argued_s"]))
        tc += r["census_fold_s"]; ti += r["insitu_fold_s"]
        pd += r["prize_dense_s"]; pa += r["prize_argued_s"]
    print("%-31s %5s %-10s %9s %9s %7.4f %7.4f %+8.4f %5s %5s %7.4f %7.4f"
          % ("TOTAL", "", "", "", "", tc, ti, ti - tc, "", "", pd, pa))
    for u in unpriced:
        if u.get("attributed"):
            print("%-31s %5.0f %-10s %9s %9.2f %7.4f %7.4f %+8.4f %5d %5s %7.4f %7.4f" % (
                u["key"].replace("matmul|out=", "").replace("|K=", " K"), u["calls"],
                "UNPRICED", "-", u["insitu_us"], 0.0, u["insitu_fold_s"], u["insitu_fold_s"],
                max(u["cores"]), "-", u["prize_dense_s"], u["prize_argued_s"]))
        else:
            print("%-31s  UNPRICED BY CENSUS, not attributed" % str(u.get("key") or u["out"]))
    print("\nin cycles at %g MHz: census %.1f Mc, in-situ %.1f Mc, prize <= %.1f Mc (dense) / "
          "%.1f Mc (argued)" % (MHZ, tc * MHZ, ti * MHZ, pd * MHZ, pa * MHZ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
