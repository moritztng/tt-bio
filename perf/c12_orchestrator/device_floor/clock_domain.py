#!/usr/bin/env python3
"""The device floor of record is an 800 MHz floor, and it is subtracted from a 1350 MHz fold.

`floor_split.py` reports `device floor 12.7062 s` and `exposed host ceiling 2.1748 s = 14.881 -
12.7062`, and passes 10-16 of C12 built on both. The 14.881 s is `c10-bare-baseline`'s median at a
pinned, during-sampled 1350 MHz. The floor is not.

Every rate in that floor is anchored to `roof_budget_512_qb2c2.json`, whose own page states its
session in its first two lines: **qb2 card 2, AICLK 800 MHz, loadavg 28.11**, session fold 24.731 s
rescaled by one scalar 0.7011 to a 17.340 s cell. `true_floor.class_rates` divides every measured
shape rate by the shape-roof session's own `cube4096_TFLOPs` and multiplies by that session's
`compute_roof_TFLOPs` (104.93), and `op_terms` divides bytes by that session's `stream_roof_GBps`
(424.7). So both halves of the floor are expressed at 800 MHz.

`c10-fold-census` measured both roofs on the same part in the session that produced the 14.881 s
fold, pinned and during-sampled at 1350 MHz (`runs/sweep2/budget.json`): cube4096 **115.685
TFLOP/s**, DRAM **442.877 GB/s**. Re-anchoring the same construction on those two numbers -- same
captures, same FLOPs, same bytes, same per-shape arms, same `max` per op -- is what this script does.

It is a DOMAIN correction, not a new measurement, and it is a bracket rather than a point, because
the three ratios between the two sessions disagree by construction:

    AICLK              800 -> 1350 MHz     1.6875x
    dense cube4096     104.93 -> 115.685   1.1025x
    DRAM stream        424.7 -> 442.877    1.0428x

A +68.8 % clock buys +10.2 % on the cube and +4.3 % on DRAM, so no single scalar maps one session
onto the other and the arithmetic half is bracketed by the two corners it could sit at.
`c12-profiled-fold` measures per-op time in situ at a pinned 1350 MHz and supersedes all of this.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

FOLD_S = 14.881          # c10-bare-baseline median, pinned during-sampled 1350 MHz
F_FIT, F_ERR = 3.9830, 0.1181     # c10-fixed-cost, clock-immune term
PERF = Path(__file__).resolve().parents[2]

# roof_budget_512_qb2c2.json, the anchor the committed floor uses. AICLK 800 MHz, loadavg 28.11.
CUBE_800, STREAM_800 = 104.93, 424.7
# c10-fold-census runs/sweep2/budget.json, pinned during-sampled 1350 MHz, known-answer controlled.
CUBE_1350, STREAM_1350 = 115.68461978077023, 442.8767243360721
CENSUS = "origin/wk/c10-fold-census:perf/c10_fold_census/runs/sweep2/budget.json"


def collect(cube_TFLOPs, stream_GBps):
    """Per-op (calls, FLOP, s_traffic, s_arith, s_floor) with the two roofs substituted."""
    sys.path.insert(0, str(PERF / "roof_true"))
    import true_floor as TF

    class A:
        roofs = "shape_roofs_qb2c3_shipped.json"
        eltwise_rate = uncovered_lo = uncovered_hi = None

    R = TF.setup(PERF, A())
    # Same construction, different anchor: re-normalise every shape rate onto the target cube and
    # re-divide bytes by the target stream roof. Nothing else is touched.
    R["RATES"] = TF.class_rates(R["WG"], R["roofs"], cube_TFLOPs * 1e12)
    rates = [r for _l, r in R["RATES"].values()]
    R["lo_rate"], R["hi_rate"] = min(rates), max(rates)
    R["stream"] = stream_GBps * 1e9
    R["eltwise_rate"] = R["lo_rate"]

    per = defaultdict(lambda: defaultdict(float))
    for sig in TF.TOP:
        calls = R["by"][sig]["calls"]
        J = TF.Join(R, sig)
        for i in range(len(J.ops)):
            B, F_mm, F_el, t_tr, t_ar = TF.op_terms(R, J, i)[:5]
            e = per[J.ops[i]["name"]]
            e["calls"] += calls
            e["FLOP"] += calls * (F_mm + F_el)
            e["s_tr"] += calls * t_tr
            e["s_ar"] += calls * t_ar
            e["s_floor"] += calls * max(t_tr, t_ar)
    return per


def split(per):
    ar = [k for k, v in per.items() if v["s_ar"] > v["s_tr"] and v["s_floor"] > 0]
    tr = [k for k, v in per.items() if v["s_ar"] <= v["s_tr"] and v["s_floor"] > 0]
    return (sum(per[k]["s_floor"] for k in ar), sum(per[k]["s_floor"] for k in tr), ar, tr)


def solve(per, ar, tgt, host):
    """Uniform arithmetic multiplier on the arithmetic-bound ops that reaches tgt at this host."""
    rest = sum(v["s_floor"] for k, v in per.items() if k not in ar)

    def device(m):
        return rest + sum(max(per[k]["s_tr"], per[k]["s_ar"] / m) for k in ar)

    need = tgt - host
    if device(float("inf")) > need:
        return None, device(float("inf"))
    lo, hi = 1.0, 1e6
    for _ in range(300):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if device(mid) > need else (lo, mid)
    return hi, device(float("inf"))


def main():
    print(__doc__.strip().split("\n\n")[0])
    print()
    print(f"  the census roofs are read from {CENSUS}")
    print(f"  cube4096 {CUBE_800} -> {CUBE_1350:.3f} TFLOP/s    "
          f"stream {STREAM_800} -> {STREAM_1350:.3f} GB/s")
    print()

    arms = [("800 MHz anchor (the committed floor)", CUBE_800, STREAM_800),
            ("1350 MHz anchor, cube+stream (central)", CUBE_1350, STREAM_1350),
            ("1350 MHz stream only, rates clock-immune", CUBE_800, STREAM_1350)]
    rows = []
    for label, cube, stream in arms:
        per = collect(cube, stream)
        sa, st, ar, tr = split(per)
        floor = sa + st
        rows.append((label, per, ar, floor, sa, st))
        host = FOLD_S - floor
        print(f"{label}")
        print(f"  device floor  {floor:8.4f} s    arithmetic-bound {sa:8.4f} s ({100*sa/floor:4.1f} %,"
              f" {len(ar)} ops)   traffic-bound {st:7.4f} s ({len(tr)} ops)")
        print(f"  host ceiling  {host:8.4f} s    F = {F_FIT:.4f} +/- {F_ERR:.4f} s exceeds it by "
              f"{F_FIT-host:+.4f} s  (band {F_FIT-F_ERR-host:+.4f} to {F_FIT+F_ERR-host:+.4f})")
        for tgt in (12.5, 10.0):
            mark = "UNDER the floor" if tgt < floor else f"leaves {tgt-floor:.4f} s for host"
            print(f"  {tgt:4.1f} s target: {mark}")
        print()

    print("PER-OP, the three arithmetic-bound ops at each anchor")
    base = rows[0][1]
    ops = sorted(rows[1][2], key=lambda k: -rows[1][1][k]["s_floor"])
    print(f"  {'op':24}{'800 MHz':>10}{'1350 MHz':>11}{'delta':>9}{'TFLOP':>9}"
          f"{'TFLOP/s @1350':>15}")
    for k in ops:
        a, b = base[k], rows[1][1][k]
        print(f"  {k.replace('ttnn.',''):24}{a['s_floor']:10.4f}{b['s_floor']:11.4f}"
              f"{b['s_floor']-a['s_floor']:+9.4f}{b['FLOP']/1e12:9.2f}"
              f"{b['FLOP']/b['s_ar']/1e12:15.2f}")

    print("\nREQUIRED ARITHMETIC MULTIPLIER on those ops, at the central 1350 MHz anchor")
    per, ar = rows[1][1], rows[1][2]
    flop = sum(per[k]["FLOP"] for k in ar)
    cur = flop / sum(per[k]["s_ar"] for k in ar) / 1e12
    print(f"  they run at {cur:.2f} TFLOP/s today over {flop/1e12:.2f} TFLOP")
    print(f"  {'target':>7}{'host':>7}{'device':>9}{'rate x':>9}{'TFLOP/s':>10}")
    for tgt in (12.5, 10.0):
        for H in (3.0, 2.0, 1.0):
            m, asym = solve(per, ar, tgt, H)
            if m is None:
                print(f"  {tgt:7.1f}{H:7.2f}{tgt-H:9.4f}   unreachable: asymptote {asym:.4f} s")
            else:
                print(f"  {tgt:7.1f}{H:7.2f}{tgt-H:9.4f}{m:9.2f}{cur*m:10.1f}")
    print(f"  asymptote at infinite arithmetic rate: {solve(per, ar, 0, 0)[1]:.4f} s "
          f"(the three still pay their own traffic)")

    out = {"fold_s": FOLD_S, "anchors": [
        {"label": l, "floor_s": f, "arith_s": sa, "traffic_s": st,
         "host_ceiling_s": FOLD_S - f, "F_minus_host_s": F_FIT - (FOLD_S - f)}
        for l, _p, _a, f, sa, st in rows]}
    p = Path(__file__).with_name("clock_domain_512.json")
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nWROTE {p}")


if __name__ == "__main__":
    main()
