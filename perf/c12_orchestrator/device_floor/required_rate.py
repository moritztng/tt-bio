#!/usr/bin/env python3
"""What arithmetic rate the three arithmetic-bound ops need for C12 to hit 12.5 s and 10.0 s.

`floor_split.py` showed 74 % of the 512 aa fold's device floor is arithmetic-bound and sits in
three ops, and that the arithmetic half is a floor conditional on TODAY'S kernels rather than a
hardware bound. This turns that into a spec: hold the traffic-bound 20 ops fixed, multiply the
three ops' arithmetic rate by m, and solve for the m each target needs.

Per-op FLOPs come from the executed graph via `roof_true/true_floor.op_terms`, so they are exact
counts, not estimates. The rate each op runs at today is FLOPs / s_arith, where s_arith prices
FLOPs at the fastest arm MEASURED for that shape on the shipped kernel.

Two honest caveats, both stated in the output:
  * exposed host time is NOT measured -- only bounded above at 2.1748 s (fold minus floor). The
    table brackets it, it does not know it. `c12-profiled-fold` is the row that measures it.
  * a uniform multiplier across three ops is a modelling convenience; real wins are uneven.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

FOLD_S = 14.881
PERF = Path(__file__).resolve().parents[2]


def collect():
    sys.path.insert(0, str(PERF / "roof_true"))
    import true_floor as TF

    class A:
        roofs = "shape_roofs_qb2c3_shipped.json"
        eltwise_rate = uncovered_lo = uncovered_hi = None

    R = TF.setup(PERF, A())
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


def main():
    per = collect()
    arith = [k for k, v in per.items() if v["s_ar"] > v["s_tr"] and v["s_floor"] > 0]
    rest = sum(v["s_floor"] for k, v in per.items() if k not in arith)
    flop = sum(per[k]["FLOP"] for k in arith)
    s_ar = sum(per[k]["s_ar"] for k in arith)
    cur = flop / s_ar / 1e12
    floor = sum(v["s_floor"] for v in per.values())

    print(f"{'op':26}{'TFLOP':>10}{'s_arith':>9}{'s_traffic':>11}{'TFLOP/s today':>15}")
    for k in sorted(arith, key=lambda k: -per[k]["s_floor"]):
        v = per[k]
        print(f"{k.replace('ttnn.',''):26}{v['FLOP']/1e12:10.2f}{v['s_ar']:9.4f}"
              f"{v['s_tr']:11.4f}{v['FLOP']/v['s_ar']/1e12:15.2f}")
    print(f"{'aggregate':26}{flop/1e12:10.2f}{s_ar:9.4f}{'':11}{cur:15.2f}")
    def device(m):
        return rest + sum(max(per[k]["s_tr"], per[k]["s_ar"] / m) for k in arith)

    # With infinite arithmetic rate the three still pay their own DRAM traffic, so the asymptote
    # is rest + their t_traffic, NOT rest alone.
    print(f"\ntraffic-bound rest of fold: {rest:.4f} s, untouched by any rate win")
    print(f"device floor today {floor:.4f} s; asymptote at infinite arithmetic rate "
          f"{device(float('inf')):.4f} s (the three still pay their traffic)")

    print(f"\n{'target':>7}{'host':>7}{'device':>9}{'rate x':>9}{'TFLOP/s':>10}")
    for tgt in (12.5, 10.0):
        for H in (2.1748, 1.5, 1.0):
            need = tgt - H
            lo, hi = 1.0, 1e6
            if device(hi) > need:
                print(f"{tgt:7.1f}{H:7.2f}{need:9.4f}   unreachable at any arithmetic rate")
                continue
            for _ in range(300):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if device(mid) > need else (lo, mid)
            print(f"{tgt:7.1f}{H:7.2f}{need:9.4f}{hi:9.2f}{cur*hi:10.1f}")
    print("\nhost is bracketed, NOT measured: the only bound is <= 2.1748 s (fold - floor).")
    print("For scale: the same chip reached 89-94 TFLOP/s on 16384x16384 K=1024 in the session")
    print("that measured these roofs, against 25.64 TFLOP/s aggregate on these three ops.")


if __name__ == "__main__":
    main()
