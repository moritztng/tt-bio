#!/usr/bin/env python3
"""Where the 512 aa fold's device floor actually sits, and which half of it a lever can move.

The floor is sum over ops of calls * max(t_traffic, t_arith), from the committed
`perf/roof_launch/op_census_512.json`. It is a lower bound on device BUSY time because tt-metal
runs one program at a time per device. The fold minus the floor bounds all exposed host time.

The split is the point of this script. `t_traffic` is bytes over a MEASURED DRAM roof and is a
genuine hardware bound. `t_arith` is FLOPs over a per-shape rate from
`shape_roofs_qb2c3_shipped.json` -- the fastest arm MEASURED for that shape on the SHIPPED kernel,
with uncovered FLOPs priced at the fastest measured class so the result stays a floor. So the
arithmetic half is a floor *conditional on today's kernels*, not a hardware floor, and a better
kernel can go under it. The traffic half cannot be beaten, only deleted.

That distinction decides which lever can reach which target, so it is computed rather than assumed.
"""
import json
import subprocess

FOLD_S = 14.881          # c10-bare-baseline median, pinned during-sampled 1350 MHz
F_FIT = 3.9830           # c10-fixed-cost, clock-immune term, +/- 0.1181
OPCENSUS = "origin/main:perf/roof_launch/op_census_512.json"


def main():
    bo = json.loads(subprocess.run(["git", "show", OPCENSUS], capture_output=True,
                                   check=True, text=True).stdout)["by_op"]
    floor = sum(v["s_floor"] for v in bo.values())
    arith = {k: v for k, v in bo.items() if v["s_arith"] > v["s_traffic"] and v["s_floor"] > 0}
    traf = {k: v for k, v in bo.items() if v["s_arith"] <= v["s_traffic"] and v["s_floor"] > 0}
    sa = sum(v["s_floor"] for v in arith.values())
    st = sum(v["s_floor"] for v in traf.values())
    host = FOLD_S - floor

    print(f"fold                       {FOLD_S:8.4f} s   (1350 MHz, during-sampled)")
    print(f"device floor               {floor:8.4f} s   sum calls*max(t_traffic,t_arith), 30 ops")
    print(f"  arithmetic-bound         {sa:8.4f} s   {100*sa/floor:5.1f} %  over {len(arith)} ops"
          f"   <- SOFT: measured shape roofs on today's kernels")
    print(f"  traffic-bound            {st:8.4f} s   {100*st/floor:5.1f} %  over {len(traf)} ops"
          f"   <- HARD: measured DRAM roof, deletable only")
    print(f"exposed host ceiling       {host:8.4f} s   fold - floor")
    print(f"  F (clock-immune, fitted) {F_FIT:8.4f} s   exceeds that room by {F_FIT-host:.4f} s")
    print()
    print("arithmetic-bound ops, which is where any sub-floor lever has to work:")
    for k, v in sorted(arith.items(), key=lambda kv: -kv[1]["s_floor"]):
        print(f"  {k:44} {v['s_floor']:7.4f} s  {v['calls']:9,.0f} calls")
    print()
    # A fold cannot be shorter than its device busy time plus its exposed host time, so any
    # target implies a host budget once the device side is fixed. Byte deletion cannot touch the
    # arithmetic-bound half at all, so `sa` is the best device time it can ever leave -- an absurd
    # bound, since it assumes every traffic-bound second is deleted.
    for tgt in (12.5, 10.0):
        need = max(0.0, floor - tgt)
        budget = tgt - sa
        verdict = (f"needs exposed host < {budget:.4f} s" if budget > 0
                   else "impossible at any host time")
        print(f"target {tgt:4.1f} s: floor must fall {need:.4f} s before host is counted. "
              f"Deleting ALL traffic-bound time leaves {sa:.4f} s of arithmetic floor, so byte "
              f"deletion alone {verdict}.")
    print(f"\nFor scale, the host term is not near zero: the fitted clock-immune F is {F_FIT:.4f} s "
          f"and even the tightest host ceiling here is {host:.4f} s.")


if __name__ == "__main__":
    main()
