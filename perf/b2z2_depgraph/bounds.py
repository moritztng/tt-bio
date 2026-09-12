#!/usr/bin/env python3
"""What is any reordering lever worth, given the map? Three bounds, all from measured data.

  REBALANCE   every core waits as little as the luckiest core already does. This is the ceiling on
              re-placing work between cores, and a flat histogram makes it small by construction.
  OVERLAP     every core that is idle between kernels is given useful work instead. This is the
              ceiling on relaxing the op boundary and on overlapping the tail of one op with the
              head of the next.
  DECHAIN     the operand daisy chain is replaced by a multicast: no core waits on its predecessor,
              no core waits to hand off, and the forward happens once instead of once per hop.

All three are generous: they assume the lever is perfect and free.
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H = json.load(open(a.hist))
    C = json.load(open(a.chain))

    span_ms = H["span_ms"]
    ncore = H["cores"]
    grid = H["budget"]["grid_core_ms"]
    wait = H["budget"]["wait_front_core_ms"]
    resv = H["budget"]["reserve_back_core_ms"]

    def excess(by_core_ms):
        v = list(by_core_ms.values())
        return sum(x - min(v) for x in v)

    reb_wait = excess(H["wait_hist"]["by_core_ms"])
    zc = C["zones"]
    resv_core_ms = {k: v / 1e3 for k, v in zc["CB-COMPUTE-RESERVE-BACK"]["by_core_us"].items()}
    reb_resv = excess(resv_core_ms)

    overlap = H["budget"]["idle_between_kernels_core_ms"]

    # DECHAIN, in core-us on the sibling's instrumented log
    def tot(z):
        return zc[z]["total_core_us"] if z in zc else 0.0

    def mn(z):
        return min(v for v in zc[z]["by_core_us"].values() if v > 0) if z in zc else 0.0

    def nfire(z):
        return zc[z]["cores_firing"] if z in zc else 0

    chain_ramp = sum(tot(z) - mn(z) * nfire(z) for z in ("B2Z2-IN0-CHAINWAIT", "B2Z2-IN1-CHAINWAIT"))
    downwait = tot("B2Z2-IN0-DOWNWAIT") + tot("B2Z2-IN1-DOWNWAIT")
    fwd = tot("B2Z2-IN0-FWD") + tot("B2Z2-IN1-FWD")
    # a multicast forwards once instead of once per hop; hops = cores firing / injectors
    inj = nfire("B2Z2-IN0-SRC") + nfire("B2Z2-IN1-SRC")
    fwd_after = fwd * inj / max(1, nfire("B2Z2-IN0-FWD") + nfire("B2Z2-IN1-FWD"))
    dechain_core_us = chain_ramp + downwait + (fwd - fwd_after)
    reader_total = sum(tot(z) for z in zc if z.startswith("B2Z2-") and z != "B2Z2-RDBAR")

    R = {
        "cell": {"arch": "WH", "host": "whglx", "span_ms": span_ms, "cores": ncore,
                 "grid_core_ms": grid},
        "budget_pct_of_grid_core_ms": {
            "in_kernel": H["budget"]["in_kernel_pct"],
            "of_which_math_blocked_on_input": 100 * wait / grid,
            "of_which_math_blocked_on_output_room": 100 * resv / grid,
            "idle_between_kernels": H["budget"]["idle_between_kernels_pct"],
        },
        "REBALANCE": {
            "what": "every core waits as little as the luckiest core already does",
            "input_wait_excess_core_ms": reb_wait,
            "output_room_excess_core_ms": reb_resv,
            "total_core_ms": reb_wait + reb_resv,
            "block_pct": 100 * (reb_wait + reb_resv) / grid,
            "block_ratio": grid / (grid - (reb_wait + reb_resv)),
        },
        "OVERLAP": {
            "what": "every core idle between kernels is given useful work",
            "total_core_ms": overlap,
            "straggler_core_ms": H["gating"]["straggler_surplus_core_ms"],
            "dispatch_gap_core_ms": overlap - H["gating"]["straggler_surplus_core_ms"],
            "block_pct": 100 * overlap / grid,
            "block_ratio": grid / (grid - overlap),
        },
        "DECHAIN": {
            "what": "multicast the operand instead of forwarding it core to core",
            "chain_ramp_core_us": chain_ramp,
            "downwait_core_us": downwait,
            "forward_saved_core_us": fwd - fwd_after,
            "total_core_us": dechain_core_us,
            "instrumented_reader_total_core_us": reader_total,
            "pct_of_instrumented_reader": 100 * dechain_core_us / reader_total,
            "core_ms": dechain_core_us / 1e3,
            "block_pct": 100 * (dechain_core_us / 1e3) / grid,
            "block_ratio": grid / (grid - dechain_core_us / 1e3),
        },
    }
    tot_ms = R["REBALANCE"]["total_core_ms"] + R["OVERLAP"]["total_core_ms"] + R["DECHAIN"]["core_ms"]
    R["UNION_ALL_THREE"] = {
        "core_ms": tot_ms, "block_pct": 100 * tot_ms / grid,
        "block_ratio": grid / (grid - tot_ms),
        "caveat": "additive and generous; the three overlap, so the true union is smaller",
    }
    open(a.out, "w").write(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
