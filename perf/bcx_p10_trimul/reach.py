#!/usr/bin/env python3
"""How many products the lever converts in a REAL round of the shipped configuration.

The round wall and the round's device column both carry the host's contention; the count of
products each route served does not. `round_ab.py` stamps `autograd.DGRAD_2D_STATS` on every
round_start, so the difference between consecutive stamps is that round's own reach. Multiplied
by the per-product delta measured in isolation (`micro2_n275.json`, median of 15 warm synced
reps), that is a round-level figure grounded in two measurements neither of which is a wall.
"""
import json
import sys

# per-product device seconds, qb1 card 1, AICLK 1350, median of 15 warm synced reps,
# perf/bcx_p10_trimul/micro2_n275.json
MICRO = {
    "[1,275,275,512] @ [128,512]^T": {"off": 2.9212e-3, "on": 0.3659e-3},
    "[1,275,275,128] @ [128,128]^T": {"off": 1.2290e-3, "on": 0.2481e-3},
}
# and the device A/B's own per-shape reading in a live block, perf/bcx_p10_trimul/census_ab_n275.json
BLOCK = {"off": 10.723e-3, "on": 2.718e-3, "calls": 6}


def main(path):
    d = json.load(open(path))
    starts = [e for e in d["events"] if e["kind"] == "round_start"]
    stop = [e for e in d["events"] if e["kind"] == "round_stop"]
    end = d["stamp"]["dgrad_2d_stats"]
    rows = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1]["dgrad_2d_before"] if i + 1 < len(starts) else end
        b = s["dgrad_2d_before"]
        rows.append({"round": s["round"], "arm": s.get("arm"),
                     "minimal": nxt["minimal"] - b["minimal"],
                     "fallback": nxt["fallback"] - b["fallback"]})
    print(f'{"rnd":>4}{"arm":>5}{"minimal":>10}{"fallback":>10}{"total":>10}')
    for r in rows:
        print(f'{r["round"]:>4}{r["arm"]:>5}{r["minimal"]:>10}{r["fallback"]:>10}'
              f'{r["minimal"] + r["fallback"]:>10}')
    body = [r for r in rows if r["round"] > 1 and r["minimal"] + r["fallback"] > 0]
    on = [r for r in body if r["arm"] == "on"]
    off = [r for r in body if r["arm"] == "off"]
    if not on:
        print("\nno steady-state `on` round with a reach count")
        return
    per = sorted(r["minimal"] for r in on)
    tot = sorted(r["minimal"] + r["fallback"] for r in on)
    n = per[len(per) // 2]
    t = tot[len(tot) // 2]
    print(f'\nproducts per round in the shipped configuration: {t} total, '
          f'{n} converted ({100 * n / t:.1f} %)')
    print(f'  `off` rounds convert {sorted(r["minimal"] for r in off)} (must be all zero)')
    print(f'  per-round spread of converted: {per}')
    # the two shapes are issued in a fixed 1:2 ratio per trimul (one in-projection dgrad,
    # two out-projection dgrads), and 6 weight transposes ride along per block
    print("\nprojection, per round:")
    blocks = n / BLOCK["calls"]
    d_block = BLOCK["off"] - BLOCK["on"]
    print(f'  blocks-equivalent at 6 converted products a block: {blocks:.0f}')
    print(f'  x the live-block delta {d_block * 1e3:.3f} ms  ->  {blocks * d_block:.3f} s of device')
    print("\n  cross-check against the isolated micro (2 of shape A + 4 of shape B a block):")
    a = MICRO["[1,275,275,512] @ [128,512]^T"]
    b = MICRO["[1,275,275,128] @ [128,128]^T"]
    per_block = 2 * (a["off"] - a["on"]) + 4 * (b["off"] - b["on"])
    print(f'    micro per block {per_block * 1e3:.3f} ms (transposes not counted) -> '
          f'{blocks * per_block:.3f} s of device')


if __name__ == "__main__":
    main(sys.argv[1])
