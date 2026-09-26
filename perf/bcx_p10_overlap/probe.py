#!/usr/bin/env python3
"""Leg 3's mechanism probe: can host work run while a device callback is in flight?

The round cannot answer this on its own. Every host term in it depends on the card's last
answer (leg 2), so a round that fails to overlap has proved a dependency and not an inability.
`--hostload` adds a worker thread whose XLA:CPU work depends on nothing at all. If ITS
iterations still refuse to land inside a device-busy window, the seam is the reason, and every
restructuring of the round loop -- two interleaved trajectories included -- is dead before it
is designed.

The test is a null hypothesis with a number attached. If the worker is oblivious to the card,
the share of its wall that falls inside device-busy windows equals the device-busy share of the
run. If the seam serialises it, that share collapses to zero.

    probe.py <out-dir-with-hostload> [--ref <out-dir-without>]
"""
import argparse
import json
import os
import statistics as st

import split as S


def load(d):
    ev = json.load(open(os.path.join(d, "round_events.json")))
    hl = json.load(open(os.path.join(d, "hostload.json")))
    return ev, hl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--drop-first", type=int, default=1)
    a = ap.parse_args()
    ev, hl = load(a.out)
    E = ev["events"]
    iv = S.device_intervals(E)
    bounds = [b for b in S.rounds(E) if b[0] > a.drop_first]
    t0, t1 = bounds[0][1], bounds[-1][2]
    span = t1 - t0

    dev = S.covered(iv, t0, t1)
    it = [(x, y) for x, y in hl["iters"] if y > t0 and x < t1]
    wall = sum(min(y, t1) - max(x, t0) for x, y in it)
    inside = sum(S.covered(iv, max(x, t0), min(y, t1)) for x, y in it)

    dev_share = dev / span
    got_share = inside / wall if wall else 0.0
    print(f"worker: edge {hl['edge']}, warm {hl['warm_s']} s, "
          f"{len(it)} iterations over the {len(bounds)} timed rounds")
    print(f"  median iteration      {st.median([y - x for x, y in it])*1e3:8.2f} ms")
    print(f"  worker wall           {wall:8.3f} s  ({100*wall/span:.1f} % of the {span:.3f} s span)")
    print(f"  device-busy wall      {dev:8.3f} s  ({100*dev_share:.1f} % of the span)")
    print()
    print(f"  worker wall INSIDE a device-busy window   {inside:8.3f} s")
    print(f"    observed share of the worker's wall     {100*got_share:8.1f} %")
    print(f"    expected if the worker is oblivious     {100*dev_share:8.1f} %")
    print(f"    ratio observed/expected                 {got_share/dev_share:8.3f}")
    print()
    if got_share / dev_share > 0.85:
        print("  => the worker is OBLIVIOUS to the device stream: independent host work DOES")
        print("     run while a device callback is in flight. The seam is not the blocker.")
    elif got_share / dev_share < 0.15:
        print("  => the worker is EXCLUDED from device-busy windows: the seam serialises host")
        print("     work against the card, and no restructuring of the round loop can overlap.")
    else:
        print("  => partial. Neither oblivious nor excluded; report the ratio, do not round it.")


if __name__ == "__main__":
    main()
