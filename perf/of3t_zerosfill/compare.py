#!/usr/bin/env python3
"""What the zero fix is worth on the WALL, and why that is not what the verb table says.

The verb table says `zeros` loses 11.9 s. The wall says the backward gains about 1.0 s. Both
are correct readings of different things, and the gap is the point of this script.

ttnn dispatch is asynchronous. `ttnn.zeros(..., device=)` builds on the host and uploads, which
BLOCKS, so it drains a queue the verbs before it filled, and a per-call stopwatch charges their
device time to it. Take the blocking op away and that device time reappears spread across the
verbs that always owned it. This prints the per-verb self-time deltas between the arms, and the
test is a conservation one: if the smearing story is right, the verbs OTHER than zeros/clone
must grow by roughly what zeros lost, minus the real saving.

The honest per-call price is the queue-drained micro-benchmark in `zerobench.py`, 5.90 ms
against 0.211 ms, because there every rep ends in `synchronize_device` and nothing can hide
behind anything.

    python3 perf/of3t_zerosfill/compare.py
"""
import json
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
ARMS = {"host": ["hist_384_noexact_host.json", "hist_384_noexact_host_r2.json",
                 "hist_384_noexact_host_r3.json"],
        "device": ["hist_384_noexact_device.json", "hist_384_noexact_device_r2.json",
                   "hist_384_noexact_device_r3.json"]}


def load(n):
    return json.loads((OUT / n).read_text())


def main():
    runs = {k: [load(n) for n in v] for k, v in ARMS.items()}
    print("BACKWARD WALL, crop 384, arm noexact, pc card 0 (Blackhole p150a), "
          "three interleaved pairs")
    print(f"{'arm':8} {'runs (s)':28} {'median':>8} {'min':>7} {'max':>7}")
    med = {}
    for k, rs in runs.items():
        xs = [r["backward"]["s"] for r in rs]
        med[k] = statistics.median(xs)
        print(f"{k:8} {str(xs):28} {med[k]:8.2f} {min(xs):7.2f} {max(xs):7.2f}")
    sep = min(runs["host"], key=lambda r: r["backward"]["s"])["backward"]["s"] - \
        max(runs["device"], key=lambda r: r["backward"]["s"])["backward"]["s"]
    print(f"\ndelta on medians   {med['host'] - med['device']:.2f} s   "
          f"{med['host'] / med['device']:.4f}x")
    print(f"separation         every host run is {sep:.2f} s slower than every device run"
          if sep > 0 else f"separation         OVERLAP: {sep:.2f} s")

    # The forward leg is the A/A floor: `grad_zeros` is backward-only (checked by AST in
    # assert_backward_only.py), so the lever cannot touch the forward and its spread across
    # all six runs is this harness's own noise.
    fw = [r["forward"]["s"] for rs in runs.values() for r in rs]
    print(f"\nA/A FLOOR, forward leg (the lever cannot reach it): {fw}")
    print(f"  spread {max(fw) - min(fw):.2f} s over {len(fw)} runs, "
          f"{(max(fw) - min(fw)) / statistics.median(fw) * 100:.1f} % of its median")

    print("\nPER-VERB SELF TIME, host median run -> device median run (s)")
    hm = sorted(runs["host"], key=lambda r: r["backward"]["s"])[1]
    dm = sorted(runs["device"], key=lambda r: r["backward"]["s"])[1]
    h = {r["verb"]: r["self_s"] for r in hm["by_verb"]}
    d = {r["verb"]: r["self_s"] for r in dm["by_verb"]}
    rows = sorted(set(h) | set(d), key=lambda v: -abs(d.get(v, 0) - h.get(v, 0)))
    print(f"{'verb':32} {'host':>8} {'device':>8} {'delta':>8}")
    for v in rows[:12]:
        print(f"{v:32} {h.get(v, 0):8.3f} {d.get(v, 0):8.3f} {d.get(v, 0) - h.get(v, 0):+8.3f}")

    lost = h.get("zeros", 0) - d.get("zeros", 0)
    gained = sum(d.get(v, 0) - h.get(v, 0) for v in set(h) | set(d)
                 if v not in ("zeros", "clone") and d.get(v, 0) > h.get(v, 0))
    tot_h = sum(h.values())
    tot_d = sum(d.values())
    print(f"\nzeros lost                       {lost:8.3f} s")
    print(f"other verbs grew                 {gained:8.3f} s   "
          f"({gained / lost * 100:.0f} % of what zeros lost)")
    print(f"clone (the fix's own cost)       {d.get('clone', 0) - h.get('clone', 0):+8.3f} s")
    print(f"all verb self-time               {tot_h:8.3f} -> {tot_d:8.3f} "
          f"({tot_d - tot_h:+.3f} s)")
    print(f"backward wall                    {hm['backward']['s']:8.2f} -> "
          f"{dm['backward']['s']:.2f} ({dm['backward']['s'] - hm['backward']['s']:+.2f} s)")
    print("\nSo the 11.9 s in the verb table was mostly other verbs' device time draining at "
          "the\nonly blocking op in the neighbourhood. The wall is the number that counts.")

    print("\nCLOCK")
    for k, rs in runs.items():
        for r in rs:
            print(f"  {k:7} {r['env']['aiclk_line']}  loadavg1 {r['env']['loadavg'][0]:.2f}")


if __name__ == "__main__":
    main()
