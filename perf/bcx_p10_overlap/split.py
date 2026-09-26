#!/usr/bin/env python3
"""Leg 1 of `bcx-p10-overlap`: for every wall millisecond of a round, is the card's stream
busy and is a host thread running.

Two independent instruments, deliberately not derived from each other:

* the DEVICE side is exact, off the meter's own callback timestamps -- the same intervals
  every other PERF10 row calls the device column, so this row's split is arithmetically
  comparable to `bcx-p10-hostfloor`'s gap table rather than a second opinion about it;
* the HOST side is measured CPU, `time.process_time_ns()` summed over every thread of the
  process, differenced between consecutive 2 ms samples. It says "a host thread ran", not
  "the main thread was not inside a callback", which is the distinction the whole row turns
  on: during a device callback this process burns ~1 core in the ttnn dispatch loop
  (`bcx-p10-hostfloor`), and that host work is ALREADY hidden under the card.

The number the row is funded by is `host_busy_dev_idle`: wall where the card's stream is
empty and a host thread is computing. It is the ceiling on what overlapping the two columns
can return, and nothing above it is reachable by restructuring alone.

    split.py <out-dir> [--cores-thresh 0.5]
"""
import argparse
import json
import os
import statistics as st


def load(d):
    ev = json.load(open(os.path.join(d, "round_events.json")))
    tl = json.load(open(os.path.join(d, "timeline.json")))
    return ev, tl


def device_intervals(events):
    """Every device callback as [t0, t1), merged, so overlapping ones are not double counted.

    They should not overlap -- XLA:CPU issues them from one thread in a sequential schedule --
    and `dev_max_depth` in the timeline blob is the direct check on that. Merging anyway means
    the device column here cannot exceed the wall by construction.
    """
    iv = sorted((e["t0"], e["t1"]) for e in events if e["kind"] == "device")
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(iv, a, b):
    """Seconds of [a, b) covered by the merged interval list `iv`."""
    tot = 0.0
    for x, y in iv:
        if y <= a:
            continue
        if x >= b:
            break
        tot += min(y, b) - max(x, a)
    return tot


def rounds(events):
    starts = [e for e in events if e["kind"] == "round_start"]
    stops = [e for e in events if e["kind"] == "round_stop"]
    bounds = []
    for i, s in enumerate(starts):
        end = starts[i + 1]["t0"] if i + 1 < len(starts) else (
            stops[0]["t0"] if stops else None)
        if end is not None:
            bounds.append((s["round"], s["t0"], end))
    return bounds


def split_round(iv, samp, t0, t1, thresh):
    """The four-way split of [t0, t1), plus the threshold-free CPU integral."""
    ts, cpu = samp
    acc = dict(wall=t1 - t0, dev=0.0, host=0.0, both=0.0, host_dev_idle=0.0,
               neither=0.0, cpu_s=0.0, cpu_s_dev_idle=0.0, cpu_s_dev_busy=0.0, n=0)
    for i in range(len(ts) - 1):
        a, b = ts[i], ts[i + 1]
        if b <= t0 or a >= t1:
            continue
        a, b = max(a, t0), min(b, t1)
        dw = b - a
        if dw <= 0:
            continue
        dc = (cpu[i + 1] - cpu[i]) / 1e6          # process CPU seconds in this interval
        cores = dc / dw if dw > 0 else 0.0
        dv = covered(iv, a, b)                    # device-busy seconds in this interval
        idle = dw - dv
        busy = cores >= thresh
        acc["n"] += 1
        acc["dev"] += dv
        acc["cpu_s"] += dc
        acc["cpu_s_dev_idle"] += dc * (idle / dw)
        acc["cpu_s_dev_busy"] += dc * (dv / dw)
        if busy:
            acc["host"] += dw
            acc["both"] += dv
            acc["host_dev_idle"] += idle
        else:
            acc["neither"] += idle
    return acc


def gaps(iv, samp, t0, t1):
    """Every device-idle stretch inside the round, with the host cores burned in it."""
    ts, cpu = samp
    edges = [t0] + [x for a, b in iv if t0 < a < t1 or t0 < b < t1 for x in (a, b)] + [t1]
    edges = sorted(set(e for e in edges if t0 <= e <= t1))
    out = []
    for a, b in zip(edges, edges[1:]):
        if covered(iv, a, b) > (b - a) * 0.5:
            continue                              # a device stretch, not a gap
        dc = 0.0
        for i in range(len(ts) - 1):
            x, y = ts[i], ts[i + 1]
            if y <= a or x >= b:
                continue
            f = (min(y, b) - max(x, a)) / (y - x)
            dc += (cpu[i + 1] - cpu[i]) / 1e6 * f
        out.append({"t": round(a - t0, 4), "wall": round(b - a, 4),
                    "cpu_s": round(dc, 4),
                    "cores": round(dc / (b - a), 2) if b > a else 0.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cores-thresh", type=float, default=0.5)
    ap.add_argument("--drop-first", type=int, default=1)
    a = ap.parse_args()
    ev, tl = load(a.out)
    events = ev["events"]
    iv = device_intervals(events)
    ts = [tl["t0"] + u / 1e6 for u in tl["t_us"]]
    samp = (ts, tl["cpu_us"])

    dts = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
    print(f"sampler: n={tl['n']} dt_ms nominal {tl['dt_ms']} "
          f"median {dts[len(dts)//2]*1e3:.3f} p99 {dts[int(len(dts)*0.99)]*1e3:.3f} "
          f"max {dts[-1]*1e3:.1f} | self_cpu {tl['self_cpu_s']} s "
          f"| dev_max_depth {tl['dev_max_depth']}")
    print(f"device callbacks: {sum(1 for e in events if e['kind']=='device')} "
          f"merged into {len(iv)} intervals")
    print()
    hdr = ("round", "wall", "dev", "host", "both", "hostONLY", "neither", "cpu_s", "cpuIDLE")
    print("%6s %8s %8s %8s %8s %9s %8s %8s %8s" % hdr)
    keep = []
    for rnd, t0, t1 in rounds(events):
        s = split_round(iv, samp, t0, t1, a.cores_thresh)
        if rnd > a.drop_first:
            keep.append(s)
        print("%6d %8.3f %8.3f %8.3f %8.3f %9.3f %8.3f %8.3f %8.3f"
              % (rnd, s["wall"], s["dev"], s["host"], s["both"],
                 s["host_dev_idle"], s["neither"], s["cpu_s"], s["cpu_s_dev_idle"]))
    if not keep:
        return
    med = {k: st.median([s[k] for s in keep]) for k in keep[0] if k != "n"}
    print()
    print(f"MEDIAN over {len(keep)} timed rounds, cores threshold {a.cores_thresh}:")
    for k in ("wall", "dev", "host", "both", "host_dev_idle", "neither",
              "cpu_s", "cpu_s_dev_idle", "cpu_s_dev_busy"):
        print(f"  {k:<18} {med[k]:8.3f}")
    print(f"  {'device idle wall':<18} {med['wall'] - med['dev']:8.3f}")
    print(f"  {'HOST HIDDEN NOW':<18} {med['both']:8.3f}  "
          f"(host-busy wall already under the card)")
    print(f"  {'OVERLAP CEILING':<18} {med['host_dev_idle']:8.3f}  "
          f"({100*med['host_dev_idle']/med['wall']:.1f} % of the round)")

    # A gap table on the median round, so leg 2 has something to name.
    mid = sorted(keep, key=lambda s: s["wall"])[len(keep) // 2]
    for rnd, t0, t1 in rounds(events):
        s = split_round(iv, samp, t0, t1, a.cores_thresh)
        if abs(s["wall"] - mid["wall"]) < 1e-9:
            print(f"\ngaps of round {rnd} (wall {s['wall']:.3f}):")
            print("%9s %8s %8s %7s" % ("t", "wall", "cpu_s", "cores"))
            for g in gaps(iv, samp, t0, t1):
                print("%9.3f %8.3f %8.3f %7.2f"
                      % (g["t"], g["wall"], g["cpu_s"], g["cores"]))
            break


if __name__ == "__main__":
    main()
