#!/usr/bin/env python3
"""Leg 5: the pair-transpose dispatch gate at the ROUND, device and host as two numbers.

One row per timed round: device seconds (the sum of the round's `device` events), round wall,
host = wall - device, the AICLK sampled DURING that round's own window, loadavg1, and the
round's OWN reach differenced from the boundary counters. Round 1 of every process is dropped
as that process's compile.

`bcx-p10-trimove` is why this exists rather than an extrapolation: its per-op census said
+0.213 s a round and the round said 0.933x. An op-level ratio does not transfer.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys


def rounds(path):
    d = json.load(open(path))
    ev = d["events"]
    clk = d.get("aiclk", [])
    dev, span = {}, {}
    for e in ev:
        if e["kind"] == "device":
            dev[e["round"]] = dev.get(e["round"], 0.0) + e["dt"]
        elif e["kind"] in ("predictor", "optimizer"):
            t0, t1 = e["t0"], e["t1"]
            a, b = span.get(e["round"], (t0, t1))
            span[e["round"]] = (min(a, t0), max(b, t1))
    # boundary counters -> each round's own reach
    bnd = {}
    for e in ev:
        if e["kind"] in ("round_start", "round_stop"):
            bnd[e["round"]] = dict(e.get("reach", {}))
    out = []
    for r in sorted(span):
        t0, t1 = span[r]
        cs = [c for t, c, _ in clk if t0 <= t <= t1]
        ls = [l for t, _, l in clk if t0 <= t <= t1]
        nxt = bnd.get(r + 1, {})
        cur = bnd.get(r, {})
        reach = {k: nxt.get(k, 0) - cur.get(k, 0) for k in cur
                 if isinstance(cur.get(k), int)}
        out.append({
            "round": r, "device": round(dev.get(r, 0.0), 3),
            "wall": round(t1 - t0, 3), "host": round(t1 - t0 - dev.get(r, 0.0), 3),
            "aiclk_med": (sorted(cs)[len(cs) // 2] if cs else None),
            "aiclk_min": (min(cs) if cs else None), "aiclk_n": len(cs),
            "load1": round(statistics.median(ls), 1) if ls else None,
            "pt_served": reach.get("pt_rm_served"), "pt_declined": reach.get("pt_rm_declined"),
        })
    return d["stamp"], out


def main():
    root = pathlib.Path(sys.argv[1])
    arms = {}
    for p in sorted(root.glob("ab_*/round_events.json")):
        stamp, rs = rounds(p)
        arm = "on" if stamp.get("pt_rm_min_c") else "off"
        warm = [r for r in rs if r["round"] > 1]          # round 1 = this process's compile
        arms.setdefault(arm, []).append((p.parent.name, stamp, warm))

    print(f"{'round':>6}{'arm':>5}{'device':>9}{'wall':>9}{'host':>8}"
          f"{'AICLK m/min/n':>16}{'load1':>7}{'pt served/decl':>16}  process")
    allrows = {}
    for arm, procs in sorted(arms.items()):
        for name, stamp, warm in procs:
            for r in warm:
                clk = f"{r['aiclk_med']}/{r['aiclk_min']}/{r['aiclk_n']}"
                rch = f"{r['pt_served']}/{r['pt_declined']}"
                print(f"{r['round']:>6}{arm:>5}{r['device']:>9.3f}{r['wall']:>9.3f}"
                      f"{r['host']:>8.3f}{clk:>16}{r['load1']:>7}{rch:>16}  {name}")
            allrows.setdefault(arm, []).extend(warm)

    print()
    print(f"{'arm':<6}{'n':>4}{'device med':>12}{'wall med':>11}{'host med':>10}"
          f"{'pt served/round':>18}{'pt declined/round':>19}")
    med = {}
    for arm in sorted(allrows):
        rs = allrows[arm]
        med[arm] = {k: statistics.median([r[k] for r in rs]) for k in ("device", "wall", "host")}
        print(f"{arm:<6}{len(rs):>4}{med[arm]['device']:>12.3f}{med[arm]['wall']:>11.3f}"
              f"{med[arm]['host']:>10.3f}"
              f"{statistics.median([r['pt_served'] or 0 for r in rs]):>18.0f}"
              f"{statistics.median([r['pt_declined'] or 0 for r in rs]):>19.0f}")
    if "off" in med and "on" in med:
        print()
        for k in ("device", "wall", "host"):
            a, b = med["off"][k], med["on"][k]
            print(f"  {k:<7} off {a:7.3f} -> on {b:7.3f}   {a/b:6.4f}x   "
                  f"{'on faster' if b < a else 'on SLOWER'}")
        off = sorted(r["device"] for r in allrows["off"])
        on = sorted(r["device"] for r in allrows["on"])
        wins = sum(1 for a, b in zip(off, on) if b < a)
        print(f"\n  rank-paired on device: on faster in {wins} of {min(len(off), len(on))}")
        print(f"  off device range {off[0]:.3f}-{off[-1]:.3f}, "
              f"on {on[0]:.3f}-{on[-1]:.3f}"
              f"{'  (DISJOINT)' if on[-1] < off[0] else '  (overlapping)'}")


if __name__ == "__main__":
    main()
