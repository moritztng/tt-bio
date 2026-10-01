#!/usr/bin/env python3
"""Cores busy (process CPU-s per wall-s) in each host gap and seam: cpugaps.py out/<arm> out/<arm>_cpu.json"""
import json, sys
d = json.load(open(sys.argv[1] + "/round_events.json")); ev = d["events"]; cpu = json.load(open(sys.argv[2]))
def cores(t0, t1):
    a = [r for r in cpu if t0 <= r[0] <= t1]
    return (round((a[-1][1] - a[0][1]) / (a[-1][0] - a[0][0]), 1), max(r[2] for r in a)) if len(a) > 1 else (None, None)
b = sorted(e["t0"] for e in ev if e["kind"] == "round_start") + [e["t0"] for e in ev if e["kind"] == "round_stop"][:1]
for i in range(2, len(b) - 1):
    t0, t1 = b[i], b[i + 1]
    dev = sorted((e for e in ev if e["kind"] == "device" and t0 <= e["t0"] and e["t1"] <= t1), key=lambda e: e["t0"])
    clk = sorted(x[1] for x in d["aiclk"] if t0 <= x[0] <= t1)
    print(f"round {i + 1}  wall {t1 - t0:.2f} s  AICLK med {clk[len(clk) // 2]} min {clk[0]}")
    prev = ("start", t0)
    for e in dev + [{"module": "round", "phase": "end", "t0": t1, "t1": t1, "dt": 0}]:
        n = e["module"] + "/" + e["phase"]
        print(f"  host  {e['t0'] - prev[1]:5.2f} s  cores {cores(prev[1], e['t0'])[0]}   -> {n}")
        if e["dt"]:
            print(f"  seam  {e['dt']:5.2f} s  cores {cores(e['t0'], e['t1'])[0]}   {n}")
        prev = (n, e["t1"])
    print("  threads", cores(t0, t1)[1])
