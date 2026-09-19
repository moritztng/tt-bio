#!/usr/bin/env python3
"""Sample AICLK per board while a run is stepping. Labelled by PCI BDF, not by list position."""
import json, subprocess, sys, time, statistics as st
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
gap = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
acc = {}
for _ in range(n):
    out = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"], capture_output=True,
                         text=True).stdout
    try:
        d = json.loads(out[out.index("{"):])
    except ValueError:
        time.sleep(gap); continue
    for dev in d["device_info"]:
        acc.setdefault(dev["board_info"]["bus_id"], []).append(int(dev["telemetry"]["aiclk"]))
    time.sleep(gap)
for bdf, v in sorted(acc.items()):
    print(f"{bdf}  n={len(v)}  median {st.median(v):.0f} MHz  min {min(v)}  max {max(v)}")
