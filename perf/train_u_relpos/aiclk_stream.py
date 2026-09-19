#!/usr/bin/env python3
"""Stream AICLK per board while the arms step, one line per sample, labelled by PCI BDF.

Streaming rather than summarising at the end: the summarising form loses everything if it is
killed when the arm it was watching finishes, which is exactly what happened to arm A1's sample.
"""
import json, subprocess, sys, time
gap = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
while True:
    out = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"], capture_output=True,
                         text=True).stdout
    try:
        d = json.loads(out[out.index("{"):])
    except ValueError:
        time.sleep(gap); continue
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for dev in d["device_info"]:
        print(f"{ts} {dev['board_info']['bus_id']} {int(dev['telemetry']['aiclk'])}", flush=True)
    time.sleep(gap)
