"""Run one command with AICLK and load1 sampled during it; print one JSON line.

    python perf/rfd4/small/clocked.py <log> <cmd...>
"""
import json
import os
import subprocess
import sys
import time

from clocksample import during


def main():
    log, cmd = sys.argv[1], sys.argv[2:]
    load0 = os.getloadavg()[0]
    t0 = time.time()
    with during() as clk, open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    s = clk.summary().get(0)
    print(json.dumps({"rc": rc, "wall_s": round(time.time() - t0, 2), "aiclk": s,
                      "aiclk_mhz": s and s["median"], "clock": clk.line(),
                      "load1": [round(load0, 2), round(os.getloadavg()[0], 2)]}))


if __name__ == "__main__":
    main()
