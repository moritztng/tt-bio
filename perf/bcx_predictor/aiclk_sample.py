"""Sample the card's own AICLK sysfs node into a jsonl for the length of a trajectory.

A trajectory outlives the process that measures it, so the in-process `stack.Clock` cannot
carry it: every timing this row publishes has to name the clock it was measured at, and on
a box that reads CLAMPED-UNDER-LOAD that is the difference between a number and an
artifact. Reads `tt_aiclk` directly rather than parsing tt-smi, whose AICLK column is a
right-aligned string.
"""
import json, os, sys, time

sys.path.insert(0, "/home/ttuser/.coworker/wt/bcx-predictor/perf/bcx_stack")
from stack import sysfs_node

out_path, dt = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
node, pci = sysfs_node()
path = f"{node}/tt_aiclk"
with open(out_path, "a", buffering=1) as fh:
    fh.write(json.dumps({"pci": pci, "node": path, "dt": dt, "pid": os.getpid()}) + "\n")
    while True:
        try:
            fh.write(json.dumps({"t": time.time(),
                                 "aiclk": int(open(path).read().split()[0]),
                                 "load1": os.getloadavg()[0]}) + "\n")
        except Exception as exc:
            fh.write(json.dumps({"t": time.time(), "error": str(exc)}) + "\n")
        time.sleep(dt)
