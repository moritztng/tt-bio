#!/usr/bin/env python3
"""1 Hz AICLK log for pc card 0, so a round window can be given its clock samples and the
count under 1200 -- which `per_round`'s min/median/max per round does not carry."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "bcx_stack"))
from stack import sysfs_node                                            # noqa: E402

node, pci = sysfs_node()
path = f"{node}/tt_aiclk"
out = open(sys.argv[1], "a", buffering=1)
out.write(f"# {path} pci={pci} start={time.time()}\n")
while True:
    try:
        out.write(f"{time.time():.3f}\t{int(open(path).read().split()[0])}\t"
                  f"{os.getloadavg()[0]:.2f}\n")
    except Exception:
        pass
    time.sleep(1.0)
