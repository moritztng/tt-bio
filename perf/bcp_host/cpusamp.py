#!/usr/bin/env python3
"""Sample a process CPU (all threads, user+sys) every 0.1 s until it exits: cpusamp.py PID OUT.json"""
import json, os, sys, time
pid, out, tck, rows = int(sys.argv[1]), sys.argv[2], os.sysconf("SC_CLK_TCK"), []
while os.path.exists(f"/proc/{pid}"):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        rows.append((time.time(), (int(f[11]) + int(f[12])) / tck, int(f[17])))
    except (OSError, IndexError):
        break
    time.sleep(0.1)
json.dump(rows, open(out, "w"))
