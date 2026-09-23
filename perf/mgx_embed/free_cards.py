"""Print the whglx chips whose lease is released or whose holder pid is dead, never 1 or 24-27."""
import json, os, glob
BLOCKED = {"1", "24", "25", "26", "27"}   # live app.japanfold.com and a co-tenant
for p in sorted(glob.glob("/home/agent/leases/j10glx02-card*.json")):
    m = json.load(open(p))
    if m.get("card") in BLOCKED:
        continue
    try:
        os.kill(int(m["pid"]), 0); alive = True
    except (OSError, TypeError, ValueError, KeyError):
        alive = False
    if m.get("released") or not alive:
        print(m.get("card"), m.get("holder"), "released" if m.get("released") else "dead-pid")
