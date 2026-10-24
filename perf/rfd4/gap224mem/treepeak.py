"""Run a command and sample the PSS of its whole process tree every 0.1 s from outside; print the command's
output, then one JSON line: peak tree PSS and the largest single child's PSS / RSS / private memory."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
from procmem import rollup, tree
p = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
peak, child = 0.0, {}
while p.poll() is None:
    rs = {q: rollup(q) for q in tree(p.pid)}
    peak = max(peak, sum(r["Pss"] for r in rs.values() if r))
    for q, r in rs.items():
        if r and q != p.pid and r["Pss"] > child.get("Pss", 0):
            child = r
    time.sleep(0.1)
print(p.stdout.read(), end="")
print(json.dumps({"host_peak_mib": round(peak), "child_max": {k: round(v) for k, v in child.items()}}))
