#!/usr/bin/env python3
"""What is outstanding across BOTH hosts, which is the only count that matters.

Run from the orchestrating side (laptop/pc); it sshes to both hosts.

Each host's progress.jsonl is local, and the hosts fold disjoint slices -- but during the early
chaotic phase qb1 took over some of qb2's slices and failed on them, so qb1's "never ok" list is full
of pairs qb2 has since completed. Those are not gaps: the merge brings qb2's copy across. Only a pair
that no host has ok is a real hole in the slab.

Run on the laptop/pc side with both hosts' records collected.
"""
import json
import subprocess
import sys

HOSTS = ("tt-quietbox", "tt-quietbox2")
ok, attempted, slices = {}, {}, None

for h in HOSTS:
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", f"ttuser@{h}",
                        "python3 -c \"import json;"
                        "rs=[json.loads(l) for l in open('/home/ttuser/abag_xm/tier_a/progress.jsonl')"
                        " if l.strip()];"
                        "print(json.dumps([[r['target'],r['model'],r.get('status')] for r in rs]))\""],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        sys.exit(f"{h}: {r.stderr[-200:]}")
    rows = json.loads(r.stdout)
    ok[h] = {(t, m) for t, m, s in rows if s == "ok"}
    attempted[h] = {(t, m) for t, m, _ in rows}

r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "ttuser@tt-quietbox",
                    "cat /home/ttuser/.coworker/wt/abag-xm-crossmodel-ranking-dataset-p4/"
                    "docs/implementation-parity-data/abag-xm-tier-a-slices.json"],
                   capture_output=True, text=True, timeout=120)
d = json.loads(r.stdout)
slices = d["slices_8"]
owner = {}
for si, targets in slices.items():
    host = "tt-quietbox2" if int(si) < 4 else "tt-quietbox"
    for t in targets:
        owner[t] = host

GENS = ("protenix-v2", "opendde-abag", "boltz2")
all_pairs = {(t, g) for t in owner for g in GENS}
union_ok = ok["tt-quietbox"] | ok["tt-quietbox2"]

print(f"slab: {len(all_pairs)} pairs ({len(owner)} targets x {len(GENS)} generators)")
print(f"union ok: {len(union_ok)}   outstanding: {len(all_pairs - union_ok)}")
for h in HOSTS:
    print(f"  {h}: {len(ok[h])} ok, {len(attempted[h] - ok[h])} attempted-but-never-ok")

# The interesting part: how many of each host's never-ok pairs are already ok on the other host?
for h in HOSTS:
    other = [x for x in HOSTS if x != h][0]
    never = attempted[h] - ok[h]
    covered = {p for p in never if p in ok[other]}
    print(f"  {h}: {len(covered)} of its {len(never)} never-ok pairs are already ok on {other} "
          f"(the merge covers these)")

real = sorted(all_pairs - union_ok)
attempted_and_failing = [p for p in real if p in attempted[HOSTS[0]] | attempted[HOSTS[1]]]
print(f"\nreal outstanding pairs: {len(real)}")
print(f"  of which ATTEMPTED and still failing: {len(attempted_and_failing)} -> {attempted_and_failing}")
print(f"  of which never attempted (just not reached yet): "
      f"{len(real) - len(attempted_and_failing)}")
mis = [p for p in real if owner.get(p[0]) and p in attempted[owner[p[0]]] and p not in ok[owner[p[0]]]]
if mis:
    print(f"\npairs whose OWNING host attempted and failed them (need a real retry): {mis}")
