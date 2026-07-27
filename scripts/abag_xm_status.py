"""Campaign status with provenance tagging.

A record is CURRENT only if mps == 3; anything with mps == 5 was written by the aborted
2026-07-27 launch and is retried automatically, since done_pairs() skips only status=ok.
Reading a stale record as a live failure is the false alarm that turn 27 defused.
"""
import collections
import json
import pathlib
import statistics
import sys

P = pathlib.Path("/home/ttuser/abag_xm/tier_a/progress.jsonl")
cur, stale, walls = [], 0, collections.defaultdict(list)
counts = collections.Counter()
if P.exists():
    for line in open(P):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("mps") != 3:
            stale += 1
            continue
        cur.append(d)
        counts[(d.get("model"), d.get("status"))] += 1
        if d.get("status") == "ok":
            walls[d["model"]].append(d.get("wall_s") or 0)

print("stale (mps!=3, auto-retried): %d   current: %d" % (stale, len(cur)))
for k, v in sorted(counts.items()):
    print("   %-28s %d" % (str(k), v))
for m, w in sorted(walls.items()):
    print("   %-14s ok=%2d  wall_s median %.0f  min %.0f  max %.0f"
          % (m, len(w), statistics.median(w), min(w), max(w)))
bad = [(d["target"], d["model"], d.get("status"), d.get("n_cifs"))
       for d in cur if d.get("status") != "ok"]
print("   CURRENT non-ok:", bad or "NONE")
odd = [(d["target"], d["model"], d.get("n_cifs"), d.get("n_paes"))
       for d in cur if d.get("status") == "ok" and (d.get("n_cifs") != 50 or d.get("n_paes") != 50)]
print("   ok but wrong artifact count:", odd or "NONE")
ht = {d.get("host_threads") for d in cur}
print("   host_threads on current records:", ht, "(null would mean uncapped)")
