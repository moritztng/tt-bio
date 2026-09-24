"""Longest gap between lease renewals per job, from hb_sample.py's renewals-<host>.jsonl.

Each distinct lease_until is one renewal (lease or heartbeat), so the gaps between
consecutive values are the silences the lease had to cover. Run in this directory.
"""
import collections, glob, json

rows, worst = [], (0.0, None)
for path in sorted(glob.glob("renewals-*.jsonl")):
    by = collections.defaultdict(list)
    for line in open(path):
        x = json.loads(line)
        by[(x["name"], x["worker"])].append(x["lease_until"])
    for (name, worker), v in sorted(by.items()):
        gaps = [b - a for a, b in zip(v, v[1:])]
        g = max(gaps)
        med = sorted(gaps)[len(gaps) // 2]
        rows.append({"job": name, "worker": worker, "renewals": len(v), "held_s": round(v[-1] - v[0], 1),
                     "median_gap_s": round(med, 2), "max_gap_s": round(g, 2)})
        worst = max(worst, (g, worker))
for r in rows:
    print(r)
print(json.dumps({"jobs": len(rows), "renewals": sum(r["renewals"] for r in rows),
                  "longest_held_s": max(r["held_s"] for r in rows),
                  "max_gap_s": round(worst[0], 2), "on": worst[1]}))
