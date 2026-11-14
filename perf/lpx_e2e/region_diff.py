"""Side-by-side region timers (self seconds) of census reps from bench.jsonl files.

    python3 perf/lpx_e2e/region_diff.py A.jsonl:arm:seed B.jsonl:arm:seed [...]
"""
import json
import sys


def census(spec):
    path, arm, seed = spec.rsplit(":", 2)
    for line in open(path):
        d = json.loads(line)
        if d.get("census") and d["arm"] == arm and d["seed"] == int(seed):
            return d["fold_s"], {r["path"]: r for r in d["census"]}
    raise SystemExit(f"no census rep for {spec}")


specs = sys.argv[1:]
runs = [census(s) for s in specs]
paths = sorted(set().union(*(r for _, r in runs)))
names = [s.rsplit(":", 2)[1] + ":" + s.rsplit(":", 2)[2] for s in specs]
print(f"{'region (self s)':58s}" + "".join(f"{n:>16s}" for n in names) + f"{'d(last-first)':>14s}")
for p in paths:
    v = [r.get(p, {}).get("self_s", 0.0) for _, r in runs]
    if max(v) < 0.5:
        continue
    print(f"{p:58s}" + "".join(f"{x:16.1f}" for x in v) + f"{v[-1] - v[0]:+14.1f}")
print(f"{'fold_s':58s}" + "".join(f"{f:16.1f}" for f, _ in runs) + f"{runs[-1][0] - runs[0][0]:+14.1f}")
