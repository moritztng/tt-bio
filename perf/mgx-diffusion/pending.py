"""Which points of each plan are still owed on the current engine tree.

    python perf/mgx-diffusion/pending.py <plan> ...

Same resume rule as ladder.py: a point counts once a warm, probe, error or refusal cell exists
for it on this engine tree and host.
"""
import json
import socket
import sys

from ladder import RUNS, git, parse, point_id

engine, host = git("rev-parse", "HEAD:tt_bio"), socket.gethostname()
seen = {point_id(c) for c in map(json.loads, RUNS.read_text().splitlines())
        if (c["engine"], c["host"]) == (engine, host)
        and (not c.get("cold") or c.get("probe") or c.get("error") or c.get("refused"))}
for path in sys.argv[1:]:
    plan = [p for p in map(parse, open(path).read().splitlines()) if p]
    owed = [p for p in plan if point_id(p) not in seen]
    print(f"{path}: {len(owed)} of {len(plan)} owed")
    for p in owed:
        print("   ", " ".join([p["model"], str(p["tokens"]), str(p["samples"])]
                             + [f"{k}={v}" for k, v in p.items() if k not in ("model", "tokens", "samples")]))
