"""Which programs ran in one fold of a bench run, from the tt-metal inspector logs.

usage: inspwin.py RUN_DIR INPUT SEED > out.tsv
Prints, per program enqueued during that fold: program id, enqueue count, whether the program was created inside the
fold (miss) or before it (cache hit), and its kernels as source@binary-hash.
"""
import json, re, sys
from collections import Counter, defaultdict

run, inp, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
ins = f"{run}/insp/generated/inspector"
for line in open(f"{run}/b/bench.jsonl"):
    d = json.loads(line)
    if d.get("input") == inp and d.get("seed") == seed:
        t1 = d["t_unix"] * 1e9
        t0 = t1 - d["wall_s"] * 1e9
        break
else:
    sys.exit("fold not found")

kern = defaultdict(set)
cur = {}
for line in open(f"{ins}/kernels.yaml"):
    m = re.match(r"\s+(\w+): (.*)", line)
    if line.startswith("- kernel:"):
        cur = {}
    elif m:
        cur[m[1]] = m[2]
        if m[1] == "program_id":
            src = cur.get("source", "?").split("/")[-1]
            h = cur.get("path", "").rstrip("/").split("/")[-1]
            kern[int(m[2])].add(f"{src}@{h}")

created = {}
for line in open(f"{ins}/programs_log.yaml"):
    if line.startswith("- "):
        ev = line[2:].strip().rstrip(":")
        pid = None
    elif ev == "program_created":
        m = re.match(r"\s+(id|timestamp_ns): (\d+)", line)
        if m and m[1] == "id":
            pid = int(m[2])
        elif m and pid is not None:
            created.setdefault(pid, int(m[2]))

wl_prog, runs = {}, Counter()
ev = wid = None
for line in open(f"{ins}/mesh_workloads_log.yaml"):
    if line[0] == "-":
        ev = line[2:].strip().rstrip(":")
        wid = None
        continue
    s = line.strip()
    if s.startswith("mesh_workload_id:"):
        wid = int(s.split()[1])
    elif ev == "mesh_workload_add_program" and s.startswith("program_id:"):
        wl_prog[wid] = int(s.split()[1])
    elif ev == "mesh_workload_set_program_binary_status" and s.startswith("timestamp_ns:"):
        if t0 <= int(s.split()[1]) <= t1:
            runs[wid] += 1

for wid, n in sorted(runs.items()):
    p = wl_prog.get(wid, -1)
    c = created.get(p, 0)
    print(p, n, "miss" if c >= t0 else "hit", " ".join(sorted(kern[p])), sep="\t")
