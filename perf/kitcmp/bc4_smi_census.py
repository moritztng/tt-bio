"""Device-busy census of a B200 BindCraft 2 run from the 1 Hz nvidia-smi log, no profiler needed.

    python bc4_smi_census.py SMI_CSV RUNS_JSONL [RUNS_JSONL ...]

For each warm trajectory's design window (ending validate_s before the record's t) it averages the samples of `nvidia-smi
--query-gpu=...,utilization.gpu,utilization.memory,...` (box_bc_arms.sh's column order). utilization.gpu is the
fraction of each sample period in which at least one kernel ran, so 1 - util is time the GPU sat idle waiting on
the host; utilization.memory is the fraction of time the HBM controller was busy, an upper bound on the achieved
fraction of the bandwidth roof, not the fraction itself.
"""
import json, statistics, sys
from datetime import datetime, timezone

smi = []
for line in open(sys.argv[1]):
    f = [x.strip() for x in line.split(",")]
    try:
        t = datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        smi.append((t, float(f[1].split()[0]), float(f[3].split()[0]), float(f[6].split()[0]), float(f[7].split()[0])))
    except (ValueError, IndexError):
        continue
rows = []
for p in sys.argv[2:]:
    for line in open(p):
        r = json.loads(line)
        if r.get("ev") != "rep" or r.get("kind") != "warm" or not r.get("iterations"):
            continue
        end = r["t"] - (r.get("validate_s") or 0)  # a record is written after its validation, if it had one
        w = [s for s in smi if end - r["design_s"] <= s[0] <= end]
        if not w:
            continue
        rows.append(dict(run=p, traj=r["trajectory"], s_per_iter=round(r["design_s"] / r["iterations"], 4),
                         terminated=r.get("terminated"), n=len(w),
                         gpu_busy=round(statistics.mean(s[3] for s in w) / 100, 3),
                         hbm_busy=round(statistics.mean(s[4] for s in w) / 100, 3),
                         power_w=round(statistics.mean(s[2] for s in w)), sm=statistics.median(s[1] for s in w)))
for r in rows:
    print(json.dumps(r))
if rows:
    print(json.dumps(dict(summary=True, n=len(rows), gpu_busy=round(statistics.median(r["gpu_busy"] for r in rows), 3),
                          hbm_busy=round(statistics.median(r["hbm_busy"] for r in rows), 3),
                          power_w=statistics.median(r["power_w"] for r in rows))))
