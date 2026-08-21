#!/bin/sh
# The four owed qb1 batch-1 walls, in one go, once the box is quiet. Runs the cheapest rung
# FIRST and stops if its A/A floor is bad: the 128 aa trunk is dispatch-bound, so it is the
# most co-tenancy-sensitive rung on the ladder, not the least (pass 9 measured 1.629x under a
# 28-core co-tenant where the 768 aa trunk held flat to 0.008 %).
cd /home/ttuser/.coworker/wt/rf3-perf-p9 || exit 1
end=$(( $(date +%s) + ${WAIT_S:-2400} ))
while [ "$(date +%s)" -lt "$end" ]; do
  l=$(cut -d' ' -f1 /proc/loadavg); li=${l%%.*}
  [ "$li" -lt 6 ] && break
  echo "waiting for a quiet box, load=$l $(date -u +%H:%M:%S)"; sleep 30
done
echo "starting at load $(cut -d' ' -f1-3 /proc/loadavg)"
for aa in 128 256 512 768; do
  P9_DEV=${P9_DEV:-1} sh perf/rf3/p9_qb1.sh "$aa" p6_l8,p6_l8_aa || exit 1
  cp -f /home/ttuser/rf3_perf_work/p9_qb1_b1_$aa.json perf/rf3/results/p9_qb1_b1_$aa.json
  /home/ttuser/tt-bio-dev/env/bin/python3 - "$aa" <<'PY'
import json, sys
aa = sys.argv[1]
d = json.load(open(f"perf/rf3/results/p9_qb1_b1_{aa}.json"))
a, b = d["arms"]["p6_l8"], d["arms"]["p6_l8_aa"]
floor = b["median_warm"]["infer_s"] / a["median_warm"]["infer_s"]
print(f"CELL {aa} aa  {a['median_warm']['infer_s']:.3f} s  "
      f"{a['ratio_vs_target']:.4f}x of {d['tt_target_device_s']}  A/A floor {floor:.4f}")
sys.exit(0 if floor < 1.02 else 3)
PY
  rc=$?
  [ "$rc" -eq 0 ] || { echo "A/A FLOOR BAD at $aa aa, stopping: the box is not measurable"; exit "$rc"; }
done
echo "ALL FOUR RUNGS DONE"
