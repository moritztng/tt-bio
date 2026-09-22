#!/bin/bash
# The loop gain the growth law is read at is not free: how well a shape bar discriminates
# depends on how much of the drive is the closed-loop feedback. This ladder measures it.
cd /home/ttuser/.coworker/wt/of3t-traj20 || exit 1
P=/home/ttuser/ptxft-venv/bin/python3
for r in 0.005 0.05 0.2 0.5; do
  for a in scaled stale miswire; do
    $P perf/of3t_traj20/traj20.py --arm "$a" --limit 40 --rho "$r" \
       --out "/home/ttuser/of3t_traj20/gl_${a}_${r}.json" >/dev/null 2>&1 \
       || echo "FAILED $a $r"
  done
done
$P perf/of3t_traj20/summarise.py /home/ttuser/of3t_traj20/gl_*.json
