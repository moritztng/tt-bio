#!/bin/bash
# 16-rank sharded AdamW step A/B: 34d2d5ee1 (every rank rounds the whole model twice) vs 22c8f8e95 (owner rounds, flags gathered).
V=~/scratch/shipflow/venv-c4f99e695d34/bin/python; O=~/r4t/scale/opt16b; NEW=~/.coworker/wt/r4t-scale; OLD=$O/old
cd $NEW
run() { echo "$1 load $(cut -d" " -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load; shift; "$@"; echo "rc=$?" >> $O/load; }
for rep in 1 2; do
  for arm in old new; do
    T=$NEW; [ $arm = old ] && T=$OLD
    run $arm$rep env PYTHONPATH=$T TT_VISIBLE_DEVICES= OMP_NUM_THREADS=1 timeout 1200 $V $T/perf/r4t_scale/sharded_step_cost.py --world 16 --steps 5 --out $O/${arm}_r$rep.json > $O/${arm}_r$rep.log 2>&1
  done
done
run newbroken env PYTHONPATH=$NEW TT_VISIBLE_DEVICES= OMP_NUM_THREADS=1 timeout 1200 $V perf/r4t_scale/sharded_step_cost.py --world 16 --steps 2 --broken --out $O/new_broken.json > $O/new_broken.log 2>&1
echo ALL >> $O/load
