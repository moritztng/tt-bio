#!/bin/bash
# 16-rank sharded AdamW step A/B (tip 785e952d6 vs exdev), after run_ex16.sh finishes.
V=~/scratch/shipflow/venv-c4f99e695d34/bin/python; O=~/r4t/scale/opt16; NEW=~/r4t/scale/exdev; OLD=~/.coworker/wt/r4t-scale
mkdir -p $O; cd $NEW
until grep -q ^ALL ~/r4t/scale/ex16/load 2>/dev/null; do sleep 10; done
run() { echo "$1 load $(cut -d' ' -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load; shift; "$@"; echo "rc=$?" >> $O/load; }
for rep in 1 2; do
  run old$rep env PYTHONPATH=$OLD TT_VISIBLE_DEVICES= timeout 1200 $V perf/r4t_scale/sharded_step_cost.py --world 16 --steps 5 --out $O/old_r$rep.json > $O/old_r$rep.log 2>&1
  run new$rep env PYTHONPATH=$NEW TT_VISIBLE_DEVICES= timeout 1200 $V perf/r4t_scale/sharded_step_cost.py --world 16 --steps 5 --out $O/new_r$rep.json > $O/new_r$rep.log 2>&1
done
run newbroken env PYTHONPATH=$NEW TT_VISIBLE_DEVICES= timeout 1200 $V perf/r4t_scale/sharded_step_cost.py --world 16 --steps 2 --broken --out $O/new_broken.json > $O/new_broken.log 2>&1
echo ALL >> $O/load
