#!/bin/bash
V=~/scratch/shipflow/venv-c4f99e695d34/bin/python; O=~/r4t/scale/ab2; NEW=~/.coworker/wt/r4t-scale; OLD=~/r4t/scale/ab_old
for rep in 1 2; do for w in 4 8; do for arm in old new; do
  R=$OLD; [ $arm = new ] && R=$NEW
  echo "$rep $w $arm load $(cut -d" " -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load
  (cd $R && PYTHONPATH=$R timeout 600 $V $NEW/perf/r4t_scale/exchange_cost.py --world $w --steps 6 --out $O/${arm}_x${w}_r$rep.json > $O/${arm}_x${w}_r$rep.log 2>&1); echo "$rep $w $arm rc=$?" >> $O/done
done; done; done; echo ALL >> $O/done
