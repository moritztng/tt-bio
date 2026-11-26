#!/bin/bash
# Exposed sync inside the real sharded AdamW step (RS + gather, 16 ranks x 178M) right after run_ex16q.sh, same window.
V=~/scratch/shipflow/venv-c4f99e695d34/bin/python; W=~/.coworker/wt/r4t-scale; O=${1:-~/r4t/scale/ex16q}
until grep -q ALL $O/load 2>/dev/null; do sleep 20; done
cd $W
run() { echo "$1 load $(cut -d" " -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load; shift; "$@"; echo "rc=$?" >> $O/load; }
for rep in 1 2; do
  run opt$rep env PYTHONPATH=$W TT_VISIBLE_DEVICES= OMP_NUM_THREADS=1 timeout 1200 $V perf/r4t_scale/sharded_step_cost.py --world 16 --steps 6 --out $O/opt_r$rep.json > $O/opt_r$rep.log 2>&1
done
run bw2 env PYTHONPATH=$W timeout 600 $V perf/r4t_scale/stream_bw.py --procs 16 --out $O/bw16_2.json > $O/bw2.log 2>&1
echo OPTDONE >> $O/load
