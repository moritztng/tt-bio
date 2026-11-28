#!/bin/bash
# 16-rank exchange A/B on qb2 host: tip (785e952d6) vs persistent-slot build, + copy bandwidth.
V=~/scratch/shipflow/venv-c4f99e695d34/bin/python; O=~/r4t/scale/ex16; NEW=~/r4t/scale/exdev; OLD=~/.coworker/wt/r4t-scale
mkdir -p $O; cd $NEW
run() { echo "$1 load $(cut -d' ' -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load; shift; "$@"; echo "rc=$?" >> $O/load; }
run bw0 env PYTHONPATH=$NEW timeout 600 $V perf/r4t_scale/stream_bw.py --procs 16 --out $O/bw16_0.json > $O/bw0.log 2>&1
for rep in 1 2; do
  run old$rep env PYTHONPATH=$OLD timeout 900 $V $NEW/perf/r4t_scale/exchange_cost.py --world 16 --steps 6 --out $O/old_r$rep.json > $O/old_r$rep.log 2>&1
  run zc$rep env PYTHONPATH=$NEW timeout 900 $V perf/r4t_scale/exchange_cost.py --world 16 --steps 6 --zero-copy --out $O/zc_r$rep.json > $O/zc_r$rep.log 2>&1
done
run zcbroken env PYTHONPATH=$NEW timeout 900 $V perf/r4t_scale/exchange_cost.py --world 16 --steps 3 --zero-copy --broken --out $O/zc_broken.json > $O/zc_broken.log 2>&1
run bw1 env PYTHONPATH=$NEW timeout 600 $V perf/r4t_scale/stream_bw.py --procs 16 --out $O/bw16_1.json > $O/bw1.log 2>&1
echo ALL >> $O/load
