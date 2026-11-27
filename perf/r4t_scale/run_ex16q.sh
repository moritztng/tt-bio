#!/bin/bash
# D47 bar point: 16-rank exchange at the tip (--zero-copy = persistent slots), STREAM-style copy bandwidth before and after
# in the same window, broken arm last. Load logged per leg; the bar needs a quiet host, so the load column decides validity.
V=~/scratch/shipflow/venv-c4f99e695d34/bin/python; W=~/.coworker/wt/r4t-scale; O=${1:-~/r4t/scale/ex16q}
mkdir -p $O; cd $W; echo "sha $(git rev-parse --short HEAD)" > $O/load
run() { echo "$1 load $(cut -d" " -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load; shift; "$@"; echo "rc=$?" >> $O/load; }
run bw0 env PYTHONPATH=$W timeout 600 $V perf/r4t_scale/stream_bw.py --procs 16 --out $O/bw16_0.json > $O/bw0.log 2>&1
for rep in 1 2; do
  run zc$rep env PYTHONPATH=$W timeout 900 $V perf/r4t_scale/exchange_cost.py --world 16 --steps 6 --zero-copy --out $O/zc_r$rep.json > $O/zc_r$rep.log 2>&1
done
run zcbroken env PYTHONPATH=$W timeout 900 $V perf/r4t_scale/exchange_cost.py --world 16 --steps 3 --zero-copy --broken --out $O/zc_broken.json > $O/zc_broken.log 2>&1
run bw1 env PYTHONPATH=$W timeout 600 $V perf/r4t_scale/stream_bw.py --procs 16 --out $O/bw16_1.json > $O/bw1.log 2>&1
echo ALL >> $O/load
