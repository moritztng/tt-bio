#!/bin/bash
# D47 grade of the blocked reduce_scatter: wait (<= 20 h) for qb2 1- and 5-min load < 3, then 16-rank copy bandwidth,
# the real sharded step at this tree x3 (RS split into wait and add), one at ce6661c8e (pre-blocking) and the broken arm.
V=~/scratch/shipflow/venv-c4f99e695d34/bin/python; W=$(cd $(dirname $(readlink -f $0))/../.. && pwd)
O=${1:-~/r4t/scale/rsblk_quiet}; OLD=${2:-~/r4t/scale/rsblk/old}; deadline=$(( $(date +%s) + 20*3600 ))
until awk "{exit !(\$1<3 && \$2<3)}" /proc/loadavg; do [ $(date +%s) -gt $deadline ] && { mkdir -p $O; echo "never quiet" > $O/never; exit 3; }; sleep 60; done
mkdir -p $O; cd $W; echo "sha $(git rev-parse --short HEAD)" > $O/load
run() { echo "$1 load $(cut -d" " -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load; shift; "$@"; echo "rc=$?" >> $O/load; }
leg() { run $1 env PYTHONPATH=$2 TT_VISIBLE_DEVICES= OMP_NUM_THREADS=1 timeout 1200 $V $2/perf/r4t_scale/sharded_step_cost.py --world 16 $3 --out $O/$1.json > $O/$1.log 2>&1; }
run bw0 env PYTHONPATH=$W timeout 600 $V perf/r4t_scale/stream_bw.py --procs 16 --out $O/bw0.json > $O/bw0.log 2>&1
leg new1 $W "--steps 6"; leg old1 $OLD "--steps 6"; leg new2 $W "--steps 6"; leg new3 $W "--steps 6"
leg broken $W "--steps 3 --broken"
run bw1 env PYTHONPATH=$W timeout 600 $V perf/r4t_scale/stream_bw.py --procs 16 --out $O/bw1.json > $O/bw1.log 2>&1
echo DONE >> $O/load
