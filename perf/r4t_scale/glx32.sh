#!/bin/bash
# r4t-scale: host-only DP sync + sharded AdamW step on one Galaxy host (anx-glx, 64 cores), no chips opened.
# Widths 16 and 32 (the Galaxy's rank count), plus a broken arm; copy bandwidth before/after. Stops on ~/fleet/HANDBACK.
W=~/fleet/r4t/r4t-scale; O=~/fleet/r4t/out; V=~/fleet/env/bin/python; mkdir -p $O; cd $W
run() { [ -e ~/fleet/HANDBACK ] && { echo HANDBACK >> $O/load; exit 4; }
  echo "$1 load $(cut -d' ' -f1-3 /proc/loadavg) $(date -u +%T)" >> $O/load; shift; "$@"; echo "rc=$?" >> $O/load; }
E="env PYTHONPATH=$W TT_VISIBLE_DEVICES= OMP_NUM_THREADS=1 timeout 1800 $V"
echo "sha 262d123e3 host $(hostname) nproc $(nproc)" > $O/load
run bw32a $E perf/r4t_scale/stream_bw.py --procs 32 --out $O/bw32a.json > $O/bw32a.log 2>&1
for w in 16 32; do
  run ex$w $E perf/r4t_scale/exchange_cost.py --world $w --steps 6 --zero-copy --out $O/ex$w.json > $O/ex$w.log 2>&1
  run st${w}a $E perf/r4t_scale/sharded_step_cost.py --world $w --steps 6 --out $O/st${w}a.json > $O/st${w}a.log 2>&1
  run st${w}b $E perf/r4t_scale/sharded_step_cost.py --world $w --steps 6 --out $O/st${w}b.json > $O/st${w}b.log 2>&1
done
run broken32 $E perf/r4t_scale/sharded_step_cost.py --world 32 --steps 3 --broken --out $O/broken32.json > $O/broken32.log 2>&1
run exbroken32 $E perf/r4t_scale/exchange_cost.py --world 32 --steps 3 --zero-copy --broken --out $O/exbroken32.json > $O/exbroken32.log 2>&1
run bw32b $E perf/r4t_scale/stream_bw.py --procs 32 --out $O/bw32b.json > $O/bw32b.log 2>&1
echo DONE >> $O/load
