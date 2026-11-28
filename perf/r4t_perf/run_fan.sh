#!/bin/bash
# r4t-perf F1: fan-in cast levers on the PairUpdate crop-256 backward (blockprof), interleaved x3:
# A = default, B = TT_BIO_WIDEN_ADD=1, C = B + TT_BIO_FANIN_CAST_FUSED=1. Waits for l24b (pid $1) and a free card 2.
cd ~/.coworker/wt/r4t-perf || exit 1
O=~/r4t/perf/fan; mkdir -p $O; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
while kill -0 ${1:?l24b pid} 2>/dev/null; do sleep 20; done
free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
for i in $(seq 1 90); do free && break; sleep 20; done
free || { echo "NO WINDOW $(date -u +%FT%TZ)" >> $O/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log
for i in 1 2 3; do for arm in A B C; do
  case $arm in A) E="0 0";; B) E="1 0";; C) E="1 1";; esac; set -- $E
  env TT_BIO_WIDEN_ADD=$1 TT_BIO_FANIN_CAST_FUSED=$2 PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
    TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 400 $PY perf/r4t_perf/blockprof.py --n 256 --reps 3 --out $O/$arm$i.json > $O/$arm$i.log 2>&1
  echo "RC_$arm$i $? $(date -u +%T) widen=$1 fused=$2 load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log
done; done
echo "END $(date -u +%FT%TZ)" >> $O/run.log
