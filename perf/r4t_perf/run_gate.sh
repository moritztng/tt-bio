#!/bin/bash
# r4t-perf G1: gate_bw lever (TT_BIO_GATE_BW_FUSED) on the PairUpdate crop-256 backward (blockprof), interleaved x3: A off, B on.
cd ~/.coworker/wt/r4t-perf || exit 1
O=~/r4t/perf/gate; mkdir -p $O; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python

free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
for i in $(seq 1 90); do free && break; sleep 20; done
free || { echo "NO WINDOW $(date -u +%FT%TZ)" >> $O/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log
for i in 1 2 3; do for arm in A B; do
  case $arm in A) E=0;; B) E=1;; esac; :
  env TT_BIO_GATE_BW_FUSED=$E PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
    TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 400 $PY perf/r4t_perf/blockprof.py --n 256 --reps 3 --out $O/$arm$i.json > $O/$arm$i.log 2>&1
  echo "RC_$arm$i $? $(date -u +%T) gate=$E load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log
done; done
echo "END $(date -u +%FT%TZ)" >> $O/run.log
