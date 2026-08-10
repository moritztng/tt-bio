#!/bin/bash
# The 117 aa half of the sweep, both models, both arms. 298 aa is covered by paired.sh, which
# alternates the arms round by round; at 117 aa the fold is short enough that a straight
# interleaved pair is good enough. Resumable: fold_ab.py skips an existing JSON.
set -u
cd /home/ttuser/.coworker/wt/perfwar-w6-c2fix-land || exit 1
export PYTHONPATH="$PWD"
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:perfwar-w6-c2fix-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
for MODEL in protenix-v2 opendde; do
  for ARM in BASE C2FIX; do
    if [ -s "perf/w6_c2fix/out/${ARM}_${MODEL}_117.json" ]; then echo "SKIP $ARM $MODEL 117"; continue; fi
    $PY perf/w6_c2fix/arm.py --arm "$ARM" >/dev/null || exit 1
    $PY perf/w6_c2fix/fold_ab.py --arm "$ARM" --model "$MODEL" --size 117 --repeat 3 >/dev/null || exit 1
    echo "117 $MODEL $ARM done $(date -u +%H:%M:%S)"
  done
done
echo "SMALL DONE $(date -u +%H:%M:%S)"
