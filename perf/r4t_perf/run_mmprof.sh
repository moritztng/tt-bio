#!/bin/bash
# r4t-perf: L2-on PairUpdate crop-256 profile with per-call-site matmul detail. Waits for run_b1h.sh's device arms
# to finish (never shares chip 2), then waits for the card to be free inside our booking.
cd ~/.coworker/wt/r4t-perf || exit 1
O=~/r4t/perf/mm; mkdir -p $O; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
while pgrep -f run_b1h.sh > /dev/null && ! grep -q "it4 RC\|NO WINDOW" ~/r4t/perf/b1h/run.log; do sleep 20; done
free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
for i in $(seq 1 90); do free && break; sleep 20; done
free || { echo "NO WINDOW $(date -u +%FT%TZ)" > $O/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $O/run.log
env TT_BIO_LNBW_WEIGHTS=1 PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 400 \
  $PY perf/r4t_perf/blockprof.py --n 256 --reps 3 --out $O/prof_on.json > $O/prof_on.log 2>&1
echo "prof RC $? $(date -u +%FT%TZ)" >> $O/run.log
