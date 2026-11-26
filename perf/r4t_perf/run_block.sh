#!/bin/bash
# r4t-perf: inside our qb2 chip-2 booking (owner fleet-r4t-perf), one PairUpdate block attribution (blockprof.py),
# then hand the window back at once so r4t-step's trunk arms (queued behind us on chip 2) start.
# Never opens the card outside the booking: a wait that times out exits.
cd ~/.coworker/wt/r4t-perf || exit 1
OUT=~/r4t/perf/block; mkdir -p $OUT
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
ok=
for i in $(seq 1 240); do
  who=$(python3 ~/spd/feed/book.py active 2 2>/dev/null)
  if [ "$who" = "fleet-r4t-perf" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; then ok=1; break; fi
  echo "$(date -u +%FT%TZ) waiting: active=$who" >> $OUT/wait.log; sleep 30
done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" > $OUT/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 900 \
  $PY perf/r4t_perf/blockprof.py --n 256 --out $OUT/block256.json > $OUT/block.log 2>&1
echo "RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
python3 ~/spd/feed/book.py rm fleet-r4t-perf
echo "END released $(date -u +%FT%TZ)" >> $OUT/run.log
