#!/bin/bash
# r4t-perf: B1' (D31) grade of device Muon NS. Device arms gram + gram --broken inside our chip-2 booking (per-matrix
# errors), card released, then the CPU grader (b1prime.py). Never opens the card while anyone else has it open.
cd ~/.coworker/wt/r4t-perf || exit 1
OUT=~/r4t/perf/b1p; mkdir -p $OUT
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
SH=perf/r4t_perf/rfd4_denoiser_2d_shapes.txt
free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
ok=; for i in $(seq 1 120); do free && { ok=1; break; }; echo "$(date -u +%FT%TZ) waiting" >> $OUT/wait.log; sleep 30; done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" > $OUT/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
for arm in "" "--broken"; do
  free || { echo "CARD TAKEN $(date -u +%FT%TZ)" >> $OUT/run.log; exit 5; }
  env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 300 \
    $PY perf/r4t_perf/ns_device.py --shapes $SH --arm gram $arm --out $OUT/dev${arm:+_broken}.json > $OUT/dev${arm:+_broken}.log 2>&1
  echo "dev$arm RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
done
OMP_NUM_THREADS=8 nice -n 10 timeout 5400 $PY -I perf/r4t_perf/b1prime.py --shapes $SH --dev $OUT/dev.json --broken $OUT/dev_broken.json \
  --out $OUT/b1prime.json > $OUT/grade.log 2>&1
echo "grade RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
