#!/bin/bash
# r4t-perf: B1'' (D33) algorithmic broken arm (4 NS iterations) on chip 2, then the CPU re-grade of 146e46937's data.
cd ~/.coworker/wt/r4t-perf || exit 1
OUT=~/r4t/perf/b1pp; mkdir -p $OUT; B1P=~/r4t/perf/b1p
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
SH=perf/r4t_perf/rfd4_denoiser_2d_shapes.txt
free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
ok=; for i in $(seq 1 60); do free && { ok=1; break; }; sleep 20; done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" > $OUT/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 300 \
  $PY perf/r4t_perf/ns_device.py --shapes $SH --arm gram --iters 4 --out $OUT/dev_iter4.json > $OUT/dev_iter4.log 2>&1
echo "iter4 RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
OMP_NUM_THREADS=8 nice -n 10 timeout 1200 $PY -I perf/r4t_perf/b1pp.py --shapes $SH --dev $B1P/dev.json --broken $B1P/dev_broken.json \
  --broken-iter $OUT/dev_iter4.json --b1prime $B1P/b1prime.json --out $OUT/b1pp.json > $OUT/grade.log 2>&1
echo "grade RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
