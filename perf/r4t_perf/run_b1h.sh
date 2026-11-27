#!/bin/bash
# r4t-perf D34: hybrid device NS (min dim >= 64) on 3 real fresh-step snapshots + both broken arms on snapshot 0,
# chip 2 inside our booking, then the CPU grade. Snapshots come from dump_momentum.py (pc), copied to $OUT.
cd ~/.coworker/wt/r4t-perf || exit 1
OUT=~/r4t/perf/b1h; B1P=~/r4t/perf/b1p
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
SH=perf/r4t_perf/rfd4_denoiser_2d_shapes.txt
free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
dev() {  # name, extra args
  for i in $(seq 1 90); do free && break; sleep 20; done
  free || { echo "NO WINDOW $1 $(date -u +%FT%TZ)" >> $OUT/run.log; exit 4; }
  env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 600 \
    $PY perf/r4t_perf/ns_device.py --shapes $SH --arm gram --min-dim 64 --out $OUT/$1.json "${@:2}" > $OUT/$1.log 2>&1
  echo "$1 RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
}
for K in 0 1 2; do dev dev$K --snap $OUT/snap$K.pt; done
dev brk --snap $OUT/snap0.pt --broken
dev it4 --snap $OUT/snap0.pt --iters 4
OMP_NUM_THREADS=8 nice -n 10 timeout 2400 $PY -I perf/r4t_perf/b1h_grade.py --snaps $OUT/snap{0,1,2}.pt \
  --devs $OUT/dev{0,1,2}.json --broken $OUT/brk.json --broken-iter $OUT/it4.json --b1prime $B1P/b1prime.json \
  --out $OUT/b1h.json > $OUT/grade.log 2>&1
echo "grade RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
