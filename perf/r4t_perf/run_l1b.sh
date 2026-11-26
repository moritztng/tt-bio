#!/bin/bash
# r4t-perf: one chip-2 pass inside this row's own booking (fleet-r4t-perf): device Newton-Schulz `gram` arm,
# the same with --broken (must FAIL B1 on exactly the scaled matrix), then the PairUpdate block attribution.
# Never opens the card while anyone else has it open or while our booking is not active; gives up after 60 min.
cd ~/.coworker/wt/r4t-perf || exit 1
OUT=~/r4t/perf/l1b; mkdir -p $OUT ~/r4t/perf/block
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
ENVS="PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf"
free() { python3 ~/spd/feed/book.py active 2 2>/dev/null | grep -qw fleet-r4t-perf && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; }
ok=; for i in $(seq 1 120); do free && { ok=1; break; }; echo "$(date -u +%FT%TZ) waiting: $(python3 ~/spd/feed/book.py active 2)" >> $OUT/wait.log; sleep 30; done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" > $OUT/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
for arm in "" "--broken"; do
  free || { echo "CARD TAKEN before ns$arm $(date -u +%FT%TZ)" >> $OUT/run.log; exit 5; }
  env $ENVS timeout 300 $PY perf/r4t_perf/ns_device.py --shapes perf/r4t_perf/rfd4_denoiser_2d_shapes.txt --arm gram $arm \
    --out $OUT/ns_gram${arm:+_broken}.json > $OUT/ns_gram${arm:+_broken}.log 2>&1
  echo "ns_gram$arm RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
done
free || { echo "CARD TAKEN before block $(date -u +%FT%TZ)" >> $OUT/run.log; exit 5; }
env $ENVS timeout 900 $PY perf/r4t_perf/blockprof.py --n 256 --out ~/r4t/perf/block/block256.json > ~/r4t/perf/block/block.log 2>&1
echo "block RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
