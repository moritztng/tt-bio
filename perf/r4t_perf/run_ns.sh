#!/bin/bash
# r4t-perf L1: after run_block.sh hands chip 2 back, book a fresh 20-min window and run the device Newton-Schulz arms
# (bf16, f32, bf16 --broken which must FAIL B1), then release. Never opens the card outside its own booking.
cd ~/.coworker/wt/r4t-perf || exit 1
OUT=~/r4t/perf/ns; mkdir -p $OUT; BLK=~/r4t/perf/block/run.log
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python; OWN=fleet-r4t-perf-ns
for i in $(seq 1 360); do grep -qE "^(END|NO WINDOW)" $BLK 2>/dev/null && break; sleep 30; done
ok=
for i in $(seq 1 240); do
  python3 ~/spd/feed/book.py add $OWN 2 1200 --note "r4t-perf device NS arms" >/dev/null 2>&1
  who=$(python3 ~/spd/feed/book.py active 2 2>/dev/null)
  if [ "$who" = "$OWN" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; then ok=1; break; fi
  echo "$(date -u +%FT%TZ) waiting: active=$who" >> $OUT/wait.log; sleep 30
done
[ -n "$ok" ] || { echo "NO WINDOW $(date -u +%FT%TZ)" > $OUT/run.log; python3 ~/spd/feed/book.py rm $OWN; exit 4; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
for arm in "bf16" "f32" "bf16 --broken"; do
  set -- $arm; tag=$1${2:+_broken}
  env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf timeout 300 \
    $PY perf/r4t_perf/ns_device.py --shapes perf/r4t_perf/rfd4_denoiser_2d_shapes.txt --arm $1 $2 --out $OUT/ns_$tag.json > $OUT/ns_$tag.log 2>&1
  echo "$tag RC $? $(date -u +%FT%TZ)" >> $OUT/run.log
done
python3 ~/spd/feed/book.py rm $OWN
echo "END released $(date -u +%FT%TZ)" >> $OUT/run.log
