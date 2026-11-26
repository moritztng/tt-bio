#!/bin/bash
# r4t-step: wait for our qb2 chip-2 booking to be live and the node free, then run the trunk-step arms in order.
cd ~/.coworker/wt/r4t-step || exit 1
OUT=~/r4t/step/trunk; mkdir -p $OUT
PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
for i in $(seq 1 240); do
  who=$(python3 ~/spd/feed/book.py active 2 2>/dev/null)
  if [ "$who" = "fleet-r4t-step" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/2 2>/dev/null)" ]; then break; fi
  echo "$(date -u +%FT%TZ) waiting: active=$who" >> $OUT/wait.log; sleep 30
done
R(){ env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-step timeout 1500 $PY perf/r4t_step/trunk_step.py "$@"; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $OUT/run.log
rm -rf $OUT/ckA $OUT/ckR $OUT/ckD
R --n 256 --k 4 --depth 2 --steps 6 --save-at 5 --out $OUT/ckA > $OUT/A.log 2>&1; echo "RC_A $?" >> $OUT/run.log
R --n 256 --k 4 --depth 2 --steps 6 --resume $OUT/ckA/adapter-00000005.safetensors --out $OUT/ckR > $OUT/R.log 2>&1; echo "RC_R $?" >> $OUT/run.log
R --n 256 --k 1 --depth 2 --steps 6 --detach-at 3 --out $OUT/ckD > $OUT/D.log 2>&1; echo "RC_D $?" >> $OUT/run.log
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
