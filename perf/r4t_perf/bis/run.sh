#!/bin/bash
# D43(2): one-lever-off bisect of RFD4 NORMAL_LEVERS on the no-update arm, batches 00 and 07 (+08).
# Arms: P (no env), ALL (explicit full list: must equal P), C (none), then ALL minus one lever each.
B=$HOME/r4t/perf/bis; SRC=$B/src; OUT=$B/out; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $SRC
ALL=$(tr " " "\n" < $B/levers.txt | grep -v "^$")
full=$(echo $ALL | tr " " ",")
arms="P ALL C $(echo $ALL)"
echo "START $(date -u +%FT%TZ) src $(cat .sha)" >> $OUT/run.log
for arm in $arms; do
  case $arm in P) lev=;; ALL) lev=$full;; C) lev=none;; *) lev=$(echo $ALL | tr " " "\n" | grep -vx "$arm" | paste -sd,);; esac
  for b in 00 07; do
    [ -s $OUT/${arm}_$b.log ] && grep -q master_digest $OUT/${arm}_$b.log && continue
    for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/0 2>/dev/null)" ] && break; sleep 20; done
    env PYTHONPATH=$SRC TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:r4t-perf \
      TT_BIO_SDPA_OWN_FORWARD=1 ${lev:+TT_BIO_LEVERS=$lev} \
      timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 --batch $D/batch$b.npz \
      --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck_$arm$b > $OUT/${arm}_$b.log 2>&1
    echo "RC_${arm}_$b $? $(date -u +%T)" >> $OUT/run.log; rm -rf $OUT/ck_$arm$b
  done
done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
