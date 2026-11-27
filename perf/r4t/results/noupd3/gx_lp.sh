#!/bin/bash
# r4t-accuracy D85: LOSS-PARITY no-update arm P (run_noupd.sh protocol, --steps 1) on the landing tree
# (attn_fp32 default on), anx-glx chips 0-8, batch N on chip N, all in parallel. Done flag: END in run.log.
set -u; SRC=~/fleet/r4t/lp-attn32/src; D=~/fleet/r4t/step; OUT=~/fleet/r4t/lp-attn32/out
mkdir -p $OUT; cd $SRC || exit 1; sha=$(cat .sha)
[ -e ~/fleet/HANDBACK ] && { echo "HANDBACK $(date -u +%FT%TZ)" >> $OUT/run.log; exit 4; }
echo "START $(date -u +%FT%TZ) host anx-glx chips 0-8 src $sha load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/run.log
( while :; do echo "$(date -u +%T) $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/load.log; sleep 15; done ) & LP=$!
for b in 0 1 2 3 4 5 6 7 8; do
  ( env PYTHONPATH=$SRC OMP_NUM_THREADS=4 TT_VISIBLE_DEVICES=$b TT_BIO_LEASE_CARDS=$b TT_BIO_LEASE_HOLDER=worker:r4t-accuracy \
      TT_BIO_SDPA_OWN_FORWARD=1 TT_BIO_SHA=$sha TT_BIO_RFD4=~/fleet/r4t/rfd4w \
      timeout 1800 ~/fleet/env/bin/python perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 --batch $D/batch0$b.npz \
      --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck_P0$b > $OUT/P_batch0$b.log 2>&1
    echo "RC_P_0$b $? chip $b $(date -u +%T)" >> $OUT/run.log; rm -rf $OUT/ck_P0$b ) &
done
wait $(jobs -p | grep -v "^$LP$") 2>/dev/null
for j in $(jobs -p); do [ $j != $LP ] && wait $j; done
kill $LP
echo "END $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/run.log
