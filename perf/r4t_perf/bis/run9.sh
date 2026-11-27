#!/bin/bash
# D43(2) confirmation: the bisect winner (default levers minus transition_b8) on all 9 dumped batches,
# no-update arm, same command as r4t-accuracy run_noupd.sh. Logs named T_batchXX.log for its grader.
#   run9.sh CARD
C=${1:?card}; B=$HOME/r4t/perf/bis; SRC=$B/src; OUT=$B/t9; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
lev=$(tr " " "\n" < $B/levers.txt | grep -v "^$" | grep -vx transition_b8 | paste -sd,)
mkdir -p $OUT; cd $SRC
echo "START $(date -u +%FT%TZ) card $C src $(cat .sha) levers $lev" >> $OUT/run.log
for b in 00 01 02 03 04 05 06 07 08; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  env PYTHONPATH=$SRC TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 TT_BIO_LEVERS=$lev \
    timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 --batch $D/batch$b.npz \
    --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck_T$b > $OUT/T_batch$b.log 2>&1
  echo "RC_T_$b $? $(date -u +%T)" >> $OUT/run.log; rm -rf $OUT/ck_T$b
done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
