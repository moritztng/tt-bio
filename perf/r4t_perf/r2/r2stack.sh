#!/bin/bash
# D41(2): R2 (card-side bf8 packing) graded as a stack on the no-update arm: default training build (D43 lever
# set, R1 on) + R4T_DEVICE_PACK=1, all 9 dumped batches, same command as run_noupd.sh. Host-packed twin = run9x
# arm `five` (digest-identical to the default build, gate_default.sh).   r2stack.sh CARD
C=${1:?card}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/r2s; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
echo "START $(date -u +%FT%TZ) card $C sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $OUT/run.log
for b in 00 01 02 03 04 05 06 07 08; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  env -u TT_BIO_LEVERS PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 R4T_DEVICE_PACK=1 timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 \
    --batch $D/batch$b.npz --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck$b > $OUT/R_batch$b.log 2>&1
  echo "RC_R_$b $? $(date -u +%T)" >> $OUT/run.log; rm -rf $OUT/ck$b
done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
