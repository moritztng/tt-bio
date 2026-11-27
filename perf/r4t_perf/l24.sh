#!/bin/bash
# L2+L4 default-on grade: PairUpdate block VJP vs float64 (r4t-tape's block_gradcheck.py on this tree), bf16 arm,
# training levers, L2/L4 off vs on, plus the broken arm with L2/L4 on (must FAIL).   l24.sh CARD
C=${1:?card}; T=$HOME/r4t/perf/l24/tree; OUT=$HOME/r4t/perf/l24; PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
cd $T; echo "START $(date -u +%FT%TZ) tree $(cd ~/.coworker/wt/r4t-perf && git rev-parse --short HEAD) + tape $(cd ~/.coworker/wt/r4t-perf && git rev-parse --short origin/wk/r4t-tape) block_gradcheck" >> $OUT/run.log
for arm in off on brk; do
  F=""; E="TT_BIO_LNBW_WEIGHTS=0 TT_BIO_MMVJP_MIN=0"
  [ $arm != off ] && E="TT_BIO_LNBW_WEIGHTS=1 TT_BIO_MMVJP_MIN=1"; [ $arm = brk ] && F="--break"
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  env $E PYTHONPATH=$T TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    timeout 1500 $PY perf/r4t_tape/block_gradcheck.py --module pair_update --n 128 --real 96 --probe 8 --seed 7 --arm bf16 $F \
    --out $OUT/$arm.json > $OUT/$arm.log 2>&1
  echo "RC_$arm $? $(date -u +%T)" >> $OUT/run.log
done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
