#!/bin/bash
# Price the D43 training lever drop: interleaved A (full serving list, = old default P) / B (training default,
# 5 serving levers off), batch00, 4 steps each, n=3 pairs. Read median step_s/fwd_s/bwd_s of steps 2-4, AICLK during.
#   abdrop.sh CARD
C=${1:?card}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/abdrop; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
lev=$(tr " " "\n" < $WT/perf/r4t_perf/bis/levers.txt | grep -v "^$" | paste -sd,)
mkdir -p $OUT; cd $WT
echo "START $(date -u +%FT%TZ) card $C sha $(git rev-parse --short HEAD)" >> $OUT/run.log
for i in 1 2 3; do for arm in A B; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  if [ $arm = A ]; then L="TT_BIO_LEVERS=$lev"; else L="R4T_NOOP=1"; fi
  env -u TT_BIO_LEVERS $L PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 4 \
    --batch $D/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck > $OUT/${arm}$i.log 2>&1
  echo "RC_${arm}$i $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck
done; done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
