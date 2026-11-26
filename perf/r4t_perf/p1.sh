#!/bin/bash
# P1 (candidate pair on the card, serving path): 9 dumped batches on the no-update arm with --cand-pair card,
# training default levers. Grade like D43 (grade_parity, only b08 insertion may fail) and compare winners and
# per-batch |total - upstream| with the host-pair twin (run9x arm `five` = default build).   p1.sh CARD
C=${1:?card}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/p1; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
echo "START $(date -u +%FT%TZ) card $C sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $OUT/run.log
for b in 00 01 02 03 04 05 06 07 08; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  env -u TT_BIO_LEVERS PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 --cand-pair card \
    --batch $D/batch$b.npz --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck$b > $OUT/P_batch$b.log 2>&1
  echo "RC_P_$b $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck$b
done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
