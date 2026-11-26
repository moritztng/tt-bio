#!/bin/bash
# D43 wiring gate: the worktree code with NO TT_BIO_LEVERS must reproduce run9x arm `five` (explicit list minus
# the 5 serving levers) digests on batches 00 and 07 (4e0f88d11bbfbe80 / 280c9d47), i.e. trunk_step now trains on the training set by default.
C=${1:?card}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/bis/gd; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
echo "START $(date -u +%FT%TZ) card $C sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $OUT/run.log
for b in 00 07; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  env -u TT_BIO_LEVERS PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 1 \
    --batch $D/batch$b.npz --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck_$b > $OUT/D_batch$b.log 2>&1
  echo "RC_D_$b $? $(date -u +%T)" >> $OUT/run.log; rm -rf $OUT/ck_$b
done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
