#!/bin/bash
# R3 (card tilize of bf16 rebuild uploads) gate + timing: A = ac5973623 export, B = worktree; interleaved x3,
# batch00, 3 steps, 2 pinned CPUs. Gate: B master digests == A (fb06ed76 / 7b26cd3e / d184d041 at 2 threads).
#   r3ab.sh CARD CPUS
C=${1:?card}; CPUS=${2:?cpus}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/r3ab; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; rm -rf $OUT/srcA; mkdir $OUT/srcA; (cd $WT && git archive ac5973623 tt_bio perf) | tar -x -C $OUT/srcA
echo "START $(date -u +%FT%TZ) card $C cpus $CPUS B=$(cd $WT && git rev-parse --short HEAD) dirty $(cd $WT && git status --short tt_bio | wc -l)" >> $OUT/run.log
for i in 1 2 3; do for arm in A B; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  SRC=$WT; [ $arm = A ] && SRC=$OUT/srcA
  cd $SRC; taskset -c $CPUS env -u TT_BIO_LEVERS OMP_NUM_THREADS=2 PYTHONPATH=$SRC TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C \
    TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 1500 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 \
    --steps 3 --batch $D/batch00.npz --exempt $WT/perf/r4t_step/g2_exempt.tsv --out $OUT/ck > $OUT/${arm}$i.log 2>&1
  echo "RC_${arm}$i $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck
done; done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
