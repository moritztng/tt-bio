#!/bin/bash
# D42(4): the training step at a Galaxy rank's host share (2 CPUs: 64-core host / 32 ranks) vs unpinned,
# interleaved, batch00, 5 steps each, n=3 pairs, training default levers. U = unpinned, T = taskset 2 CPUs +
# OMP/torch threads 2. Read median of steps 2-5 per phase; host load is recorded per run.
#   thr2.sh CARD CPUS   (e.g. thr2.sh 2 6,7)
C=${1:?card}; CPUS=${2:?cpus}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/thr2; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
echo "START $(date -u +%FT%TZ) card $C cpus $CPUS sha $(git rev-parse --short HEAD)" >> $OUT/run.log
for i in 1 2 3; do for arm in U T; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  if [ $arm = T ]; then P="taskset -c $CPUS env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"; else P="env"; fi
  $P env -u TT_BIO_LEVERS PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    TT_BIO_SDPA_OWN_FORWARD=1 timeout 1500 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 5 \
    --batch $D/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck > $OUT/${arm}$i.log 2>&1
  echo "RC_${arm}$i $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck
done; done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
