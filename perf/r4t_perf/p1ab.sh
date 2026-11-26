#!/bin/bash
# P1 timing: --cand-pair host (A) vs card (B), interleaved x3, batch00, 3 steps; also pinned to 2 CPUs (Galaxy share).
#   p1ab.sh CARD CPUS
C=${1:?card}; CPUS=${2:?cpus}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/p1ab; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
echo "START $(date -u +%FT%TZ) card $C cpus $CPUS sha $(git rev-parse --short HEAD)" >> $OUT/run.log
for i in 1 2 3; do for arm in A B; do
  for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && break; sleep 20; done
  cp=host; [ $arm = B ] && cp=card
  taskset -c $CPUS env OMP_NUM_THREADS=2 -u TT_BIO_LEVERS PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C \
    TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 1500 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 \
    --steps 3 --cand-pair $cp --batch $D/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv --out $OUT/ck > $OUT/${arm}$i.log 2>&1
  echo "RC_${arm}$i $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck
done; done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
