#!/bin/bash
# D1 (PairSplit.constant dedup over identical candidate rows): full trunk step, A = DEDUP off / B = on, interleaved
# x3, batch00, 3 steps, 2 pinned CPUs; then B on batches 01..08 step 1 for digest identity vs l2t ON_ logs.  ddab.sh CARD CPUS
C=${1:?card}; CPUS=${2:?cpus}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/ddab; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
wait_card() { for w in $(seq 1 90); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; sleep 20; done; }
run() {  # run NAME DEDUP(0/1) STEPS BATCH
  wait_card
  taskset -c $CPUS env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS OMP_NUM_THREADS=2 PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C \
    TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 1500 $PY -c "
import runpy, sys
from tt_bio.rfd4.pair import PairSplit
PairSplit.DEDUP = bool($2)
sys.argv = ['trunk_step.py', '--n', '256', '--k', '4', '--steps', '$3', '--batch', '$D/batch$4.npz',
            '--exempt', 'perf/r4t_step/g2_exempt.tsv', '--out', '$OUT/ck']
runpy.run_path('perf/r4t_step/trunk_step.py', run_name='__main__')" > $OUT/$1.log 2>&1
  echo "RC_$1 $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck
}
echo "START $(date -u +%FT%TZ) card $C cpus $CPUS sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $OUT/run.log
for i in 1 2 3; do run A$i 0 3 00; run B$i 1 3 00; done
for b in 01 02 03 04 05 06 07 08; do run ON_batch$b 1 1 $b; done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
