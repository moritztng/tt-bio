#!/bin/bash
# G1 (fused gate backward): full trunk step, A = --gate-bw off / B = on, interleaved x3, batch00, 3 steps, 2 pinned CPUs.
# Waits for the gateg grade (pid $3) so the card is ours alone.   gtab.sh CARD CPUS [WAITPID]
C=${1:?card}; CPUS=${2:?cpus}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/gtab; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
[ -n "$3" ] && while kill -0 $3 2>/dev/null; do sleep 20; done
wait_card() { for w in $(seq 1 90); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; sleep 20; done; }
run() {  # run NAME on|off
  wait_card
  taskset -c $CPUS env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED OMP_NUM_THREADS=2 PYTHONPATH=$WT \
    TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 1500 \
    $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 3 --batch $D/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv \
    --out $OUT/ck --gate-bw $2 > $OUT/$1.log 2>&1
  echo "RC_$1 $? $(date -u +%T) load $(cut -d" " -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck
}
echo "START $(date -u +%FT%TZ) card $C cpus $CPUS sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $OUT/run.log
for i in 1 2 3; do run A$i off; run B$i on; done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
