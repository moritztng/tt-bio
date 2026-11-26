#!/bin/bash
# K4 (NOGRAD_IS_INFERENCE in the grad-off K-candidate pass): 9 batches x 1 step, A = --cand-levers serving,
# B = serving; then a discrimination arm (B with --gate-bw off: digest must differ); then timing A/B x3, batch00,
# 3 steps, 2 pinned CPUs.   k4.sh CARD CPUS
C=${1:?card}; CPUS=${2:?cpus}; WT=$HOME/.coworker/wt/r4t-perf; OUT=$HOME/r4t/perf/k4; D=$HOME/r4t/step
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
mkdir -p $OUT; cd $WT
wait_card() { for w in $(seq 1 60); do [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; sleep 20; done; }
run() {  # NAME BATCH STEPS PIN extra-args...
  local nm=$1 b=$2 st=$3 pin=$4; shift 4; wait_card
  local pre=""; [ $pin = 1 ] && pre="taskset -c $CPUS"
  $pre env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED OMP_NUM_THREADS=2 PYTHONPATH=$WT \
    TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 1500 \
    $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps $st --batch $D/batch$b.npz --exempt perf/r4t_step/g2_exempt.tsv \
    --out $OUT/ck "$@" > $OUT/$nm.txt 2>&1
  echo "RC_$nm $? $(date -u +%T) load $(cut -d" " -f1 /proc/loadavg)" >> $OUT/run.log; rm -rf $OUT/ck
}
echo "START $(date -u +%FT%TZ) card $C sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $OUT/run.log
for b in 00 01 02 03 04 05 06 07 08; do run gA_$b $b 1 0 --cand-levers serving; run gB_$b $b 1 0 --cand-levers inference; done
run gX_00 00 1 0 --cand-levers inference --gate-bw off
for i in 1 2 3; do run tA$i 00 3 1 --cand-levers serving; run tB$i 00 3 1 --cand-levers inference; done
echo "END $(date -u +%FT%TZ)" >> $OUT/run.log
