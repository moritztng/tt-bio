#!/bin/bash
# One arm of the step-gap measurement. arm_name world global_batch prefetch steps chips
set -u
ARM=$1; WORLD=$2; GB=$3; PF=$4; STEPS=$5; CHIPS=$6
WT=/home/ttuser/.coworker/wt/train-s-stepgap
cd $WT
source /home/ttuser/tt-bio-dev/env/bin/activate
OUT=$WT/perf/train_s_stepgap/$ARM
rm -rf "$OUT"; mkdir -p "$OUT"
IFS=',' read -ra CA <<< "$CHIPS"
pids=()
for r in $(seq 0 $((WORLD-1))); do
  TT_VISIBLE_DEVICES=${CA[$r]} TT_BIO_LEASE_CARDS=$CHIPS \
  TT_BIO_LEASE_HOLDER=worker:train-s-stepgap PYTHONPATH=$WT \
  python3 scripts/abb3_port/repro.py --out "$OUT" --steps $STEPS \
    --global-batch $GB --micro 4 --tokens 256 --blocks 8 --seed 0 \
    --data sabdab --split train --released-true /home/ttuser/abb3_data/base-loss/true \
    --rank $r --world $WORLD --chips $CHIPS --prefetch $PF --no-resume \
    --checkpoint-minutes 600 --rendezvous /dev/shm/abb3-stepgap-$ARM \
    > "$OUT/rank$r.log" 2>&1 &
  pids+=($!)
done
echo "arm $ARM pids ${pids[*]}"
for p in "${pids[@]}"; do wait $p; echo "rank pid $p exit $?"; done
