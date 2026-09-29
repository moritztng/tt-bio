#!/bin/bash
# One arm of the relpos cadence A/B.  run_arm.sh <arm> <steps> <chips> [--host-features]
set -u
ARM=$1; STEPS=$2; CHIPS=$3; shift 3
WT=/home/ttuser/.coworker/wt/train-u-relpos-ondevice
cd $WT
source /home/ttuser/tt-bio-dev/env/bin/activate
OUT=$WT/perf/train_u_relpos/$ARM
rm -rf "$OUT"; mkdir -p "$OUT"
IFS=',' read -ra CA <<< "$CHIPS"
pids=()
for r in 0 1; do
  TT_VISIBLE_DEVICES=${CA[$r]} TT_BIO_LEASE_CARDS=$CHIPS \
  TT_BIO_LEASE_HOLDER=worker:train-u-relpos-ondevice PYTHONPATH=$WT \
  python3 perf/train_u_relpos/arm.py --out "$OUT" --steps $STEPS \
    --global-batch 64 --micro 4 --tokens 256 --blocks 8 --seed 0 \
    --data sabdab --split train --released-true /home/ttuser/abb3_data/base-loss/true \
    --rank $r --world 2 --chips $CHIPS --no-resume \
    --checkpoint-minutes 600 --rendezvous /dev/shm/abb3-relpos-$ARM "$@" \
    > "$OUT/rank$r.log" 2>&1 &
  pids+=($!)
done
echo "arm $ARM pids ${pids[*]}"
for p in "${pids[@]}"; do wait $p; echo "rank pid $p exit $?"; done
