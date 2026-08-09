#!/bin/bash
# Driver for the qb1 landing-stack sweep, card 3. One device context per process.
# Runs to completion detached; fold_arm.py skips arms whose JSON already exists.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
LOG="$WT/perf/land/out/sweep.log"

run_arm () {  # tag tree expect fused model size repeat
  echo "=== $(date -u +%H:%M:%S) arm $1 fused=$4 $5 $6 repeat=$7" >>"$LOG"
  timeout 900 "$PY" perf/land/fold_arm.py --tag "$1" --tree "$2" --expect "$3" \
      --fused "$4" --model "$5" --size "$6" --repeat "$7" >>"$LOG" 2>&1
  echo "=== $(date -u +%H:%M:%S) arm $1 rc=$?" >>"$LOG"
}

# Session 1, protenix-v2 298 aa, priority order from the plan.
run_arm L4  arms/L4 0e9ee663 0 protenix-v2 298 7
run_arm L4F arms/L4 0e9ee663 1 protenix-v2 298 7
run_arm L1F arms/L1 64e65f6a 1 protenix-v2 298 7
run_arm L2  arms/L2 c42ed26a 0 protenix-v2 298 7
run_arm L3  arms/L3 92c92d9e 0 protenix-v2 298 7

# D6 block A/B: the fold cannot resolve ~140 ms, the in-process block A/B can.
echo "=== $(date -u +%H:%M:%S) D6 block A/B" >>"$LOG"
( cd "$WT/arms/L1" && PYTHONPATH=$PWD timeout 900 "$PY" perf/outputside/block_ab.py \
    --model protenix-v2 --n 298 --warm 3 --reps 9 \
    --out "$WT/perf/land/out/d6_block_pv2_298.json" >>"$LOG" 2>&1 )
( cd "$WT/arms/L1" && PYTHONPATH=$PWD timeout 900 "$PY" perf/outputside/block_ab.py \
    --model opendde --n 298 --warm 3 --reps 9 \
    --out "$WT/perf/land/out/d6_block_odde_298.json" >>"$LOG" 2>&1 )

# Session 2, same arms, second process so cross-session drift is visible.
run_arm L4b  arms/L4 0e9ee663 0 protenix-v2 298 7
run_arm L2b  arms/L2 c42ed26a 0 protenix-v2 298 7
run_arm L3b  arms/L3 92c92d9e 0 protenix-v2 298 7
run_arm L4Fb arms/L4 0e9ee663 1 protenix-v2 298 7
run_arm L1Fb arms/L1 64e65f6a 1 protenix-v2 298 7

# opendde 298 aa for the stack and the isolable arms.
run_arm L0o  arms/L0 83499742 0 opendde 298 5
run_arm L4o  arms/L4 0e9ee663 0 opendde 298 5
run_arm L2o  arms/L2 c42ed26a 0 opendde 298 5
run_arm L3o  arms/L3 92c92d9e 0 opendde 298 5
run_arm L4Fo arms/L4 0e9ee663 1 opendde 298 5

echo "=== $(date -u +%H:%M:%S) SWEEP COMPLETE" >>"$LOG"
