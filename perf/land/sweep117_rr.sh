#!/bin/bash
# Exec pass 4: the 117 aa performance question (state doc §16.1).
#
# §16.1 killed the previous 117 aa ladder because the arms ran SEQUENTIALLY while two other legs'
# parity gates folded on cards 0 and 2 of this host. A load ramp then maps directly onto arm order,
# which is why protenix-v2 showed a monotone regression and opendde a gain in the same run.
#
# Waiting for an idle host is not available: this box has four cards and several live workers, and
# card 3 has a co-tenant right now. So instead of removing the load, make the arms share it --
# round-robin L0 / L2 / L4 within each round, several rounds, compare per-round pairs and medians.
# A drift that is monotone in wall-clock now lands on every arm equally instead of on the last one.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TT_BIO_DRAM_PEAK
PY=/usr/bin/python3
LOG="$WT/perf/land/out/sweep117rr.log"

run_arm () {  # tag tree expect model repeat
  # Record what else is on the box, so a contaminated round can be identified rather than guessed.
  local ld holders
  ld=$(cut -d' ' -f1-3 /proc/loadavg)
  holders=$(ls -l /proc/*/fd 2>/dev/null | grep -c tenstorrent)
  echo "=== $(date -u +%H:%M:%S) arm $1 $4 117 loadavg=$ld tt_fds=$holders" >>"$LOG"
  timeout 600 "$PY" perf/land/fold_arm.py --tag "$1" --tree "$2" --expect "$3" \
      --fused 0 --model "$4" --size 117 --repeat "$5" >>"$LOG" 2>&1
  echo "=== $(date -u +%H:%M:%S) arm $1 rc=$?" >>"$LOG"
}

for r in 1 2 3; do
  echo "########## round $r" >>"$LOG"
  run_arm "r${r}L0"  arms/L0 83499742 protenix-v2 3
  run_arm "r${r}L2"  arms/L2 c42ed26a protenix-v2 3
  run_arm "r${r}L4"  arms/L4 0e9ee663 protenix-v2 3
  run_arm "r${r}L0o" arms/L0 83499742 opendde     3
  run_arm "r${r}L4o" arms/L4 0e9ee663 opendde     3
done

echo "=== $(date -u +%H:%M:%S) SWEEP117RR COMPLETE" >>"$LOG"
