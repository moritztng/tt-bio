#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
cd "$WT" || exit 1
gate=$(ps -eo args= | grep -c "[r]elease_gate")
if [ "$gate" -ne 0 ]; then echo "REFUSING: $gate release_gate process(es) live"; exit 75; fi
export BENCHLOCK_LOAD_WAIT_S=9000 BENCHLOCK_WAIT_S=9000
exec /home/ttuser/.coworker/scripts/benchlock.sh c14-matmul-ceiling -- \
  bash "$WT/perf/c14_matmul_ceiling/session_f4_inner.sh"
