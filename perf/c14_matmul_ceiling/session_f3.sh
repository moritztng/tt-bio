#!/usr/bin/env bash
# Queue the confirming session behind benchlock. BENCHLOCK_LOAD_WAIT_S is raised from its 900 s
# default on purpose: at the default benchlock stops waiting and runs anyway with a warning, which
# is how f1 came to be measured under the release gate.
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
cd "$WT" || exit 1
gate=$(ps -eo args= | grep -c "[r]elease_gate")
if [ "$gate" -ne 0 ]; then
  echo "REFUSING: $gate release_gate process(es) live"; exit 75
fi
export BENCHLOCK_LOAD_WAIT_S=9000 BENCHLOCK_WAIT_S=9000
exec /home/ttuser/.coworker/scripts/benchlock.sh c14-matmul-ceiling -- \
  bash "$WT/perf/c14_matmul_ceiling/session_f3_inner.sh"
