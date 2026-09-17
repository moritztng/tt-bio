#!/usr/bin/env bash
# Launch one capture_ab session. Args: <run-name> <size> <arms-json> <reps> [benchlock]
set -euo pipefail
cd /home/ttuser/.coworker/wt/c10-core-grid
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export PYTHONPATH=$PWD
unset TT_METAL_HOME TT_METAL_RUNTIME_ROOT LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
export TMPDIR=$PWD/perf/c10_core_grid/tmp
mkdir -p "$TMPDIR"
NAME=${1:?run name}; SIZE=${2:?size}; ARMS=${3:?arms}; REPS=${4:?reps}; LOCK=${5:-nolock}
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || exit 64
OUT=$PWD/perf/c10_core_grid/runs/$NAME
mkdir -p "$(dirname "$OUT")"
RUN=(env C10_NODE=1 TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:c10-core-grid
     python3 perf/c10_core_grid/capture_ab.py --size "$SIZE" --out "$OUT" --arms "$ARMS" --reps "$REPS")
if [[ "$LOCK" == "lock" ]]; then
  BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-600} BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-300} \
    /home/ttuser/.coworker/scripts/benchlock.sh c10-core-grid -- "${RUN[@]}"
else
  "${RUN[@]}"
fi
