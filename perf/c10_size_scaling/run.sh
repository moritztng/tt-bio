#!/usr/bin/env bash
set -euo pipefail
cd /home/ttuser/.coworker/wt/c10-size-scaling
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export PYTHONPATH=$PWD
unset TT_METAL_HOME TT_METAL_RUNTIME_ROOT LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
export BENCHLOCK_WAIT_S=60 BENCHLOCK_LOAD_WAIT_S=60
export TMPDIR=$PWD/perf/c10_size_scaling/tmp
mkdir -p "$TMPDIR"
RUN_NAME=${1:?run name required}
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || exit 64
shift
SIZES=("$@")
[[ $# -eq 0 ]] && SIZES=(512 768 640 384)
OUT=$PWD/perf/c10_size_scaling/runs/$RUN_NAME
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
cp perf/c10_size_scaling/criterion.json "$OUT/criterion.json"
cp perf/c10_size_scaling/prediction.json "$OUT/prediction.json"
/home/ttuser/.coworker/scripts/benchlock.sh c10-size-scaling -- \
  bash perf/c10_size_scaling/locked.sh "$OUT" "${SIZES[@]}" > "$OUT/launch.log" 2>&1
