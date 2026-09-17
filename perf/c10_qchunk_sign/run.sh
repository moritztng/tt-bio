#!/usr/bin/env bash
set -euo pipefail
cd /home/ttuser/.coworker/wt/c10-qchunk-sign
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export PYTHONPATH=$PWD
unset TT_METAL_HOME TT_METAL_RUNTIME_ROOT LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
export BENCHLOCK_WAIT_S=60 BENCHLOCK_LOAD_WAIT_S=60
export TMPDIR=$PWD/perf/c10_qchunk_sign/tmp
mkdir -p "$TMPDIR"
RUN_NAME=${1:?run name required}
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || exit 64
OUT=$PWD/perf/c10_qchunk_sign/runs/$RUN_NAME
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
cp perf/c10_qchunk_sign/criterion.json "$OUT/criterion.json"
/home/ttuser/.coworker/scripts/benchlock.sh c10-qchunk-sign -- \
  bash perf/c10_qchunk_sign/locked.sh "$OUT" > "$OUT/launch.log" 2>&1
