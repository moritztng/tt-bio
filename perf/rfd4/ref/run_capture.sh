#!/bin/bash
# run_capture.sh <feature> <spec-under-specs/> <seed> [hydra overrides...]
set -euo pipefail
ROOT=/home/moritz/rfd4-ref-env
SRC=$ROOT/src/RFD4-Proteina
FEATURE=$1; SPEC=$2; SEED=${3:-0}; shift 3 || true
export PROJECT_ROOT=$SRC
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
cd "$SRC"
# CAPTURE_FLAGS passes capture.py's own flags, e.g. CAPTURE_FLAGS="--f64 --per-module 2"
# SPEC "-" means the overrides carry the inputs (an example config supplies its own).
INPUTS_ARG=()
[ "$SPEC" = "-" ] || INPUTS_ARG=("inputs=$ROOT/specs/$SPEC")
exec "$ROOT/venv/bin/python" "$ROOT/harness/capture.py" \
  --feature "$FEATURE" --seed "$SEED" ${CAPTURE_FLAGS:-} \
  "${INPUTS_ARG[@]}" "out_dir=$ROOT/runs/$FEATURE" "seed=$SEED" \
  num_replicates=1 batch_size=1 '~metrics' compile_lengths=null "$@"
