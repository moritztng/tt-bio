#!/bin/bash
# Capture goldens for every RFD4 feature family, one run per (feature, size).
# Module goldens only: generation stops as soon as every hooked module has its capture.
# The finished-design reference (a full CPU trajectory) is a separate, much longer run.
set -uo pipefail
ROOT=/home/moritz/rfd4-ref-env
R=$ROOT/harness/run_capture.sh
export CAPTURE_FLAGS="--per-class 4 --per-module 1 --sampler-steps 3 --stop-after-captures"

run () {  # run <feature> <spec> <seed> [overrides...]
  local f=$1; shift
  echo "=== $f $(date -u +%FT%TZ)"
  timeout 7200 "$R" "$f" "$@" 2>&1 | tail -40
  echo "--- $f rc=$?"
}

run unconditional_128 unconditional_128.yaml 0 recycle_strategy=medium/flat
run unconditional_256 unconditional_256.yaml 0 recycle_strategy=medium/flat
run binder            binder.yaml            1 recycle_strategy=medium/front_loaded
run enzyme            enzyme.yaml            2 recycle_strategy=medium/midpeak
run motif_scaffold    motif_scaffold.yaml    3 recycle_strategy=medium/midpeak
run small_molecule    small_molecule.yaml    4 recycle_strategy=medium/midpeak
run nucleic_acid      nucleic_acid_dna.yaml  5 recycle_strategy=medium/front_loaded
run inverse_folding   inverse_folding.yaml   6 generation=inverse_folding recycle_strategy=medium/flat
echo "CAPTURE_ALL_DONE $(date -u +%FT%TZ)"
