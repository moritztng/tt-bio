#!/bin/bash
# Second wave: the families whose example ships its own config, plus the float64 subset.
# Waits for capture_all.sh to write its end marker, then runs.
set -uo pipefail
ROOT=/home/moritz/rfd4-ref-env
R=$ROOT/harness/run_capture.sh
AB=$ROOT/src/RFD4-Proteina/examples/antibody_design

until grep -q CAPTURE_ALL_DONE "$ROOT/logs/capture_all.log" 2>/dev/null; do sleep 60; done

run () { local f=$1; shift; echo "=== $f $(date -u +%FT%TZ)"; timeout 7200 "$R" "$f" "$@" 2>&1 | tail -30; echo "--- $f rc=$?"; }

# The nanobody and partial-diffusion examples carry their own Hydra config.
export CAPTURE_FLAGS="--per-class 4 --per-module 1 --sampler-steps 3 --config-dir $AB --config-name vhh"
run nanobody_cdr - 9 "inputs=$AB/structures/vhh/mzb1_7AAH__h-NbBCII10_cdr123.cif.zst"
export CAPTURE_FLAGS="--per-class 4 --per-module 1 --sampler-steps 3 --config-dir $AB --config-name partial"
run partial_diffusion - 10 "inputs=$AB/structures/vhh/mzb1_7AAH__h-NbBCII10_cdr123.cif.zst"

# float64 reference for the modules whose bar is in doubt, same inputs, one small case.
export CAPTURE_FLAGS="--per-class 2 --per-module 1 --sampler-steps 1 --stop-after-steps 2 --f64"
run f64_unconditional unconditional_128.yaml 0 recycle_strategy=medium/flat
echo "CAPTURE_REST_DONE $(date -u +%FT%TZ)"
