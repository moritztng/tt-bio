#!/bin/bash
# partial diffusion, rerun: upstream's partial config needs a GENERATED annotated complex
# (examples/antibody_design/README.md step 3), not the example input CIF. The first attempt
# fed it the input and died in structure export with "NaN values detected ... coord group 1".
set -uo pipefail
ROOT=/home/moritz/rfd4-ref-env
AB=$ROOT/src/RFD4-Proteina/examples/antibody_design
PARENT=$ROOT/runs/nanobody_cdr/inference_samples/mzb1_7AAH__h-NbBCII10_cdr123/mzb1_7AAH__h-NbBCII10_cdr123_generated.cif
export CAPTURE_FLAGS="--per-class 4 --per-module 1 --sampler-steps 3 --config-dir $AB --config-name partial"
echo "=== partial_diffusion $(date -u +%FT%TZ)"
timeout 7200 "$ROOT/harness/run_capture.sh" partial_diffusion - 10 "inputs=$PARENT" 2>&1 | tail -40
echo "--- partial_diffusion rc=$?"
echo "CAPTURE_PARTIAL_DONE $(date -u +%FT%TZ)"
