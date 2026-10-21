#!/bin/bash
# ~512-token coverage: the gate's largest size. Upstream's benchmark monomers stop at 250
# residues, so specs/monomer-512.cif.zst is the same annotated CIF with its three length
# fields set to 512. Module I/O only (2 sampler steps), not a full design: a 512-residue
# CPU design would run many hours and the goldens the port is graded on are per-module.
set -uo pipefail
ROOT=/home/moritz/rfd4-ref-env
export CAPTURE_FLAGS="--per-class 2 --per-module 1 --sampler-steps 2 --stop-after-steps 2 --stop-after-captures"
echo "=== unconditional_512 $(date -u +%FT%TZ)"
timeout 21600 "$ROOT/harness/run_capture.sh" unconditional_512 unconditional_512.yaml 11 2>&1 | tail -30
echo "--- unconditional_512 rc=$?"
echo "CAPTURE_512_DONE $(date -u +%FT%TZ)"
