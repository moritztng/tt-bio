#!/bin/bash
# One WHOLE campaign on a researcher-shaped target: hPDL1 with its 60-70 loop unresolved, the
# hotspots that survive it, every design stage, ProteinMPNN, the validation ensemble and the
# acceptance filters, on the shipped defaults. The round batch answers "does it run"; this
# answers "are the designs sane".
#   design.sh <case> <max_trajectories> [pid-to-wait-for]
set -uo pipefail
cd "$(dirname "$0")/../.."
case=${1:-gap}; budget=${2:-4}; wait_for=${3:-}
if [ -n "$wait_for" ]; then
    while kill -0 "$wait_for" 2>/dev/null; do sleep 20; done
    echo "=== $wait_for finished, card free $(date -u +%H:%M:%SZ)"
fi
export PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_inputs/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3}
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bgx-inputs
echo "=== design $case budget=$budget $(date -u +%H:%M:%SZ)"
/home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_inputs/round.py --case "$case" --real \
    --max-trajectories "$budget" --designs 1 --out "$PWD/perf/bgx_inputs/out" \
    --card "$TT_VISIBLE_DEVICES"
echo "=== design $case rc=$? $(date -u +%H:%M:%SZ)"
