#!/bin/bash
# Every device-worthy input in perf/bgx_inputs/round.py, one process each, card 3 on qb2.
# One process per case because ttnn takes one device context per process, and a case that
# wedges must not take the rest of the batch with it: `timeout` bounds each one.
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_inputs/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3}
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bgx-inputs
P=/home/ttuser/bcx_e2e_venv/bin/python3
for case in "$@"; do
    echo "=== $case $(date -u +%H:%M:%SZ)"
    timeout 1500 $P -u perf/bgx_inputs/round.py --case "$case" --rounds 2 \
        --card "$TT_VISIBLE_DEVICES" 2>&1 | tail -25
    echo "=== $case rc=$? $(date -u +%H:%M:%SZ)"
done
echo "=== batch done $(date -u +%H:%M:%SZ)"
