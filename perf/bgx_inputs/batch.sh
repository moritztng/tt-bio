#!/bin/bash
# Every device-worthy input in perf/bgx_inputs/round.py, one process each, on the card this
# row holds. One process per case because ttnn takes one device context per process, and a
# case that wedges must not take the rest of the batch with it: `timeout` bounds each one.
# Card, interpreter and BindCraft 2 checkout all come from the environment, because the
# matrix has to run on a Wormhole Galaxy as well as on qb1 and qb2 and none of the three
# agree on any of them. The defaults are qb1's, so a bgx re-run there is unchanged:
#   TT_VISIBLE_DEVICES=30 BCX_PY=~/bwx/venv/bin/python BCX_BC2=~/bwx/bc2 \
#   BCX_AF2=~/bwx/af2_params perf/bgx_inputs/batch.sh <case>...
set -uo pipefail
cd "$(dirname "$0")/../.."
export BCX_BC2=${BCX_BC2:-/home/ttuser/bcx_e2e/bc2}
export BCX_AF2=${BCX_AF2:-/home/ttuser/bcx_e2e/af2_params}
export PYTHONPATH=$PWD:$BCX_BC2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_inputs/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3}
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bgx-inputs
P=${BCX_PY:-/home/ttuser/bcx_e2e_venv/bin/python3}
for case in "$@"; do
    echo "=== $case $(date -u +%H:%M:%SZ)"
    timeout 1500 $P -u perf/bgx_inputs/round.py --case "$case" --rounds 2 \
        --card "$TT_VISIBLE_DEVICES" 2>&1 | tail -25
    echo "=== $case rc=$? $(date -u +%H:%M:%SZ)"
done
echo "=== batch done $(date -u +%H:%M:%SZ)"
