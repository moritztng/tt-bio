#!/bin/bash
# Both suites pinned to THIS row's card. The earlier 68-passed run was pinned to card 2, which is
# not this row's lease; one test in test_bindcraft2.py opens a device, so the suite has to run on
# card 3 to count. Chained behind whatever else holds the card.
set -uo pipefail
cd "$(dirname "$0")/../.."
for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
export PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_inputs/out/xlacache
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bgx-inputs
echo "=== suites on card 3 $(date -u +%H:%M:%SZ)"
timeout 1800 /home/ttuser/bcx_e2e_venv/bin/python3 -m pytest tests/test_bcinputs.py \
    tests/test_bindcraft2.py -q
echo "=== suites rc=$? $(date -u +%H:%M:%SZ)"
