#!/bin/bash
# Both suites pinned to THIS row's card: one test in test_bindcraft2.py opens a device, so a
# suite run on a card the row does not hold does not count. Chained behind whatever else holds
# the card, by pid. Card, interpreter and BC2 checkout come from the environment; the defaults
# are qb1's.
set -uo pipefail
cd "$(dirname "$0")/../.."
for pid in "$@"; do
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
export BCX_BC2=${BCX_BC2:-/home/ttuser/bcx_e2e/bc2}
export BCX_AF2=${BCX_AF2:-/home/ttuser/bcx_e2e/af2_params}
export PYTHONPATH=$PWD:$BCX_BC2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_inputs/out/xlacache
card=${TT_VISIBLE_DEVICES:-3}
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bgx-inputs}
echo "=== suites on card $card $(date -u +%H:%M:%SZ)"
timeout 1800 ${BCX_PY:-/home/ttuser/bcx_e2e_venv/bin/python3} -m pytest tests/test_bcinputs.py \
    tests/test_bindcraft2.py -q
echo "=== suites rc=$? $(date -u +%H:%M:%SZ)"
