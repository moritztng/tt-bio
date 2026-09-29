#!/bin/bash
# The pair again, pinned to card 3, after the pin_card test was taught to name its own pin.
set -uo pipefail
cd /home/ttuser/.coworker/wt/bgx-inputs
export PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_inputs/out/xlacache
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bgx-inputs
echo "=== suites on card 3, run 2 $(date -u +%H:%M:%SZ)"
timeout 1800 /home/ttuser/bcx_e2e_venv/bin/python3 -m pytest tests/test_bcinputs.py \
    tests/test_bindcraft2.py -q
echo "=== suites rc=$? $(date -u +%H:%M:%SZ)"
