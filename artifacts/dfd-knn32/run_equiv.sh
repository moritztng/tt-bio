#!/usr/bin/env bash
# Equivariance re-run with the neighbour-cap guard present (read-only guard; expected unchanged).
set -u
cd /home/moritz/.coworker/wt/dfd-knn32
TT_VISIBLE_DEVICES= PYTHONPATH="$PWD" nice -n 10 /home/moritz/tt-bio/env/bin/python3.10 \
  -m pytest tests/test_diffdock_equivariance.py -q -s
echo "=== equivariance rc=$? $(date -Is)"
