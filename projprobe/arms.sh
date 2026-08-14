#!/usr/bin/env bash
set -u
cd /home/ttuser/.coworker/wt/relion-backprojection
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:relion-backprojection
export PYTHONPATH=/home/ttuser/.coworker/wt/relion-backprojection
PY=/home/ttuser/tt-bio-dev/env/bin/python3
run() {
  echo "########## $*"
  $PY projprobe/bproj_e2e.py "$@" 2>&1 \
    | grep -E "INTEGRATED|k slices|DRAM read|DRAM write|^  L1|sha256|slices:|rel L2|wall " \
    | grep -v "^2026"
}
"$@"
