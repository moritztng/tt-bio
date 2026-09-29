#!/bin/bash
# One size-ladder rung on qb1.  run_rung.sh <card> <tag> [rung.py args...]
#
# The compile lock is LIVE (JAX_COMPILATION_CACHE_DIR is set and nothing stubs
# `one_worker_compiles` out), which is the case a user gets.
set -euo pipefail
cd "$(dirname "$0")/../.."
card=$1; tag=$2; shift 2
out=perf/bgx_size/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_size/out/xlacache
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:bgx-size
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
