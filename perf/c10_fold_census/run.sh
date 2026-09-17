#!/usr/bin/env bash
# One bounded measurement session on the assigned qb2 node, at a pinned and during-sampled
# 1350 MHz. Usage: bash perf/c10_fold_census/run.sh <run-name> <node> [extra replay args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/home/ttuser/tt-bio-dev/env/bin:$PATH
export PYTHONPATH=$PWD
unset TT_METAL_HOME TT_METAL_RUNTIME_ROOT LD_LIBRARY_PATH
export OMP_NUM_THREADS=2
export TMPDIR=$PWD/perf/c10_fold_census/tmp
mkdir -p "$TMPDIR"
NAME=${1:?run name required}; NODE=${2:?node required}; shift 2
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || exit 64
[[ "$NODE" =~ ^[0-9]$ ]] || exit 64
OUT=$PWD/perf/c10_fold_census/runs/$NAME
TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE TT_BIO_LEASE_HOLDER=worker:c10-fold-census \
  python3 perf/c10_fold_census/replay.py --out "$OUT" --node "$NODE" "$@"
