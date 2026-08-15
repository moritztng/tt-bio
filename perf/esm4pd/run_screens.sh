#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:esmfold2-to-4x-per-dollar
echo "### xfer_rate $(date -u +%H:%M:%S)"
$PY perf/esm4pd/xfer_rate.py perf/esm4pd/xfer_rate_c0.json 2>&1 | tail -12
echo "### grid13 $(date -u +%H:%M:%S)"
$PY perf/esm4pd/fold_ab4.py --size 512 --rounds 2 --tag grid13 --grid 13x10     --cifdir perf/esm4pd/cif --out perf/esm4pd/fold_grid13_c0.json 2>&1 | tail -25
echo "### done $(date -u +%H:%M:%S)"
