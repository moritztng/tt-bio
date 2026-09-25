#!/usr/bin/env bash
# The whole ladder, in the order the pieces depend on each other: the float64 and bf16 CPU
# references first (the device arms need each depth's f64 forward to check against), then the
# masked activation norms off the captured boundaries, then the device arms, then the scoring
# and the pre-registered correlation test.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkdepth
O=/home/ttuser/of3t_trunkdepth
cd "$W" || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python

echo "##### REFARMS $(date -u +%FT%TZ)"
bash perf/of3t_trunkdepth/refarms.sh

echo "##### NORMS $(date -u +%FT%TZ)"
PYTHONPATH="$W" OMP_NUM_THREADS=8 "$PY" perf/of3t_trunkdepth/norms.py \
    --out perf/of3t_trunkdepth/NORMS.json

echo "##### DEVARMS $(date -u +%FT%TZ)"
bash perf/of3t_trunkdepth/devarms.sh b0 b8 b16 b23 b32 b40 b47 break47

echo "##### SCOREARMS $(date -u +%FT%TZ)"
bash perf/of3t_trunkdepth/scorearms.sh

echo "##### LADDER $(date -u +%FT%TZ)"
PYTHONPATH="$W" "$PY" perf/of3t_trunkdepth/ladder_report.py

echo "CHAIN_ALLDONE $(date -u +%FT%TZ)"
