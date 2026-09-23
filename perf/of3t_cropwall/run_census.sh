#!/usr/bin/env bash
# One concat census rung on card 2. Bytes, not times: no quiet-box wait is implied.
# Usage: run_census.sh <tokens> [extra args...]
set -u
N=$1; shift || true
WT=/home/ttuser/.coworker/wt/of3t-cropwall
cd "$WT"
exec env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-cropwall \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_cropwall/concat_census.py \
    --tokens "$N" --out perf/of3t_cropwall/out/concat_"$N".json "$@"
