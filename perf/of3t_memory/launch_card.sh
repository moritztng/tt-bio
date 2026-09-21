#!/bin/bash
# $1 = card, $2 = sizes, $3 = arms, $4 = out tag, $5.. = extra args
CARD="$1"; SIZES="$2"; ARMS="$3"; TAG="$4"; shift 4
cd /home/ttuser/.coworker/wt/of3t-memory || exit 1
export TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" TT_BIO_LEASE_HOLDER=worker:of3t-memory
export PYTHONPATH=/home/ttuser/.coworker/wt/of3t-memory
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_memory/alloc_profile.py \
  --sizes "$SIZES" --arms "$ARMS" "$@" \
  --out "perf/of3t_memory/out/${TAG}.json"
