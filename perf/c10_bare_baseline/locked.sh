#!/usr/bin/env bash
set -euo pipefail
OUT=${1:?output required}
# The unchanged shared lock logs a load timeout but continues. Reject that here.
if grep -q 'benchlock: WARNING\|benchlock: TIMED OUT' "$OUT/launch.log"; then exit 75; fi
for SIZE in 512 298; do
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c10-bare-baseline \
    python3 perf/c10_bare_baseline/capture.py --size "$SIZE" --out "$OUT/$SIZE"
done
