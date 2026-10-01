#!/bin/bash
# Run attribution rungs one after another on one card, each its own process.
#   chain.sh <board-tag> target:binder [target:binder ...]
# Env: PY, BCX_BC2, PARAMS, TT_VISIBLE_DEVICES, ROUNDS (default 3).
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
tag=$1; shift
export PYTHONPATH=$root:$BCX_BC2 TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcw-census}
for r in "$@"; do
  t=${r%%:*}; b=${r##*:}; out=$root/perf/bcw_census/out/${tag}_${t}_${b}
  mkdir -p "$out"
  echo "$(date -u +%FT%TZ) start $r" >> "$root/perf/bcw_census/out/${tag}_chain.log"
  "$PY" -u "$root/perf/bcw_census/attrib.py" --every 2 --step-mb 24 -- \
     --target "$t" --binder "$b" --rounds "${ROUNDS:-3}" --trajectories 1 \
     --max-trajectories 1 --final-designs 1 --params "$PARAMS" --out "$out" > "$out/run.log" 2>&1
  echo "$(date -u +%FT%TZ) end $r rc=$?" >> "$root/perf/bcw_census/out/${tag}_chain.log"
done
echo "$(date -u +%FT%TZ) CHAIN DONE" >> "$root/perf/bcw_census/out/${tag}_chain.log"
