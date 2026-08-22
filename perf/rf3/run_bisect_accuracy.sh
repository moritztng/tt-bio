#!/usr/bin/env bash
# Re-fold the two RF3 accuracy rows on the fixed numerics.
#
# The rows in docs/implementation-parity.md were measured on the regressed tree, so they have to be
# re-FOLDED, not re-scored: the committed caches are coordinates and --rescore would happily score
# stale ones. --work points at a fresh directory so the committed caches survive, and --ref-cache
# reuses the reference halves, which are CPU torch and arm-independent by construction, so this
# pays only the device rollout.
#
# No benchlock: this is a numerics measurement, not a timing one. Co-tenancy cannot change bits, and
# the box is busy with another worker's perf campaign for the foreseeable future.
set -u
WT=/home/ttuser/.coworker/wt/rf3-perf-regression-512aa-bisect
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
LEASE_ENV="TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-perf-regression-512aa-bisect"
R=$WT/perf/rf3/results
cd "$WT" || exit 1

for fx in ubq_76 7roa_117; do
  echo "=== $fx start $(date -u +%H:%M:%S) ==="
  env PYTHONPATH="$PP" $LEASE_ENV \
    "$PY" scripts/rf3_port/accuracy_cell.py --fixture "$fx" --seeds 0,1,2,3,4 \
      --work "$R/accuracy_${fx}_scopefix" \
      --ref-cache "$R/accuracy_${fx}" \
      --out "$R/accuracy_${fx}_scopefix.json" > "$R/accuracy_${fx}_scopefix.log" 2>&1
  echo "=== $fx exit $? $(date -u +%H:%M:%S) ==="
  tail -12 "$R/accuracy_${fx}_scopefix.log"
done
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
