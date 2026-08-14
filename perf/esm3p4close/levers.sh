#!/usr/bin/env bash
# L-B (host-boundary census) and L-A (diffusion trace screen), queued behind the size sweep so the
# two never share the box. Both need benchlock: L-B reports a per-site wall against a 0.45 s gate
# and L-A a per-step wall against an 8.8 ms gate, and co-tenant noise on this box is 1-10 %.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-3p4x-close
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:esmfold2-3p4x-close PYTHONPATH="$WT"

while ! grep -q "SWEEP DONE" perf/esm3p4close/sweep.log; do sleep 10; done
echo "=== sweep done, levers start $(date -Is) ==="

echo "=== L-B xfer census start $(date -Is) ==="
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-3p4x-close -- \
  "$PY" -u perf/esm3p4close/xfer_census.py --size 512 --out perf/esm3p4close/xfer_census_512_c1.json
echo "=== L-B rc=$? end $(date -Is) ==="

echo "=== L-A trace screen start $(date -Is) ==="
TT_BIO_TRACE_REGION_SIZE=1073741824 \
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-3p4x-close -- \
  env TT_BIO_TRACE_REGION_SIZE=1073741824 \
  "$PY" -u perf/esm3p4close/trace_screen.py --size 512 --reps 10 \
    --out perf/esm3p4close/trace_screen_512_c1.json
echo "=== L-A rc=$? end $(date -Is) ==="
echo "=== LEVERS DONE $(date -Is) ==="
