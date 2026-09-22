#!/usr/bin/env bash
# The CPU reference arms of the seven-depth ladder. For each depth, three policies on the SAME
# captured boundary and the same upstream 0.4.3 tree:
#   f64      the reference every number is scored against
#   bf16pure every parameter and activation bfloat16, no autocast -- THAT DEPTH's own floor
#   bf16auto upstream's shipped recipe, so the floor is not a strawman of our construction
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkdepth
O=/home/ttuser/of3t_trunkdepth
CAP=$O/cap_ladder
PY=/home/ttuser/tt-bio-dev/env/bin/python
T043=/home/ttuser/of3t_trunk043ref/of3pkg043
cd "$W" || exit 1
export PYTHONPATH="/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W"
export OMP_NUM_THREADS=8

for k in 0 8 16 23 32 40 47; do
  for pol in f64 bf16pure bf16auto; do
    n="REF_b${k}_${pol}"
    echo "=== $n  $(date -u +%FT%TZ) ==="
    "$PY" perf/of3t_trunkgrad/ref_grad.py --tree "$T043" --cap "$CAP" --policy "$pol" \
        --crop 64 --first "$k" --blocks 1 \
        --out "$O/$n.pt" --forward-out "$O/${n}_fwd.pt" --input-grad-out "$O/${n}_ig.pt" \
        --report "perf/of3t_trunkdepth/$n.json" \
        2>&1 | grep -vE "UserWarning|warnings.warn|  from openfold3"
    echo "=== $n exit ${PIPESTATUS[0]} ==="
  done
done
echo "REFARMS_ALLDONE $(date -u +%FT%TZ)"
