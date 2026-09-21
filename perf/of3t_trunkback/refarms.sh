#!/usr/bin/env bash
# The float64 reference arms this row scores against: the 48-block trunk (re-run as a control
# that this row reproduces its predecessor bit for bit) and each of the three captured
# boundaries as a SINGLE block, which is the depth decomposition. CPU only.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkback
O=/home/ttuser/of3t_trunkback
PY=/home/ttuser/tt-bio-dev/env/bin/python
T043=/home/ttuser/of3t_trunk043ref/of3pkg043
BC=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
mkdir -p "$O"
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W"
export OMP_NUM_THREADS=8

stack() {  # name policy
  echo "=== $1  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_trunkgrad/ref_grad.py --tree "$T043" --policy "$2" --crop 64 \
      --boundary-check "$BC" \
      --out "$O/$1.pt" --forward-out "$O/${1}_fwd.pt" \
      --input-grad-out "$O/${1}_ig.pt" \
      --report "perf/of3t_trunkback/$1.json" 2>&1 | grep -vE "UserWarning|warnings.warn|  from openfold3"
  echo "=== $1 exit ${PIPESTATUS[0]} ==="
}

one() {  # block policy
  local k=$1 pol=$2 n="REF_b$1_$2"
  echo "=== $n  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_trunkgrad/ref_grad.py --tree "$T043" --policy "$pol" --crop 64 \
      --first "$k" --blocks 1 \
      --out "$O/$n.pt" --forward-out "$O/${n}_fwd.pt" \
      --input-grad-out "$O/${n}_ig.pt" \
      --report "perf/of3t_trunkback/$n.json" 2>&1 | grep -vE "UserWarning|warnings.warn|  from openfold3"
  echo "=== $n exit ${PIPESTATUS[0]} ==="
}

stack REF043_F64_48  f64
stack REF043_BF16_48 bf16auto
for k in 0 23 47; do one "$k" f64; one "$k" bf16auto; done
echo "REFARMS_ALLDONE $(date -u +%FT%TZ)"
