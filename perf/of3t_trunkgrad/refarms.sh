#!/usr/bin/env bash
# The three reference arms. CPU only, one upstream tree per process.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkgrad
O=/home/ttuser/of3t_trunkgrad
PY=/home/ttuser/tt-bio-dev/env/bin/python
T043=/home/ttuser/of3t_trunk043ref/of3pkg043
T050=/home/ttuser/of3t_gradients/of3pkg
BC=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W"
export OMP_NUM_THREADS=8

run() {  # name tree policy
  echo "=== $1  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_trunkgrad/ref_grad.py --tree "$2" --policy "$3" --crop 64 \
      --boundary-check "$BC" \
      --out "$O/$1.pt" --forward-out "$O/${1}_fwd.pt" \
      --report "perf/of3t_trunkgrad/$1.json" 2>&1 | grep -vE "UserWarning|warnings.warn|  from openfold3"
  echo "=== $1 exit ${PIPESTATUS[0]} ==="
}

run REF043_F64  "$T043" f64
run REF050_F64  "$T050" f64
run REF043_BF16 "$T043" bf16auto
echo "REFARMS_ALLDONE $(date -u +%FT%TZ)"
