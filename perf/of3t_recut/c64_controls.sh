#!/usr/bin/env bash
# of3t-recut job 2: the repaired injection's control, in both directions, at crop 64.
#
#   --legacy-total-cotangent  must reproduce of3t-trunkg043's banked REF_F64_c64 arm BIT for BIT
#                             (loss -0.3073181442478619, squared gradient norm
#                             1.8714981803225081, all 2,736 tensors). A flag that merely exists
#                             has tested nothing.
#   default                   must DIFFER, and the difference must be the s_out <- z_out route.
#   default --checkpoint      must be bit-identical to default plain. --checkpoint now leaves the
#                             LAST block eager because the correction needs its graph, so the
#                             flag's inertness is re-earned here rather than inherited.
#
# CPU only, one upstream tree per process, no device is opened.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-recut
O=/home/ttuser/of3t_recut
PY=/home/ttuser/tt-bio-dev/env/bin/python
T043=/home/ttuser/of3t_trunk043ref/of3pkg043
BC=/home/ttuser/of3t_trunk043ref/boundary_c64.pt
CAP=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
cd "$W"
mkdir -p "$O"
export PYTHONPATH="/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps"
export OMP_NUM_THREADS=${THREADS:-8}

run() {  # name extra...
  local nm=$1; shift
  echo "=== $nm start $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_trunkg043/ref_grad.py --tree "$T043" --policy f64 --crop 64 \
      --boundary "$BC" --cap-last "$CAP" \
      --out "$O/$nm.pt" --report "perf/of3t_recut/${nm^^}.json" "$@" 2>&1 \
    | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach"
  echo "=== $nm exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
}

run c64_legacy        --legacy-total-cotangent
run c64_corrected
run c64_corrected_ckpt --checkpoint
