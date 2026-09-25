#!/usr/bin/env bash
# of3t-cotcoh: upstream 0.4.3's own float64 cotangent at the same sites, on the MODEL frame.
# CPU only, no card is opened. `perf/of3t_apbleaf/refrun.sh` with the boundary and the cotangent
# swapped for `of3t-modelframe`'s pair and the capture widened to two families at 48 blocks.
set -uo pipefail
D=/home/ttuser/of3t_frame384
W=/home/ttuser/.coworker/wt/of3t-cotcoh
O=/tmp/of3t/of3t-cotcoh
M=/home/ttuser/of3t_modelframe
mkdir -p "$O"
export PYTHONPATH=$D/ref:$D/deps
cd "$D"
echo "=== refcot model-frame f64 start $(date -u +%FT%TZ) host=$(hostname) ==="
OMP_NUM_THREADS=16 timeout 5400 /home/ttuser/tt-bio-dev/env/bin/python \
  "$W/perf/of3t_cotcoh/refcot.py" \
  --cot-out "$O/cot_ref_model_n384.pt" \
  --cot-report "$W/perf/of3t_cotcoh/CAP_REF_model_n384.json" \
  -- --tree "$D/of3pkg043" \
  --boundary "$M/boundary_model_n384.pt" --cap-last "$M/cot_model_n384.pt" \
  --policy f64 --crop 384 --threads 16 --checkpoint \
  --out "$O/ref_sqnorms_model_n384.pt" \
  --report "$W/perf/of3t_cotcoh/REF_ARM_model_n384.json" 2>&1 \
  | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)"
echo "=== refcot exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
