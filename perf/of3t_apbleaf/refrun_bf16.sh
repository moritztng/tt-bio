#!/usr/bin/env bash
# of3t-apbleaf: the float64 reference's own layer_norm_a operands at padded 384, host-only.
# Same producer, same tree, same boundary and same cotangent as `ref_f64_n384.pt`; the control
# is that the 96 parameter gradients come back bit-identical to that file.
set -uo pipefail
D=/home/ttuser/of3t_frame384
W=/home/ttuser/.coworker/wt/of3t-apbleaf
O=/tmp/of3t/of3t-apbleaf
export PYTHONPATH=$D/ref:$D/deps
cd "$D"
echo "=== refln n384 bf16auto start $(date -u +%FT%TZ) host=$(hostname) ==="
OMP_NUM_THREADS=16 timeout 5400 /home/ttuser/tt-bio-dev/env/bin/python \
  "$W/perf/of3t_apbleaf/refln.py" --tree "$D/of3pkg043" \
  --boundary boundary_n384.pt --cap-last block47_boundary.pt \
  --policy bf16auto --crop 384 --threads 16 --checkpoint \
  --ln-out "$O/ln_ref_bf16_n384.pt" --out "$O/ref_ln_bf16_n384.pt" \
  --report "$W/perf/of3t_apbleaf/REF_LN_BF16_N384.json" 2>&1 \
  | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)"
echo "=== refln exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
