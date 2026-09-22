#!/usr/bin/env bash
# of3t-cotterm: the float64 reference's APB-boundary operands, host-only on qb1.
# Same producer, tree, boundary and cotangent as `of3t-apbleaf/REF_LN_N384.json`; the control is
# that the 2,736 parameter gradients come back bit-identical to the banked `ref_f64_n384.pt`.
#   refrun.sh <c64|n384>
set -uo pipefail
D=/home/ttuser/of3t_frame384
W=/home/ttuser/.coworker/wt/of3t-cotterm
O=/tmp/of3t/of3t-cotterm
mkdir -p "$O"
export PYTHONPATH=$D/ref:$D/deps
cd "$D"
case "$1" in
  c64)  B=boundary_c64.pt;  CROP=64  ;;
  n384) B=boundary_n384.pt; CROP=384 ;;
  *) echo "unknown scope $1"; exit 2 ;;
esac
echo "=== refapb $1 f64 start $(date -u +%FT%TZ) host=$(hostname) ==="
OMP_NUM_THREADS=16 timeout 5400 /home/ttuser/tt-bio-dev/env/bin/python \
  "$W/perf/of3t_cotterm/refapb.py" --apb-out "$O/apb_ref_$1.pt" --real-rows 56 \
  --keep-z-block 44 \
  --tree "$D/of3pkg043" --boundary "$B" --cap-last block47_boundary.pt \
  --policy f64 --crop "$CROP" --threads 16 --checkpoint \
  --ln-out "$O/ln_ref_$1.pt" --out "$O/ref_ln_$1.pt" \
  --report "$W/perf/of3t_cotterm/REF_APB_$1.json" 2>&1 \
  | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)"
echo "=== refapb exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
ls -la "$O/apb_ref_$1.pt" "$O/ln_ref_$1.pt" 2>/dev/null
