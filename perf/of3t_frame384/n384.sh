#!/bin/bash
# of3t-frame384 deliverable 1: the two crop-384 local references, from the SAME capture the
# device arms are driven from. Producer is of3t-trunkg043 ref_grad.py (perf/of3t_trunkg043/),
# with --checkpoint added additively; the c64 bit-identity control says the flag is inert.
set -uo pipefail
D=/home/ttuser/of3t_frame384
PY=/home/ttuser/tt-bio-dev/env/bin/python
export PYTHONPATH=$D/ref:$D/deps
cd $D
run() {  # name policy crop threads extra...
  local nm=$1 pol=$2 crop=$3 th=$4; shift 4
  echo "=== $nm $(date -u +%FT%TZ) ==="
  OMP_NUM_THREADS=$th $PY ref_grad.py --tree $D/of3pkg043 \
    --boundary $([ $crop = 64 ] && echo boundary_c64.pt || echo boundary_n384.pt) \
    --cap-last block47_boundary.pt --policy $pol --crop $crop --threads $th "$@" \
    --out $D/${nm}.pt --report $D/${nm^^}.json 2>&1 \
    | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)"
  echo "=== $nm exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
}
case "$1" in
  f64n384)  run ref_f64_n384      f64      384 16 --checkpoint ;;
  bf16n384) run ref_bf16auto_n384 bf16auto 384 16 --checkpoint ;;
  c64bf16)  run c64_bf16_plain    bf16auto 64   4
            run c64_bf16_ckpt     bf16auto 64   4 --checkpoint ;;
esac
