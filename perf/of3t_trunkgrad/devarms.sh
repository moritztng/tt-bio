#!/usr/bin/env bash
# The three device arms: the shipped convention, the pre-scaled convention, and the break control.
# crop 64, 48 blocks as one taped stack, card 0.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkgrad
O=/home/ttuser/of3t_trunkgrad
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W"
export PYTHONPATH="$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-trunkgrad
MAN=/home/ttuser/of3t_pairformer/MANIFEST_bundle_min_462c1f52.json
FREF=$O/REF043_F64_fwd.pt
COMMON=(--block 0 --stack 48 --crop 64 --manifest-json "$MAN"
        --out-dir perf/of3t_trunkgrad --forward-reference "$FREF")

arm() {  # tag spb [extra...]
  local tag=$1 spb=$2; shift 2
  echo "=== $tag spb=$spb $* $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py "${COMMON[@]}" \
      --scale-pair-bias "$spb" --tag "$tag" \
      --dump-grads "$O/$tag.pt" --dump-forward "$O/${tag}_fwd.pt" "$@" \
      2>&1 | grep -vE "^Config\{|Initial ttnn.CONFIG|UserWarning|warnings.warn"
  echo "=== $tag exit ${PIPESTATUS[0]} ==="
}

arm DEV_SHIPPED off
arm DEV_SCALED  on
arm DEV_BREAK   off --permute-cot 20260920
echo "DEVARMS_ALLDONE $(date -u +%FT%TZ)"
