#!/usr/bin/env bash
# The device arms. Card 1, this worker's grant. crop 64 throughout.
#   *_48   the whole 48-block taped stack, which is the predecessor's scope
#   DEV_bK one block ALONE on its own captured boundary -- no chaining, which is the
#          decomposition this row exists to make
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkback
O=/home/ttuser/of3t_trunkback
PY=/home/ttuser/tt-bio-dev/env/bin/python
mkdir -p "$O"
cd "$W"
export PYTHONPATH="$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:of3t-trunkback
MAN=/home/ttuser/of3t_pairformer/MANIFEST_bundle_min_462c1f52.json
COMMON=(--crop 64 --manifest-json "$MAN" --out-dir perf/of3t_trunkback)

stack() {  # tag spb [extra...]
  local tag=$1 spb=$2; shift 2
  echo "=== $tag spb=$spb $* $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py "${COMMON[@]}" \
      --block 0 --stack 48 --forward-reference "$O/REF043_F64_48_fwd.pt" \
      --scale-pair-bias "$spb" --tag "$tag" \
      --dump-grads "$O/$tag.pt" --dump-forward "$O/${tag}_fwd.pt" --dump-input-grads "$O/${tag}_ig.pt" "$@" \
      2>&1 | grep -vE "^Config\{|Initial ttnn.CONFIG|UserWarning|warnings.warn"
  echo "=== $tag exit ${PIPESTATUS[0]} ==="
}

one() {  # block [extra...]
  local k=$1 tag; tag="DEV_b$1"; shift
  echo "=== $tag $* $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py "${COMMON[@]}" \
      --block "$k" --forward-reference "$O/REF_b${k}_f64_fwd.pt" \
      --scale-pair-bias off --tag "$tag" \
      --dump-grads "$O/$tag.pt" --dump-forward "$O/${tag}_fwd.pt" --dump-input-grads "$O/${tag}_ig.pt" "$@" \
      2>&1 | grep -vE "^Config\{|Initial ttnn.CONFIG|UserWarning|warnings.warn"
  echo "=== $tag exit ${PIPESTATUS[0]} ==="
}

for x in "$@"; do
  case "$x" in
    b0)     one 0 ;;
    b23)    one 23 ;;
    b47)    one 47 ;;
    s48)    stack DEV_SHIPPED_48 off ;;
    break)  stack DEV_BREAK_48   off --permute-cot 20260920 ;;
    *) echo "unknown arm $x"; exit 2 ;;
  esac
done
echo "DEVARMS_ALLDONE $(date -u +%FT%TZ)"
