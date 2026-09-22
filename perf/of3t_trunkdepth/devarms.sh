#!/usr/bin/env bash
# The device arms of the seven-depth ladder. Card 1, this worker's grant, crop 64, one block
# alone at each depth: that block's own captured input in, that block's own captured output
# cotangent back, no chaining and no error arriving from above. scale_pair_bias=False on every
# arm; no shipped default is touched.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-trunkdepth
O=/home/ttuser/of3t_trunkdepth
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$W" || exit 1
export PYTHONPATH="$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:of3t-trunkdepth
MAN=$O/MANIFEST_bundle_min_462c1f52.json
COMMON=(--crop 64 --manifest-json "$MAN" --cap "$O/cap_ladder"
        --capture-report perf/of3t_trunkdepth/CAPTURE_LADDER.json
        --out-dir perf/of3t_trunkdepth --scale-pair-bias off)

one() {  # block tag [extra...]
  local k=$1 tag=$2; shift 2
  echo "=== $tag block $k $* $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py "${COMMON[@]}" \
      --block "$k" --forward-reference "$O/REF_b${k}_f64_fwd.pt" --tag "$tag" \
      --dump-grads "$O/$tag.pt" --dump-forward "$O/${tag}_fwd.pt" \
      --dump-input-grads "$O/${tag}_ig.pt" "$@" \
      2>&1 | grep -vE "^Config\{|Initial ttnn.CONFIG|UserWarning|warnings.warn"
  echo "=== $tag exit ${PIPESTATUS[0]} ==="
}

for x in "$@"; do
  case "$x" in
    break0)  one 0  DEV_BREAK_b0  --permute-cot 20260920 ;;
    break47) one 47 DEV_BREAK_b47 --permute-cot 20260920 ;;
    b*)      one "${x#b}" "DEV_${x}" ;;
    *) echo "unknown arm $x"; exit 2 ;;
  esac
done
echo "DEVARMS_ALLDONE $(date -u +%FT%TZ)"
