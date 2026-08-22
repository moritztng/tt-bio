#!/usr/bin/env bash
# Drive the whole OpenBind-0 reference sweep on a rented box.
#
#   bash gpu_ob_sweep.sh ob        # OpenBind-0, openfold3 0.5.0
#   bash gpu_ob_sweep.sh p2        # OpenFold3 preview2, openfold3 0.4.5
#
# One process per cell, so a cell that OOMs or crashes cannot take the sweep with it, and the
# 2.3 GB checkpoint load is paid once per cell rather than once per fold.
#
# `samples` is --num-diffusion-samples. 1 is the single-structure point every other model on the
# perf page runs; 5 is OpenFold3's own shipped default and the ensemble point. A TT number is
# comparable only against the row with the same length AND the same sample count.
set -uo pipefail
ARM=${1:?arm: ob or p2}
PY=/root/venv-$ARM/bin/python
case "$ARM" in
  ob) CKPT=/root/ckpt/of3-ob-2025-06-30-174k.pt ;;
  p2) CKPT=/root/ckpt/of3-p2-155k.pt ;;
  *)  echo "unknown arm $ARM" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
IN=$HERE/inputs
RES=/root/results/$ARM
REPS=${REPS:-3}
mkdir -p "$RES"

cells=(
  "ob_apo_128 1"  "ob_apo_256 1"  "ob_apo_512 1"  "ob_apo_768 1"  "ob_apo_1024 1"
  "ob_lig_s_298 1" "ob_lig_m_298 1" "ob_lig_l_298 1" "ob_lig_m_512 1"
  "ob_apo_512 5"  "ob_lig_m_298 5"
)

for cell in "${cells[@]}"; do
  set -- $cell
  name=$1 samples=$2
  tag=${name}_s${samples}
  out=$RES/$tag.json
  if [ -s "$out" ] && grep -q '"ok": true' "$out"; then
    echo "== skip $tag (already ok) =="
    continue
  fi
  echo "== $ARM $tag : $(date -u +%FT%TZ) =="
  "$PY" "$HERE/gpu_ob_run.py" --spec "$IN/$name.spec.json" --ckpt "$CKPT" \
    --work "/root/work/$ARM/$tag" --report "$out" --reps "$REPS" --samples "$samples" \
    --label "$tag" > "$RES/$tag.log" 2>&1
  rc=$?
  echo "-- rc=$rc $(grep -o '"ok": [a-z]*' "$out" 2>/dev/null | head -1)"
  tail -3 "$RES/$tag.log"
  rm -rf "/root/work/$ARM/$tag/out"        # CIFs are large and the numbers are in the report
done
echo "== sweep $ARM done : $(date -u +%FT%TZ) =="
