#!/usr/bin/env bash
# of3t-twoside step 4: the per-block cotangent, upstream-injected-bf16 beside the injected
# float64, on the model frame. The instrument is perf/of3t_cotcoh/refcot.py, unchanged -- it
# wraps ref_grad.build and registers hooks on the two LayerNorm families of every block, masked
# to the 56 real rows.
#
# WHY THIS SURVIVES D242. of3t-twoside's step-1 control showed cot_model_n384.pt is not the
# cotangent that drove grads_f64_043.pt's trunk, so nothing on this frame may be read against
# that reference. These two runs differ ONLY in --policy. Their difference is bf16 against
# float64 inside ONE frame and does not depend on that frame's cotangent being the reference's.
# CPU only, no card is opened.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-twoside
R=/home/ttuser/of3t_twoside
M=/home/ttuser/of3t_modelframe
T=/home/ttuser/of3t_refprec
PY=/home/ttuser/tt-bio-dev/env/bin/python
export PYTHONPATH="$T/of3pkg043:$T/deps:$T/pylibs"
cd "$R"

run() {
  local nm=$1 pol=$2
  echo "=== curve $nm start $(date -u +%FT%TZ) host $(hostname) policy $pol ==="
  OMP_NUM_THREADS=14 nice -n 10 timeout 5400 "$PY" "$W/perf/of3t_cotcoh/refcot.py" \
    --cot-out "$R/cot_${nm}.pt" --cot-report "$R/CAP_${nm}.json" \
    -- --tree "$T/of3pkg043" --boundary "$M/boundary_model_n384.pt" \
    --cap-last "$M/cot_model_n384.pt" --policy "$pol" --crop 384 --threads 14 --checkpoint \
    --out "$R/sqnorms_${nm}.pt" --report "$R/ARM_${nm}.json" 2>&1 \
    | grep -vE 'UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach'
  echo "=== curve $nm exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
}

run curve_f64 f64
run curve_bf16 bf16auto
