#!/usr/bin/env bash
# of3t-frameself step 4 (brief amendments 4 and 5): the one cell the campaign never filled.
#
#                       replayed graph        ORIGINAL graph
#   cotangent drive     --selftest  0.7945    --graphdrive   <- this
#   loss drive          n/a                   --blockprobe   3.04e-15 vs the reference
#
# Same batch, same checkpoint, same draws as the capture, loss still gated bit-identical to
# 1.267624369070698. No full backward, so no gradient-norm gate and no witness: the loss gate is
# what certifies this is the reference's own step.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-frameself
cd "$W"
source perf/refpath.sh
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" /home/ttuser/of3t_frame384/deps)"
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
O=/home/ttuser/of3t_frameself
mkdir -p "$O"
ref_require "$REF_BUNDLE/batch_step003.pt" "$REF_BUNDLE/draws_recycles0.pt" \
            "$REF_BUNDLE/grads_f64_043.pt" /home/ttuser/of3-weights/of3-p2-155k.pt \
            /home/ttuser/of3t_twoside/ctrl_f64.pt "$O/split_zonly.pt"

BLOCKS=${BLOCKS:-47}
THREADS=${THREADS:-14}
export OMP_NUM_THREADS=$THREADS
echo "=== graphdrive start $(date -u +%FT%TZ) host $(hostname) threads $THREADS blocks $BLOCKS ==="
nice -n 5 "$PY" perf/of3t_modelframe/capture_model_frame.py \
  --batch "$REF_BUNDLE/batch_step003.pt" \
  --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
  --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
  --replay-draws "$REF_BUNDLE/draws_recycles0.pt" \
  --ref-grads "$REF_BUNDLE/grads_f64_043.pt" \
  --expect-loss 1.267624369070698 \
  --graphdrive "$BLOCKS" \
  --graphdrive-injected /home/ttuser/of3t_twoside/ctrl_f64.pt \
  --graphdrive-zonly "$O/split_zonly.pt" \
  --out-dir "$O" --threads "$THREADS" --tag graphdrive_n384 2>&1 \
  | grep -vE "UserWarning|warnings.warn|^  from openfold3|Consider using tensor.detach"
echo "=== graphdrive exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
