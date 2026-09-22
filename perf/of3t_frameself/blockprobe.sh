#!/usr/bin/env bash
# of3t-frameself step 3: read the REFERENCE's own trunk gradient with a second instrument.
#
# torch.autograd.grad prunes the graph to the parameters it is asked for, so asking for the last
# blocks costs those blocks' backward and not the whole 48. Same batch, same checkpoint, same
# draws as the capture, and the loss is still gated bit-identical to 1.267624369070698 before the
# probe is allowed to say anything. No full backward, so no gradient-norm gate and no witness.
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
            /home/ttuser/of3t_twoside/ctrl_f64.pt

BLOCKS=${BLOCKS:-47,46}
THREADS=${THREADS:-14}
export OMP_NUM_THREADS=$THREADS
echo "=== blockprobe start $(date -u +%FT%TZ) host $(hostname) threads $THREADS blocks $BLOCKS ==="
nice -n 5 "$PY" perf/of3t_modelframe/capture_model_frame.py \
  --batch "$REF_BUNDLE/batch_step003.pt" \
  --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
  --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
  --replay-draws "$REF_BUNDLE/draws_recycles0.pt" \
  --ref-grads "$REF_BUNDLE/grads_f64_043.pt" \
  --expect-loss 1.267624369070698 \
  --blockprobe "$BLOCKS" \
  --blockprobe-injected /home/ttuser/of3t_twoside/ctrl_f64.pt \
  --out-dir "$O" --threads "$THREADS" --tag ${TAG:-blockprobe_n384} 2>&1 \
  | grep -vE "UserWarning|warnings.warn|^  from openfold3|Consider using tensor.detach"
echo "=== blockprobe exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
