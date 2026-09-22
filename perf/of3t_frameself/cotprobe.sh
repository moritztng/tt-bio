#!/usr/bin/env bash
# of3t-frameself step 2: read the stack output cotangent with a SECOND instrument --
# torch.autograd.grad on the loss instead of a tensor hook. It stops at the stack outputs, so it
# never traverses the 48-block trunk backward and costs a forward, not a whole capture.
#
# This is of3t-modelframe/capture.sh with --selftest added and a private --out-dir. Same batch,
# same checkpoint, same draws, same expected loss and gradient norm, so the capture half of the
# run is held to the published gates before the self-test is allowed to say anything. The pair
# it writes is compared by sha256 against of3t-modelframe's published pair, which also makes
# this a qb1-to-qb2 reproducibility control the campaign has never run.
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

THREADS=${THREADS:-14}
export OMP_NUM_THREADS=$THREADS
echo "=== cotprobe start $(date -u +%FT%TZ) host $(hostname) threads $THREADS ==="
nice -n 5 "$PY" perf/of3t_modelframe/capture_model_frame.py \
  --batch "$REF_BUNDLE/batch_step003.pt" \
  --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
  --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
  --replay-draws "$REF_BUNDLE/draws_recycles0.pt" \
  --ref-grads "$REF_BUNDLE/grads_f64_043.pt" \
  --expect-loss 1.267624369070698 \
  --expect-grad-norm 3.206188185139011 \
  --cotprobe --cotprobe-compare /home/ttuser/of3t_modelframe/cot_model_n384.pt \
  --out-dir "$O" --threads "$THREADS" --tag cotprobe_n384 2>&1 \
  | grep -vE "UserWarning|warnings.warn|^  from openfold3|Consider using tensor.detach"
echo "=== cotprobe exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
ls -la "$O"
