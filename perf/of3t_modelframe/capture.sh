#!/usr/bin/env bash
# of3t-modelframe step 1+2: the trunk's real boundary and real incoming cotangent, from the
# reference's own float64 backward on batch_step003. One run produces both -- they are two
# sides of the same tape and taking them in separate processes would let them drift.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelframe
cd "$W"
source perf/refpath.sh
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" /home/ttuser/of3t_frame384/deps)"
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
O=/home/ttuser/of3t_modelframe
mkdir -p "$O"
ref_require "$REF_BUNDLE/batch_step003.pt" "$REF_BUNDLE/draws_recycles0.pt" \
            "$REF_BUNDLE/grads_f64_043.pt" /home/ttuser/of3-weights/of3-p2-155k.pt

THREADS=${THREADS:-14}
export OMP_NUM_THREADS=$THREADS
echo "=== capture start $(date -u +%FT%TZ) host $(hostname) threads $THREADS ==="
nice -n 5 "$PY" perf/of3t_modelframe/capture_model_frame.py \
  --batch "$REF_BUNDLE/batch_step003.pt" \
  --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
  --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
  --replay-draws "$REF_BUNDLE/draws_recycles0.pt" \
  --ref-grads "$REF_BUNDLE/grads_f64_043.pt" \
  --expect-loss 1.267624369070698 \
  --expect-grad-norm 3.206188185139011 \
  --out-dir "$O" --threads "$THREADS" --tag model_n384 2>&1 \
  | grep -vE "UserWarning|warnings.warn|^  from openfold3|Consider using tensor.detach"
echo "=== capture exit ${PIPESTATUS[0]} $(date -u +%FT%TZ) ==="
ls -la "$O"
