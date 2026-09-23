#!/usr/bin/env bash
# of3t-stackbound: the second boundary's three whole-model references, on qb1 CPU.
#
#   refs.sh f64     float64, draws SAMPLED at --seed and recorded (draws.pt). Every other arm
#                   and the capture replay these draws, so they share one step.
#   refs.sh bf16    upstream's bf16 autocast step over fp32 parameters (the graded reference)
#   refs.sh f32     upstream's fp32 step with its own casts (the instrument floor)
#
# Same producer as bundle_min_043 (perf/of3t_reference/bundle_min.py, unmodified), same tree,
# same checkpoint, r = 0, dropout off, deterministic kernels. What differs is the batch:
# 4hhb, step 2 of the frozen stream, 384 real tokens of 384, built on pc by build_batch.py.
# w0.pt is a float64 copy of the checkpoint the producer writes unconditionally; it is deleted
# after the run to keep qb1 inside its disk, its sha256 stays in the producer's manifest.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-stackbound
cd "$W"
source perf/refpath.sh
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" /home/ttuser/of3t_frame384/deps)"
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
O=/home/ttuser/of3t_stackbound
B=$O/batch_step002.pt
B_SHA=667b7530e421eba525fbb630d925d182e4e5abd5c1d357cfaea058b9b03e54b3
SEED=20260923
ARM=${1:?usage: refs.sh f64|bf16|f32}
case "$ARM" in
  f64)  X=(--dtype float64 --autocast auto); TH=14 ;;
  bf16) X=(--dtype float32 --autocast bf16 --replay-draws "$O/ref_f64/draws.pt"); TH=7 ;;
  f32)  X=(--dtype float32 --autocast upstream --replay-draws "$O/ref_f64/draws.pt"); TH=7 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
export OMP_NUM_THREADS=$TH
S=$(date +%s)
echo "=== ref $ARM start $(date -u +%FT%TZ) host $(hostname) threads $TH seed $SEED ==="
nice -n 5 "$PY" perf/of3t_reference/bundle_min.py --batch "$B" --batch-sha256 "$B_SHA" \
  --out "$O/ref_$ARM" --seed $SEED --num-recycles 0 --fd-samples 0 \
  --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt "${X[@]}" 2>&1 \
  | grep -vE "UserWarning|warnings.warn|^  from openfold3|Consider using tensor.detach"
rc=${PIPESTATUS[0]}
rm -f "$O/ref_$ARM/w0.pt"
echo "=== ref $ARM exit $rc elapsed $(($(date +%s)-S))s $(date -u +%FT%TZ) ==="
exit $rc
