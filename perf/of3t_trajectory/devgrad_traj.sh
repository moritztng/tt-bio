#!/usr/bin/env bash
# Our device gradient over all 48 structures at the 0.4.3 boundary, and this time the TENSORS,
# not only the summary statistics: without them the gradient can only ever be compared to the
# one reference it was run against. Everything else is devgrad043.sh verbatim, so the run
# reproduces the 547-tensor arm the record already carries.
#
#   devgrad_traj.sh            the real run
#   devgrad_traj.sh permcot    the break control, every sample seeded with the wrong cotangent
#   devgrad_traj.sh sm64       AMENDMENT 3's bound, every softmax on the host in float64
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_tape:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-trajectory
PY=/home/ttuser/tt-bio-dev/env/bin/python
OUT=/home/ttuser/of3t_trajectory
mkdir -p "$OUT"
case "${1:-}" in
  permcot) TAG=_trajpermcot;  PT=_permcot;  EXTRA=--permute-cot ;;
  sm64)    TAG=_trajsm64;     PT=_sm64;     EXTRA=--softmax-f64 ;;
  *)       TAG=_traj;         PT=;          EXTRA= ;;
esac
echo "=== device gradient, 48 structures, tag $TAG, card $CARD  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_diffusion/device_gradient.py --structs all --tag "$TAG" \
    --cap /home/ttuser/of3t_rebase/diffcap043 \
    --out-dir perf/of3t_trajectory \
    --dump-per-tensor $EXTRA \
    --dump-grads "$OUT/device_grads_043all$PT.pt"
echo "=== device gradient exit $? $(date -u +%FT%TZ) ==="
echo "DEVGRAD_TRAJ_DONE $(date -u +%FT%TZ)"
