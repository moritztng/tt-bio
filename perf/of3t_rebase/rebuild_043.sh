#!/usr/bin/env bash
# Rebuild BUNDLE-MIN at upstream 0.4.3, the revision of3-p2-155k.pt belongs to (D23/R126).
#
# Same batch, same replayed draws, same r = 0, same float64 as the published 0.5.0 bundle, so the
# ONLY thing that moves between the two artifacts is the upstream revision. Run A and run B are
# two fresh processes of the identical command: PROTOCOL A13, the check that a reference has been
# reproduced rather than merely measured.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R=${OF3T_REBASE_RUN:-$HOME/of3t_rebase_run}
B=/home/ttuser/of3t/bundle_min
PY=/home/ttuser/tt-bio-dev/env/bin/python
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS")"
ref_assert "$PY"
mkdir -p "$R/run"
export OMP_NUM_THREADS=14
BATCH_SHA=3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f
cd "$W"

for run in A B; do
  echo "=== run $run  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_reference/bundle_min.py \
      --batch "$B/batch_step003.pt" --batch-sha256 "$BATCH_SHA" \
      --out "$R/run/out_043_$run" --dtype float64 --num-recycles 0 \
      --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
      --replay-draws "$B/draws_recycles0.pt" --fd-samples 0
  echo "=== run $run exit $? $(date -u +%FT%TZ) ==="
done

echo "=== PROTOCOL A13: is it reproduced? $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_reference/compare_grads.py \
    "$R/run/out_043_A/grads_f64.pt" "$R/run/out_043_B/grads_f64.pt" \
    --json-out "$R/run/reproduction_A13_043.json"
echo "=== A13 exit $? ==="

echo "=== finite-difference validation, run C $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_reference/bundle_min.py \
    --batch "$B/batch_step003.pt" --batch-sha256 "$BATCH_SHA" \
    --out "$R/run/out_043_C_fd" --dtype float64 --num-recycles 0 \
    --checkpoint /home/ttuser/of3-weights/of3-p2-155k.pt \
    --replay-draws "$B/draws_recycles0.pt" --fd-samples 8 --fd-h 1e-4
echo "=== run C exit $? $(date -u +%FT%TZ) ==="
echo "ALLDONE $(date -u +%FT%TZ)"
