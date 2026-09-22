#!/usr/bin/env bash
# The blocks the ladder skipped (42..46), because the weight census says the anomaly lives there.
#
# The seven-boundary ladder sampled 40 and then 47. The all-48 weight census found the
# distinguishing variable is a SMOOTH SIX-BLOCK RAMP from 42 to 47 -- single-track weight norm^2
# goes 1.76x, 2.60x, 3.65x, 5.69x, 6.62x, 5.08x of the 48-block median -- so the entire ramp sits
# inside the ladder's one gap, and "the factor is a property of the last block" is an inference
# from a sampling hole.
#
# Registered prediction, before this runs:
#   weights story  -> blocks 43..46 carry graded factors and BLOCK 46 IS WORSE THAN BLOCK 47 on
#                     the single track (further from 1.0 than 0.8867), since 46 holds the largest
#                     single-track weight anomaly in the model at 6.62x median.
#   position story -> blocks 42..46 all read ~1.00 on both tracks, exactly like block 40, and
#                     only block 47 carries a factor.
#
# Block 46 is also the only one of these with a pair-track reference gradient of the same order
# as block 47's (4.1100e-02 against 6.6141e-02) and a single track four orders above the middle
# of the stack, so both of its tracks are actually readable.

set -uo pipefail
W=/home/ttuser/of3t_rebase/wt
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:/home/ttuser/of3t_gradients/pylibs"
export OMP_NUM_THREADS=8
PY=/home/ttuser/tt-bio-dev/env/bin/python
B=/home/ttuser/of3t_rebase/bundle_min_043
echo "=== weight-ramp capture (blocks 42..46) at 0.4.3  $(date -u +%FT%TZ) ==="
"$PY" perf/of3t_gradients/capture_trunk_boundary.py \
    --bundle "$B" --manifest-json "$B/MANIFEST.json" --grads grads_f64_043.pt \
    --blocks 42,43,44,45,46 --no-dropout \
    --out /home/ttuser/of3t_rebase/cap043_ramp \
    --report perf/of3t_rebase/capture_trunk_boundary_043_ramp.json
echo "=== capture exit $? $(date -u +%FT%TZ) ==="
echo "CAPRAMP_ALLDONE $(date -u +%FT%TZ)"
