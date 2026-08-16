#!/usr/bin/env bash
# Blackhole neutrality for the Transition L1 cap: A = b1a3fe61d (the branch point on main),
# B = wk/wh-perf-protenix-v2. _IS_SMALL_GRID is False at 11x10, so the expected result is
# byte-identical CIFs, not merely walls inside the A/A floor.
set -eu
WT=/home/ttuser/.coworker/wt
XMODEL_PY=/home/ttuser/tt-bio-dev/env/bin/python3 \
OWNER=worker:wh-perf-protenix-v2 CARD=3 \
  $WT/wh-perf-protenix-v2/perf/of3_4xpd/run_xmodel_ab.sh protenix-v2 \
    $WT/wh-perf-protenix-v2-bhmain $WT/wh-perf-protenix-v2 \
    $WT/wh-perf-protenix-v2/perf/wh-protenix/results/bh_neutral_qb2c3 3
echo BHDONE > /home/ttuser/.coworker/wt/wh-perf-protenix-v2/perf/wh-protenix/results/bh_neutral.done
