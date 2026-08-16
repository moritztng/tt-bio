#!/usr/bin/env bash
# Round 2 of the Blackhole neutrality A/B, same card 3, order B A B A. Round 1 left B1 at
# 51.154 s with a 0.475 s within-process spread under loadavg 4.26, an order of magnitude
# above every other leg (0.017-0.048 s), so that leg is contended and not usable as the B
# arm. Four more clean legs settle it. B first this round so B does not always follow A.
set -eu
WT=/home/ttuser/.coworker/wt
XMODEL_PY=/home/ttuser/tt-bio-dev/env/bin/python3 \
OWNER=worker:wh-perf-protenix-v2 CARD=3 \
  $WT/wh-perf-protenix-v2/perf/of3_4xpd/run_xmodel_ab.sh protenix-v2 \
    $WT/wh-perf-protenix-v2 $WT/wh-perf-protenix-v2-bhmain \
    $WT/wh-perf-protenix-v2/perf/wh-protenix/results/bh_neutral_qb2c3_r2 3
echo BHDONE2 > $WT/wh-perf-protenix-v2/perf/wh-protenix/results/bh_neutral_r2.done
