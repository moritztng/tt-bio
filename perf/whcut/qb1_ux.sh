#!/bin/bash
# The two release-gate legs the cutover still owed, both on qb1 where the OpenFold3
# checkpoint actually exists (~/of3-weights/of3-p2-155k.pt). pcs card 0 is still held by
# the release gate running on main, and stacking on a busy box is what cost §4.2 two legs.
#
# 1. ux_regression, which the plan called for "for completeness, not as a blocker": K3
#    changes no CLI surface, so this is expected to be unaffected.
# 2. the openfold3 perf leg. It FAILED in the first §4.2 run for a reason that had nothing
#    to do with the assembly: perf_regression ran without OF3_CKPT set. With the checkpoint
#    pointed at, it should measure, which turns §4.2 from 12-plus-an-asterisk into 13.
set -u
TREE=/home/ttuser/.coworker/wt/japanfold-wh-cutover
PY=/home/ttuser/tt-bio/env/bin/python3
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
cd "$TREE" || exit 1
echo "UX+OF3 START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"

env TT_VISIBLE_DEVICES=1 TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$TREE" \
    TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/perf_regression.py --model openfold3 > perf/whcut/out/qb1_of3_perf.log 2>&1
echo "OF3 PERF EXIT $?"

env TT_VISIBLE_DEVICES=0 TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$TREE" \
    TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/ux_regression.py > perf/whcut/out/qb1_ux.log 2>&1
echo "UX EXIT $?"
echo "UX+OF3 DONE $(date -u +%FT%TZ)"
