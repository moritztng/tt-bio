#!/bin/bash
# The 48-replicate device-only step with the slot-ordering defect FIXED: does a warm rep hold
# 2,944 gradients and stay fast? Same argv as `of3t-p10noexact` read 56.839 s on, one card,
# rooted in this row's own worktree.
cd /home/ttuser/.coworker/wt/of3t-p10trainout || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/p10trainout
O=perf/of3t_p10trainout/out
mkdir -p "$L" "$O"
echo "=== fixed start $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg) sha $(git rev-parse --short HEAD)" >> "$L/arms.log"
env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:of3t-p10trainout \
    $PY perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 4 --samples 48 --chunk 4 \
    --reps 2 --no-exact --loss-shape model --out $O/step_noexact_fixed_48_384.json > "$L/fixed.log" 2>&1
echo "=== fixed done  $(date -u +%FT%TZ) rc=$?" >> "$L/arms.log"
