#!/bin/bash
# bgx-traj: the two fixed-default arms, then the release gate, on qb2 card 0.
#
# Detached and chained rather than run side by side: they share one card. The arms are the
# on-card proof that the size-aware default picks 1 at 704 tokens and 2 at 512 and that both
# complete; the gate is what the fix has to clear to land on main.
set -u
cd /home/ttuser/.coworker/wt/bgx-traj
while pgrep -f "arm.sh fix" > /dev/null; do sleep 20; done
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bgx-traj
mkdir -p perf/bgx_traj/gate
exec timeout 21600 /home/ttuser/tt-bio-dev/env/bin/python scripts/release_gate.py \
    --keep --load-ceiling 10 \
    --journal /home/ttuser/.coworker/wt/bgx-traj/perf/bgx_traj/gate/journal.jsonl
