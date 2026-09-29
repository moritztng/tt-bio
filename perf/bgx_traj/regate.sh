#!/bin/bash
# bgx-traj re-gate at the merge tree, qb2 card 0: the shipped default end-to-end at 544 and 288
# tokens, then two release-gate arms scoring THIS worktree (PYTHONPATH), protenix-v1 first so it
# is the first fold on the card after a BindCraft 2 run exited (the F5 question).
set -u
cd /home/ttuser/.coworker/wt/bgx-traj
O=perf/bgx_traj/out
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bgx-traj
echo "=== start $(date -u +%H:%M:%SZ) $(git rev-parse --short HEAD)"
timeout 1500 bash perf/bgx_traj/arm.sh fix544auto 415 auto 3 > $O/fix544auto.log 2>&1; echo "=== fix544auto rc=$? $(date -u +%H:%M:%SZ)"
timeout 1500 bash perf/bgx_traj/arm.sh fix288auto 146 auto 3 > $O/fix288auto.log 2>&1; echo "=== fix288auto rc=$? $(date -u +%H:%M:%SZ)"
export PYTHONPATH=$PWD
mkdir -p perf/bgx_traj/regate
for m in protenix-v1 boltz2; do
  timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python -u scripts/release_gate.py --model $m --keep \
    --journal $PWD/perf/bgx_traj/regate/journal.jsonl > $O/regate_$m.log 2>&1
  echo "=== gate $m rc=$? $(date -u +%H:%M:%SZ)"
done
echo "=== end $(date -u +%H:%M:%SZ)"
