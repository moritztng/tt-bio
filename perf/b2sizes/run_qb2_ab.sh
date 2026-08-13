#!/bin/bash
# The qb2 leg of the triatt q-split (TT_BIO_TRIATT_MASK_Q_SPLIT) A/B at 768 and 1024 aa.
#
# Why this run exists: every number behind the shipped `_Q_SPLIT_MAX_S = 1024` (commit 063f89db)
# was measured on qb1 -- 130-core p150a, ttnn 0.67.4. qb2 is 110 cores and ttnn 0.68.0, and the
# lever's whole mechanism is a per-core split, so the grid is exactly the variable that could
# invert it. This is a post-merge validation on the second host, not a lever screen.
#
# Arm meaning on current main: `on` pins PM._Q_SPLIT = False, i.e. main BEFORE 063f89db.
# `qsplit` pins it True, i.e. main's shipped default at these sizes. Reference arms run first;
# set_arm clears _SDPA_Q_CHUNK_OVER_L1 / _PM_OVER_L1 per arm (the e3b0b95b class of fix).
# 1024 runs first: it is the size that owes a three-sample reference floor.
WT=/home/ttuser/.coworker/wt/boltz2-qb2-768-1024-clean-ab
cd "$WT" || exit 70
PY=/home/ttuser/tt-bio-dev/env/bin/python3
E="env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:boltz2-qb2-768-1024-clean-ab PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm $PY -u"

echo "=== 1024 aa $(date -Is) ==="
$E perf/other512/fold_ab_multi.py --model boltz2 --sizes 1024 \
   --arms on,on,on,qsplit,qsplit --out perf/b2sizes/qsplit_ab_1024_qb2.json
echo "1024 RC=$?"

echo "=== 768 aa $(date -Is) ==="
$E perf/other512/fold_ab_multi.py --model boltz2 --sizes 768 \
   --arms on,on,on,qsplit --out perf/b2sizes/qsplit_ab_768_qb2.json
echo "768 RC=$?"
echo "=== done $(date -Is) ==="
