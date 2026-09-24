#!/bin/bash
# one ladder rung, detached, rooted in this worktree
N=$1; REPS=${2:-1}; LIM=${3:-9000}
cd /home/ttuser/.coworker/wt/bcx-large
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-large
exec timeout -s INT $LIM python3 perf/bcx_large/ladder.py --n $N --reps $REPS ${EXTRA_ARGS:-}
