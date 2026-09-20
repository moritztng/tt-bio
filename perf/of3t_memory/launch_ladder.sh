#!/bin/bash
cd /home/ttuser/.coworker/wt/of3t-memory || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-memory
export PYTHONPATH=/home/ttuser/.coworker/wt/of3t-memory
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_memory/alloc_profile.py \
  --sizes 384,640,768 --blocks 1 --arms boundary,off,keep \
  --out perf/of3t_memory/out/ladder_1blk_v2.json
