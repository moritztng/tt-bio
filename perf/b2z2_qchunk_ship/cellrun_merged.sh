#!/usr/bin/env bash
set -u
cd /home/ttuser/.coworker/wt/b2z2-qchunk-ship
export PYTHONPATH=/home/ttuser/.coworker/wt/b2z2-qchunk-ship
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:b2z2-qchunk-ship
exec /home/ttuser/.coworker/scripts/benchlock.sh b2z2-qchunk-ship -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_qchunk/qchunk_bh.py \
  --out perf/b2z2_qchunk_ship/cell_512_qb2_c0_merged.json \
  --cifdir perf/b2z2_qchunk_ship/cif_merged --reps 7 --op-reps 20
