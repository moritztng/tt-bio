#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/b2z2-conf-device-ship
cd "$WT"
export PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:b2z2-conf-device-ship
exec /home/ttuser/.coworker/scripts/benchlock.sh b2z2-conf-device-ship -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_hostzero/control298.py \
  --size 512 --reps 4 \
  --flag TT_BIO_DEVICE_CONFIDENCE,TT_BIO_DEVICE_CONF_HEADS --arm conf \
  --out perf/b2z2_confship/cell_512_qb2_c0.json \
  --cifdir perf/b2z2_confship/cif512
