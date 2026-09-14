#!/bin/bash
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
export PYTHONPATH=$W
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $W
$PY -u perf/b2z2_hostzero/control298.py --flag TT_BIO_DEVICE_ZINIT --size 512 \
  --out $W/perf/b2z2_zinit_ship/control512_v2.json --cifdir $W/perf/b2z2_zinit_ship/cif512_v2
$PY -u perf/b2z2_hostzero/control298.py --flag TT_BIO_DEVICE_ZINIT --size 298 \
  --out $W/perf/b2z2_zinit_ship/control298_v2.json --cifdir $W/perf/b2z2_zinit_ship/cif298_v2
echo ALLDONE
