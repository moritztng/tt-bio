#!/bin/bash
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
export PYTHONPATH=$W
cd $W
/home/ttuser/tt-bio-dev/env/bin/python3 -u perf/b2z2_cond/fold_cond.py \
  --flag TT_BIO_DEVICE_ZINIT --reader _device_zinit \
  --plan "base:0,1,2,3;on:0,1,2,3" --sizes 512 \
  --out $W/perf/b2z2_zinit_ship/acc_512.json --cifdir $W/perf/b2z2_zinit_ship/cif_acc
echo ACCDONE
