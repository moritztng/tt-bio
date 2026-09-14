#!/bin/bash
# Steps 3 and 4 moved to card 2. Card 3 silently stalled TWICE this pass -- a 512 aa fold that
# normally takes 25 s sat at 120 % CPU with no output for 5 min and 12 min respectively. Card 1
# is this row's gate, card 0 a co-tenant, so card 2 is the free one.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=1,2 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
export PYTHONPATH=$W
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $W
S=perf/b2z2_zinit_ship
$PY -u perf/b2z2_cond/fold_cond.py --flag TT_BIO_DEVICE_ZINIT --reader _device_zinit \
  --plan 'base:0,1,2,3,4,5,6,7;on:0,1,2,3,4,5,6,7' --sizes 512 \
  --out $W/$S/acc512_p7.json --cifdir $W/$S/cif_acc_p7 && echo STEP3DONE
# main's OWN shipped conditioning on the same column: the calibration the verdict rests on.
$PY -u perf/b2z2_hostzero/control298.py --flag TT_BIO_DEVICE_CONDITIONING --size 512 \
  --out $W/$S/calib_cond512_p7.json --cifdir $W/$S/cif_calib_cond_p7 && echo CALIBDONE
echo ALLDONE
