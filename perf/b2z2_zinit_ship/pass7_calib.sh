#!/bin/bash
# The calibration the 512 aa verdict rests on, re-measured on the REBASED tree rather than
# quoted from pass 2: how far does main's OWN shipped, already-accepted TT_BIO_DEVICE_CONDITIONING
# move the same pseudo-domain column on the same fixture? If a shipped lever moves it as far as
# this one does, the column is not an instrument that can carry a 0.60 A bar.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=1,3 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
export PYTHONPATH=$W
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $W
S=perf/b2z2_zinit_ship
# base = conditioning OFF (host torch pair track), on = conditioning ON = main's shipped default.
$PY -u perf/b2z2_hostzero/control298.py --flag TT_BIO_DEVICE_CONDITIONING --size 512 \
  --out $W/$S/calib_cond512_p7.json --cifdir $W/$S/cif_calib_cond_p7 && echo CALIBDONE
echo ALLDONE
