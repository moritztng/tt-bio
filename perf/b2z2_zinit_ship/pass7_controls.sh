#!/bin/bash
# The Angstrom re-take on the REBASED tree. Pass 1-2's readings are from a tree 156 commits
# behind and were scored against a main digest that has since moved.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
# card 3 is idle (lsof on every /dev/tenstorrent node); card 1, this row's grant, is held by
# this row's own 44-leg gate. Grant widened on these commands only.
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=1,3 TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
export PYTHONPATH=$W
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $W
S=perf/b2z2_zinit_ship
$PY -u perf/b2z2_hostzero/control298.py --flag TT_BIO_DEVICE_ZINIT --size 298 \
  --out $W/$S/control298_p7.json --cifdir $W/$S/cif298_p7 && echo STEP1DONE
$PY -u perf/b2z2_hostzero/control298.py --flag TT_BIO_DEVICE_ZINIT --size 512 \
  --out $W/$S/control512_p7.json --cifdir $W/$S/cif512_p7 && echo STEP2DONE
$PY -u perf/b2z2_cond/fold_cond.py --flag TT_BIO_DEVICE_ZINIT --reader _device_zinit \
  --plan 'base:0,1,2,3,4,5,6,7;on:0,1,2,3,4,5,6,7' --sizes 512 \
  --out $W/$S/acc512_p7.json --cifdir $W/$S/cif_acc_p7 && echo STEP3DONE
echo ALLDONE
