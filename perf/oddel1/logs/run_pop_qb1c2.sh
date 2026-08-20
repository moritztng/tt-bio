#!/bin/sh
# Defect B localization (state doc 6.1) on qb1 card 2. No benchlock: this run attributes a 63.6 s
# interval delta between call populations inside ONE fold, so 1-10 % co-tenant noise cannot flip it.
WT=/home/ttuser/.coworker/wt/opendde-size-generality-l1-work-split
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:opendde-size-generality-l1-work-split
export PYTHONPATH=$WT
export TT_BIO_AB_TRIMUL_POP=1
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py \
  --model opendde --sizes 640,768 --arms on \
  --out perf/oddel1/pop_640_768_qb1c2.json
