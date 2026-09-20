#!/usr/bin/env bash
set -u
cd /home/ttuser/.coworker/wt/land-standing
exec ~/.coworker/scripts/benchlock.sh land-standing -- \
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:land-standing \
      TT_BIO_AICLK=1350 \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/c14_bfp8/fold_ab.py \
      --sizes 512 --blocks 5 --folds 3 --card 2 --quiet-wait 900 \
      --out perf/c14_bfp8/region_t_ab_qb2_main.json \
      --cifdir perf/c14_bfp8/region_t_qb2_cifs
