#!/bin/bash
# Real-step (r4t-step trunk_step, full network, crop 256) DP points inside the r4t-scale-dp2 window on cards 0,1:
# 2 chips replicated, then 1 chip on card 0 under the same host load. Out ~/r4t/scale/trunk{2,1}/, flag lines in dp2.done.
. $HOME/.coworker/wt/r4t-scale/perf/r4t_scale/trunk_env.sh
O=$HOME/r4t/scale
TT_BIO_LEASE_CARDS=0,1 timeout 1200 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $B --exempt perf/r4t_step/g2_exempt.tsv \
  --steps 10 --dp-cards 0,1 --out $O/trunk2 > $O/trunk2.driver.log 2>&1; echo "trunk2 rc=$?" >> $O/dp2.done
mkdir -p $O/trunk1
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 timeout 900 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $B \
  --exempt perf/r4t_step/g2_exempt.tsv --steps 10 --out $O/trunk1/ck > $O/trunk1/A.log 2>&1; echo "trunk1 rc=$?" >> $O/dp2.done
$PY perf/r4t_scale/trunk_curve.py $O/trunk1 $O/trunk2 > $O/trunk_curve.json 2>&1
