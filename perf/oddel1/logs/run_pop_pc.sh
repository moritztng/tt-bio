#!/bin/sh
WT=/home/moritz/.coworker/wt/opendde-size-generality-l1-work-split
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:opendde-size-generality-l1-work-split
export PYTHONPATH=$WT
export TT_BIO_AB_TRIMUL_POP=1
exec /home/moritz/.coworker/scripts/benchlock.sh opendde-size-generality-l1-work-split -- \
  /home/moritz/tt-bio/env/bin/python -u perf/other512/fold_ab_multi.py \
  --model opendde --sizes 640,768 --arms on \
  --out perf/oddel1/pop_640_768_pc0.json
