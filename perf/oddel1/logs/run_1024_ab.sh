#!/bin/sh
cd /home/moritz/.coworker/wt/opendde-size-generality-l1-work-split || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:opendde-size-generality-l1-work-split
export PYTHONPATH=/home/moritz/.coworker/wt/opendde-size-generality-l1-work-split
exec /home/moritz/tt-bio/env/bin/python -u perf/other512/fold_ab_multi.py \
  --model opendde --sizes 1024 --arms on,qpercore,on,qpercore \
  --out perf/oddel1/fold_ab_1024_pc0.json
