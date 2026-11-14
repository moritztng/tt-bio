#!/bin/sh
# Existing users unmoved: Protenix-v2 (normal) and OpenDDE through the serving path on this branch, resolvers instrumented.
W=/home/ttuser/.coworker/wt/mol1-perf-triattn
O=$W/perf/mol1_triattn/inert
mkdir -p $O; cd $W; export PYTHONPATH=$W
for M in protenix-v2 opendde; do
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:mol1-perf-triattn timeout 2400 \
    /home/ttuser/tt-bio-dev/env/bin/python3 perf/mol1_triattn/inert.py $O/$M.hits.json --out $O/$M --chip 2 --model $M \
    --arm normal --inputs l256,l512,l1024 --warm 1 --samples 1 > $O/$M.log 2>&1
  echo "rc=$?" >> $O/$M.log
done
echo ALLDONE > $O/DONE
