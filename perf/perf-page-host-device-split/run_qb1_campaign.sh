#!/bin/sh
# The p150a side of the perf-page board audit: the same split harness the published qb2
# numbers came from, run on tt-quietbox's real p150a board (0x0040, 13x10 grid). Card 2 is
# the only free card on that box. Cheapest model first, so a short window still lands whole
# models rather than half of the most expensive one.
cd /home/ttuser/.coworker/wt/perf-page-p150a-column-audit || exit 1
export BENCHLOCK_LOAD_WAIT_S=2400
CARD=2 TAG=qb1c2 HOLDER=perf-page-p150a-column-audit \
  sh perf/perf-page-host-device-split/run_all.sh \
     boltz2 esmfold2 openfold3 protenix-v2 opendde
echo "########## rf3 ##########"
~/.coworker/scripts/benchlock.sh perf-page-p150a-column-audit -- \
  env PYTHONPATH="/home/ttuser/.coworker/wt/perf-page-p150a-column-audit:/home/ttuser/rf3_perf_deps" \
      TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
      TT_BIO_LEASE_HOLDER=worker:perf-page-p150a-column-audit \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/rf3/page512_tt.py \
      --repeat 3 --arm a0 --label qb1c2 \
      --out perf/rf3/page512/shipped_qb1c2.json
echo "########## rf3 rc=$? ##########"
echo "CAMPAIGN-COMPLETE $(date -Is)"
