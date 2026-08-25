#!/usr/bin/env bash
# Perf legs for the RF3 fused-HiFi arm. Each timed leg runs under benchlock, one at a time.
set -u
WT=/home/ttuser/.coworker/wt/rf3-fused-hifi-precision-arm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP=$WT:/home/ttuser/rf3_perf_deps
BL=/home/ttuser/.coworker/scripts/benchlock.sh
R=$WT/perf/rf3/results
L=$R/logs
cd $WT
LEASE_ENV="TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-fused-hifi-precision-arm"

run() { # run <owner> <logname> <cmd...>
  local owner=$1 log=$2; shift 2
  echo "=== $(date -Is) START $log" >> $L/legs.log
  bash $BL "$owner" -- env PYTHONPATH=$PP $LEASE_ENV "$@" > $L/$log.log 2>&1
  echo "=== $(date -Is) END $log rc=$?" >> $L/legs.log
}

# 1. the page-512 protocol: the number comparable to the committed 49.29 s. A/A control first.
run rf3-hifi-page512 page512_a1_p1 $PY -u perf/rf3/page512_tt.py --arm a1 --repeat 2 --label aa-control-p1 --out $R/page512_a1_p1.json
run rf3-hifi-page512 page512_a1_p2 $PY -u perf/rf3/page512_tt.py --arm a1 --repeat 2 --label aa-control-p2 --out $R/page512_a1_p2.json
run rf3-hifi-page512 page512_a5_p1 $PY -u perf/rf3/page512_tt.py --arm a5 --repeat 2 --label hifi-p1 --out $R/page512_a5_p1.json
run rf3-hifi-page512 page512_a5_p2 $PY -u perf/rf3/page512_tt.py --arm a5 --repeat 2 --label hifi-p2 --out $R/page512_a5_p2.json

# 2. per-knob attribution at 512, one process, arms interleaved, L1 latches cleared between them.
run rf3-hifi-512 hifi_ckc_512 $PY -u perf/rf3/hifi_ckc_ab.py --aa 512 --arms a1,a5,a7,a8,a9 --sweeps 3 --out $R/hifi_ckc_512.json

# 3. 768: this harness IS the protocol that produced 100.95 s, so its a1 median is both A/A and cell.
run rf3-hifi-768 hifi_ckc_768 $PY -u perf/rf3/hifi_ckc_ab.py --aa 768 --arms a1,a5,a9 --sweeps 3 --out $R/hifi_ckc_768.json
echo "ALL LEGS DONE $(date -Is)" >> $L/legs.log
