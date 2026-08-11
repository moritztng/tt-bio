#!/bin/bash
# Capacity check: does the integrated build allocate worse than main at the size class where the
# L1 clash lives (>640 tokens)? Same two arms, 640 aa fixture.
set -u
WT=/home/ttuser/.coworker/wt/integrated-ab-h200-gap
cd "$WT" || exit 1
mkdir -p perf/integ512
rm -f perf/integ512/DONE640
exec > perf/integ512/cap640.log 2>&1
echo "host=$(hostname) start=$(date -Is) commit=$(git rev-parse --short HEAD)"
uptime
/home/ttuser/.coworker/scripts/benchlock.sh integrated-ab-h200-gap-cap640 -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:integrated-ab-h200-gap PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/size512/fold_ab512.py \
    --sizes 640 --arms main,int,main,int \
    --out "$WT/perf/integ512/cap640.json"
echo "EXIT=$? end=$(date -Is)"
touch perf/integ512/DONE640
