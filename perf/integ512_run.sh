#!/bin/bash
# The integrated A/B: current main vs everything landed, one process, alternating arms, at 512 aa.
set -u
WT=/home/ttuser/.coworker/wt/integrated-ab-h200-gap
cd "$WT" || exit 1
mkdir -p perf/integ512
rm -f perf/integ512/DONE
exec > perf/integ512/fold_ab_integ512.log 2>&1
echo "host=$(hostname) start=$(date -Is) commit=$(git rev-parse --short HEAD)"
uptime
/home/ttuser/.coworker/scripts/benchlock.sh integrated-ab-h200-gap -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:integrated-ab-h200-gap PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/size512/fold_ab512.py \
    --sizes 512 \
    --arms main,int,main,int,int,int_noe6,int_nok1k2,int_notr,int_noe6,int_nok1k2,int_notr \
    --out "$WT/perf/integ512/fold_ab_integ512.json"
echo "EXIT=$? end=$(date -Is)"
touch perf/integ512/DONE
