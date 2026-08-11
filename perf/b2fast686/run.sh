#!/usr/bin/env bash
# The boltz2 --fast 686 aa A/B, qb2 card physical 1. Wrapped in benchlock: a co-tenanted timing is
# a wrong number, not a slow one.
set -u
cd "$(dirname "$0")/../.."
WT=$PWD
{
  echo "host=$(hostname) start=$(date -Is) commit=$(git rev-parse --short HEAD)"
  uptime
} >> perf/b2fast686/ab686.log
exec /home/ttuser/.coworker/scripts/benchlock.sh boltz2-686-fast-integrated -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:boltz2-686-fast-integrated \
      PYTHONPATH="$WT" \
  /home/ttuser/tt-bio/env/bin/python3 perf/b2fast686/fold_ab_686fast.py \
      --arms main,int,main,int,int,int_noe6,int_nok1k2,int_notr,int_noe6,int_nok1k2,int_notr \
      --out perf/b2fast686/ab686.json
