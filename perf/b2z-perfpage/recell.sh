#!/usr/bin/env bash
# b2z-perfpage-recell: re-measure the published Boltz-2 Blackhole 512 aa cell on the merged tree
# (0f3f9f67), in ONE session, under the published protocol, with the 298 aa control alongside.
set -eu
WT=/home/ttuser/.coworker/wt/b2z-perfpage-recell
cd "$WT"
exec ~/.coworker/scripts/benchlock.sh worker:b2z-perfpage-recell -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2z-perfpage-recell \
  PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z_levers/fold_ab.py \
    --out perf/b2z-perfpage/recell_512_qb2c0.json \
    --cifdir perf/b2z-perfpage/cif \
    --reps 6 --warmup --sizes 512,298
