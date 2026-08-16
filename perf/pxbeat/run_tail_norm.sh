#!/usr/bin/env bash
WT=/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200
cd "$WT" || exit 1
BENCHLOCK_WAIT_S=1800 /home/ttuser/.coworker/scripts/benchlock.sh worker:protenix-v2-beat-dgx-h200 -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-v2-beat-dgx-h200 PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/pxbeat/tail_norm_screen.py \
  --out perf/pxbeat/tail_norm_512_c1.json
